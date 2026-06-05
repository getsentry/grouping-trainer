"""
NOTE: this script is vibe-coded.

Score test pairs with per-project BM25 and save similarities to GCS, mirroring eval/save_embeddings.py.

Output schema is a drop-in `gcs_model2` for `eval.compare`: a `similarities.csv` with a single `cos_sim_1` column
holding the symmetrized BM25 score for each pair. The "1" is a placeholder — BM25 has no MRL dim concept; pass
`--dim_model2 1` to `eval.compare`.

IDF is computed per-project to mirror prod's per-project vector DB lookup. Each pair's score is averaged across both
orderings (query→candidate and candidate→query) so the result is independent of which side was the "query."

Caveats on what to expect from `eval.compare` on this baseline:

- BM25 looked weak on a 3000-pair smoke test (median per-project AUC ~0.64; per-platform AUC < 0.5 for go, php,
  csharp). The boundary-sampled data over-represents pairs where lexical similarity disagrees with the GROUP/SEPARATE
  label (e.g., templated Go errors like ``failed to generate unique username: name=<varies>`` — same semantic, very
  different lexical). Per-platform thresholds in `eval.compare` can absorb scale differences but not signal inversion.
- Switching to per-platform or global IDF moved global AUC by ~0.02 — within noise; not worth the memory cost at full
  scale (a global N x N score matrix is GB-sized).
- camelCase/code-aware tokenization, BM25 b-parameter tuning, and key=value masking all left AUC essentially
  unchanged. Don't bother engineering these unless you have a specific failure case in mind.

Install the optional dep:

    uv sync --extra eval-bm25

Full run:

python eval/save_bm25_scores.py \
    --run_gcs_dir gs://$GROUPING_TRAINER_BUCKET/runs/bm25

BM25 scores are not in [0, 1], so the right threshold needs to be swept rather than copied from a cos_sim model:

python -m eval.compare \
    --gcs_model1 gs://$GROUPING_TRAINER_BUCKET/runs/v1/similarities/test_full3 \
    --gcs_model2 gs://$GROUPING_TRAINER_BUCKET/runs/bm25/similarities/test_full3 \
    --threshold_model1 0.99 \
    --threshold_model2 <swept-value> \
    --dim_model2 1
"""

import logging
import os.path
import subprocess
import tempfile
import time

import bm25s
import numpy as np
import polars as pl
from tap import tapify
from tqdm.auto import tqdm

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def _score_project(texts_query: list[str], texts_candidate: list[str]) -> np.ndarray:
    """Score each (query, candidate) pair within a single project using symmetrized BM25.

    The corpus is the set of unique stacktraces appearing on either side of any pair in the project. IDF and average
    document length are derived from that corpus. Returned score for pair i is
    ``0.5 * (BM25(query_i, candidate_i) + BM25(candidate_i, query_i))`` — averaging both orderings makes the score
    order-independent, which matches how this dataset's pairs are unordered.
    """
    texts_unique = sorted({*texts_query, *texts_candidate})
    idx_by_text = {text: idx for idx, text in enumerate(texts_unique)}
    num_unique = len(texts_unique)

    # stopwords=[] disables stopword removal (default is English stopwords, which is wrong for stacktraces — frames
    # contain identifiers like "in", "as", "do" that we want to keep). stemmer defaults to None.
    tokens_corpus = bm25s.tokenize(texts_unique, stopwords=[], show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens_corpus, show_progress=False)

    # Retrieve every doc for every query to materialize the full NxN score matrix. Memory is num_unique^2 floats per
    # project; this is fine while per-project unique-stacktrace counts stay in the low thousands. If a future project
    # blows past that, switch to chunked retrieval here.
    docs_ranked, scores_ranked = retriever.retrieve(tokens_corpus, k=num_unique, show_progress=False, return_as="tuple")
    scores_dense = np.zeros((num_unique, num_unique), dtype=np.float32)
    for row_idx in range(num_unique):
        scores_dense[row_idx, docs_ranked[row_idx]] = scores_ranked[row_idx]

    scores_pair = np.empty(len(texts_query), dtype=np.float32)
    for pair_idx, (text_query, text_candidate) in enumerate(zip(texts_query, texts_candidate, strict=True)):
        idx_query = idx_by_text[text_query]
        idx_candidate = idx_by_text[text_candidate]
        scores_pair[pair_idx] = 0.5 * (scores_dense[idx_query, idx_candidate] + scores_dense[idx_candidate, idx_query])
    return scores_pair


def main(
    run_gcs_dir: str,
    df_path: str = "final_csvs/test_full3.csv",
    sample_size: int | None = None,
):
    """Score test pairs with per-project BM25 and save similarities to GCS.

    Parameters
    ----------
    run_gcs_dir
        GCS path under which to write ``similarities/{name_dataset}/``. Mirrors save_embeddings.py's layout so the
        result is a drop-in ``gcs_model2`` for `eval.compare`. Example:
        ``gs://$GROUPING_TRAINER_BUCKET/runs/bm25``
    df_path
        Path to the validation/test CSV file.
    sample_size
        Number of rows to sample. None (default) uses the full dataset.
    """
    gt.logging.configure_logging(process_type="save_bm25_scores")
    logging.getLogger("bm25s").setLevel(logging.WARNING)

    run_gcs_dir = run_gcs_dir.rstrip("/")
    name_dataset = os.path.splitext(os.path.basename(df_path))[0]
    dir_gcs_output = f"{run_gcs_dir}/similarities/{name_dataset}"

    df_pairs = gt.data.load_val_df(paths=(df_path,), sample_size=sample_size)
    logger.info(f"Test df shape: {df_pairs.shape}")

    start = time.monotonic()
    df_indexed = df_pairs.with_row_index("_pair_idx")
    num_projects = df_indexed["project_id"].n_unique()
    scores_by_pair_idx: dict[int, float] = {}
    for _, df_project in tqdm(df_indexed.group_by("project_id"), desc="Scoring projects", total=num_projects):
        scores_project = _score_project(
            df_project["query_stacktrace_string"].to_list(),
            df_project["candidate_stacktrace_string"].to_list(),
        )
        for pair_idx, score in zip(df_project["_pair_idx"].to_list(), scores_project, strict=True):
            scores_by_pair_idx[pair_idx] = float(score)
    logger.info(f"Scored {len(df_pairs)} pairs in {time.monotonic() - start:.1f}s")

    scores_ordered = np.array([scores_by_pair_idx[idx] for idx in range(len(df_pairs))], dtype=np.float32)
    df_out = df_pairs.with_columns(pl.Series(name="cos_sim_1", values=scores_ordered))

    with tempfile.TemporaryDirectory() as dir_tmp_output:
        df_out.write_csv(f"{dir_tmp_output}/similarities.csv")

        logger.info(f"Uploading to {dir_gcs_output}...")
        subprocess.run(["gcloud", "storage", "rsync", "-r", dir_tmp_output, dir_gcs_output], check=True)
        logger.info(f"Uploaded to {dir_gcs_output}")


if __name__ == "__main__":
    tapify(main)
