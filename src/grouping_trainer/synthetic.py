"""
Sludge Pour Un Bébé Robot

python -m grouping_trainer.synthetic \
    --gcs_model_folder gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v1/inference \
    --csv_paths final_csvs/train_more.csv final_csvs/train_more2.csv \
    --positives --negatives

Mitigates a bias in the labeled pairs. The sampling code intentionally samples mostly around v1's decision boundary to
get the biggest bang for our buck. This bias may not be good b/c:
- The model won't see easy negatives during training that it will see while crawling the index
- The label for pairs whose v1 distance is in [0.001, 0.01) can only change from GROUP to SEPARATE, which might cause
  the trained model to over-emphasize subtle differences. Seeing easy positives should counteract this bias.

In practice, including these easier examples makes training a bit more stable, particularly when training a model that
was only MLM-pretrained. It's not a big impact. Excluding it is prolly fine if you wanna make training runs finish
earlier. I typically include them to theoretically counter the biases above. Haven't studied it much.
"""

import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from tap import tapify
from tqdm.auto import tqdm

import grouping_trainer as gt


@dataclass(frozen=True, kw_only=True)
class LabelResult:
    idx: int
    label: str
    confidence_score: float | None
    "`None` means the confidence score was not able to be parsed from `response_output`"
    response_output: str
    prompt: str
    thinking_output: str


def top_combos(
    distances: np.ndarray,
    min_distance: float,
    max_distance: float,
    sort_distances_ascending: bool,
    num_combos: int | None = None,
) -> list[tuple[int, ...]]:
    """
    Return at most `num_combos` indices with distance in `[min_distance, max_distance]`. `distances` can have any shape.

    The output of this function isn't too useful w/ a symmetric distance matrix.
    """
    if num_combos is not None and num_combos < 0:
        raise ValueError("num_combos must be positive")
    if num_combos == 0:
        return []

    flat = distances.ravel()
    mask = (min_distance <= distances) & (distances <= max_distance)

    if not np.any(mask):
        return []

    indices = np.where(mask)[0]  # in raveled space
    indices_sorted = np.argsort(flat[indices]) if sort_distances_ascending else np.argsort(flat[indices])[::-1]
    indices_selected = indices[indices_sorted]  # back to raveled space
    top_indices = indices_selected[:num_combos]  # still in raveled space
    unraveled = np.unravel_index(top_indices, distances.shape)  # finally unravel
    return list(zip(*unraveled, strict=True))  # :-]


def mine_from_distance_matrix(
    query_candidate_distances: np.ndarray,
    min_distance: float,
    max_distance: float,
    num_candidates_per_query: int,
    sort_distances_ascending: bool,
) -> list[list[int]]:
    if query_candidate_distances.ndim != 2:
        raise ValueError("query_candidate_distances must be a 2-D array")

    # Stratified across queries to avoid overrepresenting universally distant candidates
    candidate_indices_per_query = [
        top_combos(
            query_distances,
            min_distance=min_distance,
            max_distance=max_distance,
            num_combos=num_candidates_per_query,
            sort_distances_ascending=sort_distances_ascending,
        )
        for query_distances in query_candidate_distances
    ]

    # Check that we can slice the first dimension of the candidate axis
    for candidate_indices in candidate_indices_per_query:
        for candidate_index in candidate_indices:
            assert len(candidate_index) == 1

    # Polars doesn't support np.integer-slicing. Convenient to return plain ints
    return [
        [candidate_index[0] for candidate_index in candidate_indices]
        for candidate_indices in candidate_indices_per_query
    ]


def mine_semi_easy_negatives_from_distance_matrix(
    query_candidate_distances: np.ndarray,
    min_distance: float = 0.3,  # can't be any smaller w/o letting false negatives slip in
    max_distance: float = 0.5,  # not higher to avoid too many easy negatives
    num_candidates_per_query: int = 5,  # feel free to make this 20. Can always sample down after writing it
):
    """
    Returns the furthest `num_candidates_per_query` candidate indices per query matching the distance filters.
    """
    return mine_from_distance_matrix(
        query_candidate_distances,
        min_distance,
        max_distance,
        num_candidates_per_query,
        sort_distances_ascending=False,
    )


def mine_easy_positives_from_distance_matrix(
    query_candidate_distances: np.ndarray,
    min_distance: float = 0.0001,  # exclude near-duplicates. v1 distance percentile is < 10%
    max_distance: float = 0.0025,  # 3% label noise, but these diffs are so subtle that it should be fine
    num_candidates_per_query: int = 5,
):
    """
    Returns the closest `num_candidates_per_query` candidate indices per query matching the distance filters.
    """
    return mine_from_distance_matrix(
        query_candidate_distances,
        min_distance,
        max_distance,
        num_candidates_per_query,
        sort_distances_ascending=True,
    )


def record_from_pair(
    record_query: dict[str, Any],
    record_candidate: dict[str, Any],
    distance: float,
    source: str,
    synthetic_label: Literal["GROUP", "SEPARATE"],
    seer_threshold: float = 0.01,  # v1 grouping threshold in prod
):
    # GroupHash-specific info
    query_kv = {k: v for k, v in record_query.items() if k.startswith("query_")}
    candidate_kv = {k: v for k, v in record_candidate.items() if k.startswith("candidate_")}

    # Non-GroupHash, non-pair-specific info
    rest_kv = {k: v for k, v in record_query.items() if (k not in query_kv) and (k not in candidate_kv)}
    rest_kv["source"] = source

    # Pair-specific info we'll override
    rest_kv["distance"] = distance
    rest_kv["is_grouped"] = distance < seer_threshold

    # New label
    label_result_kv = asdict(
        LabelResult(
            idx=0,
            label=synthetic_label,
            confidence_score=None,
            response_output="",
            prompt="",
            thinking_output="",
        )
    )
    _ = label_result_kv.pop("idx")

    return (
        query_kv | candidate_kv | rest_kv | label_result_kv  # last to override label result values in rest_kv
    )


def synthetic_df(
    df: pl.DataFrame,
    query_candidate_index_pairs: list[tuple[int, int]],
    distances_matrix: np.ndarray,
    source: str,
    synthetic_label: Literal["GROUP", "SEPARATE"],
):
    if not query_candidate_index_pairs:
        return df.clear()
    df = pl.DataFrame(
        [
            record_from_pair(
                df.row(query_idx, named=True),
                df.row(candidate_idx, named=True),
                distances_matrix[query_idx, candidate_idx],
                source,
                synthetic_label,
            )
            for query_idx, candidate_idx in query_candidate_index_pairs
        ],
        infer_schema_length=None,
    )
    return gt.utils.deduplicate_pairs(df)


def encode_deduplicated(
    model: gt.utils.SentenceTransformer, queries: list[str], candidates: list[str], batch_size: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    texts_unique, inverse_indices = np.unique(queries + candidates, return_inverse=True)
    embeddings_unique = model.encode(
        cast(list[str], texts_unique.tolist()),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    all_embeddings = embeddings_unique[inverse_indices]
    num_queries = len(queries)
    query_embeddings = all_embeddings[:num_queries]
    candidate_embeddings = all_embeddings[num_queries:]
    return query_embeddings, candidate_embeddings


def mine_from_project(
    df_project: pl.DataFrame,
    distances: np.ndarray,
    mine_from_distance_matrix_fn: Callable[[np.ndarray], list[list[int]]],
    source: str,
    synthetic_label: Literal["GROUP", "SEPARATE"],
) -> pl.DataFrame:
    candidate_indices_per_query = mine_from_distance_matrix_fn(distances)
    query_candidate_index_pairs = [
        (query_idx, candidate_idx)
        for query_idx, candidate_indices in enumerate(candidate_indices_per_query)
        for candidate_idx in candidate_indices
    ]
    return synthetic_df(df_project, query_candidate_index_pairs, distances, source, synthetic_label)


def mine_semi_easy_negatives(df_project: pl.DataFrame, distances: np.ndarray) -> pl.DataFrame:
    return mine_from_project(
        df_project,
        distances,
        mine_semi_easy_negatives_from_distance_matrix,
        source="synthetic-negative-semi-easy",
        synthetic_label="SEPARATE",
    )


def mine_easy_positives(df_project: pl.DataFrame, distances: np.ndarray) -> pl.DataFrame:
    return mine_from_project(
        df_project,
        distances,
        mine_easy_positives_from_distance_matrix,
        source="synthetic-positive-easy",
        synthetic_label="GROUP",
    )


def main(
    gcs_model_folder: str,
    csv_paths: tuple[str, ...],
    positives: bool = False,
    negatives: bool = False,
    text_prefix: str = "",
    batch_size: int = 2,
    *,
    no_gpu: bool = False,
    zone: str | None = None,
):
    """
    Mine synthetic positives and negatives from labeled pair CSVs.

    Parameters
    ----------
    gcs_model_folder
        GCS path to a SentenceTransformer model directory
        (e.g. gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v1/inference).
    csv_paths
        Paths to labeled pair CSVs to mine from.
    positives
        Mine easy positives.
    negatives
        Mine semi-easy negatives.
    text_prefix
        String to prepend to every text before tokenization (e.g. "clustering: ").
    batch_size
        Batch size for encoding.
    no_gpu
        Don't flex-start an L4 and run this same invocation there, instead run it locally.
    zone
        Override the default GCP zone for the gpu type when launching the GPU instance. Useful when flex-start capacity
        is dry in the default zone for the requested gpu type.
    """
    if not positives and not negatives:
        raise SystemExit("At least one of --positives or --negatives must be provided.")

    if not (no_gpu or gt.launch.is_on_remote()):
        run_name = os.path.basename(os.path.dirname(gcs_model_folder.rstrip("/")))
        gt.launch.run_argv_remotely(
            gpu="l4",
            job_type=gt.launch.JobType.SYNTH,
            name_suffix=gt.launch.shortname_from_run_name(run_name),
            zone=zone,
        )
        return

    dir_model = tempfile.mkdtemp()
    subprocess.run(["gcloud", "storage", "rsync", "-r", gcs_model_folder, dir_model], check=True)
    model = gt.utils.SentenceTransformer(dir_model, trust_remote_code=True, text_prefix=text_prefix)

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M-%S")
    df = gt.data.load_train_df(paths=csv_paths)
    df = df.sort(pl.col("query_stacktrace_string").str.len_chars().mean().over("org_id", "project_id"))
    for (org_id, project_id), df_project in tqdm(
        df.group_by("org_id", "project_id"), total=len(df["project_id"].unique()), desc="Projects"
    ):
        path_synthetic = Path("dataset_augmented") / f"org_{org_id}" / f"project_{project_id}" / "synthetic" / timestamp
        path_negatives = path_synthetic / "negatives" / "semi-easy.csv"
        path_positives = path_synthetic / "positives" / "easy.csv"

        df_project = df_project.sort("query_stacktrace_string")
        query_embeddings, candidate_embeddings = encode_deduplicated(
            model,
            df_project["query_stacktrace_string"].to_list(),
            df_project["candidate_stacktrace_string"].to_list(),
            batch_size=batch_size,
        )
        distances = 1 - (query_embeddings @ candidate_embeddings.T)

        if negatives:
            df_negatives = mine_semi_easy_negatives(df_project, distances)
            path_negatives.parent.mkdir(parents=True, exist_ok=True)
            df_negatives.write_csv(path_negatives)

        if positives:
            df_positives = mine_easy_positives(df_project, distances)
            path_positives.parent.mkdir(parents=True, exist_ok=True)
            df_positives.write_csv(path_positives)

    subprocess.run(
        [
            "gcloud",
            "storage",
            "rsync",
            "-r",
            "dataset_augmented",
            f"gs://{os.environ['GROUPING_TRAINER_BUCKET']}/dataset_augmented",
        ],
        check=True,
    )


if __name__ == "__main__":
    tapify(main)
