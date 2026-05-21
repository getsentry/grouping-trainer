"""
Encode test data with Vertex Gemini's `gemini-embedding-2`, save embeddings and similarities, and upload to GCS.

Output schema mirrors eval/save_embeddings.py so the resulting GCS dir is a drop-in `gcs_model2` for `eval.compare`.

Full run:

python eval/save_gemini_embeddings.py \
    --run_gcs_dir gs://$GROUPING_TRAINER_BUCKET/runs/gemini-embedding-2 \
    --output_dimensionality 3072 \
    --truncate_dims 64 128 256 512 768 1536 3072
"""

import logging
import os.path
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.genai
import numpy as np
import polars as pl
import torch
from google.genai import types
from sentence_transformers.util import pairwise_cos_sim
from tap import tapify
from tqdm.auto import tqdm

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def _encode(
    client: google.genai.Client,
    texts: list[str],
    *,
    model: str,
    task_prefix: str,
    output_dimensionality: int,
    max_concurrency: int,
) -> np.ndarray:
    """Dedup, embed one text at a time with a thread pool, scatter back.

    `gemini-embedding-2`'s embedContent API rejects multi-content batches ("only supports one content at a time"),
    so each call sends a single text. Parallelism comes from threads — the genai client's HttpRetryOptions handles
    rate-limit pushback.
    """
    texts_unique = list(dict.fromkeys(texts))
    logger.info(f"Encoding {len(texts_unique)} unique texts ({len(texts)} total before dedup)")

    config = types.EmbedContentConfig(output_dimensionality=output_dimensionality)

    def _embed_one(text: str) -> tuple[str, list[float]]:
        response = client.models.embed_content(model=model, contents=task_prefix + text, config=config)
        embeddings = response.embeddings or []
        if len(embeddings) != 1 or embeddings[0].values is None:
            raise ValueError(f"Unexpected embedding response for text {text[:100]!r}: {embeddings}")
        return text, embeddings[0].values

    text_to_embedding: dict[str, list[float]] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = [pool.submit(_embed_one, t) for t in texts_unique]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            text, values = future.result()
            text_to_embedding[text] = values

    return np.array([text_to_embedding[text] for text in texts], dtype=np.float32)


def main(
    run_gcs_dir: str,
    model: str = "gemini-embedding-2",
    df_path: str = "final_csvs/test_full3.csv",
    task_prefix: str = "task: clustering | query: ",  # better than sentence similarity. Idk if open-ended prompts work
    output_dimensionality: int = 3072,
    truncate_dims: tuple[int, ...] = (64, 128, 256, 512, 768, 1536, 3072),
    region: str = "global",
    sample_size: int | None = None,
    max_concurrency: int = 16,
):
    """
    Encode df_path texts via Vertex Gemini and save embeddings + cosine similarities, mirroring save_embeddings.py.

    Parameters
    ----------
    run_gcs_dir
        GCS path under which to write `similarities/{name_dataset}/`. Mirrors save_embeddings.py's layout so the
        result is a drop-in `gcs_model2` for `eval.compare`. Example:
        gs://$GROUPING_TRAINER_BUCKET/runs/gemini-embedding-2
    model
        Vertex model name. Default `gemini-embedding-2`.
    df_path
        Path to the validation/test CSV file.
    task_prefix
        Prepended to every text. `gemini-embedding-2` does not support the `task_type` config parameter; the
        recommended way to specify the task is to bake it into the prompt. The clustering format is symmetric, so
        the same prefix is used on both query and candidate stacktraces. See eval/gemini-embedding.md.
    output_dimensionality
        Native embedding dim. The model is MRL-trained and 3072 is the full size; smaller values are auto-renormalized.
    truncate_dims
        Grid of post-hoc MRL truncations. A `cos_sim_{dim}` column is added for each.
    region
        Vertex region. Defaults to `us-central1`.
    sample_size
        Number of rows to sample. None (default) uses the full dataset.
    max_concurrency
        Number of parallel `embed_content` calls. v2 only accepts one text per request, so throughput is bounded by
        this. Bump up if quota allows; back off if you see persistent 429s past the SDK's retries.
    """
    gt.logging.configure_logging(process_type="save_gemini_embeddings")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    run_gcs_dir = run_gcs_dir.rstrip("/")
    name_dataset = os.path.splitext(os.path.basename(df_path))[0]
    dir_gcs_output = f"{run_gcs_dir}/similarities/{name_dataset}"

    df = gt.data.load_val_df(paths=(df_path,), sample_size=sample_size)
    logger.info(f"Test df shape: {df.shape}")

    texts_query = df["query_stacktrace_string"].to_list()
    texts_candidate = df["candidate_stacktrace_string"].to_list()
    n_q = len(texts_query)

    logger.info(f"Creating Vertex client in region {region!r}")
    client = google.genai.Client(
        vertexai=True,
        location=region,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=5,
                http_status_codes=[429, 499, 500, 502, 503, 504],
            ),
        ),
    )

    start = time.monotonic()
    all_embeddings = _encode(
        client,
        texts_query + texts_candidate,
        model=model,
        task_prefix=task_prefix,
        output_dimensionality=output_dimensionality,
        max_concurrency=max_concurrency,
    )
    logger.info(f"Encoded all texts in {time.monotonic() - start:.1f}s")

    embeddings_query = all_embeddings[:n_q]
    embeddings_candidate = all_embeddings[n_q:]

    for dim in truncate_dims:
        cos_sims = (
            pairwise_cos_sim(
                torch.as_tensor(embeddings_query[..., :dim]),
                torch.as_tensor(embeddings_candidate[..., :dim]),
            )
            .detach()
            .cpu()
            .numpy()
        )
        df = df.with_columns(pl.Series(name=f"cos_sim_{dim}", values=cos_sims))

    with tempfile.TemporaryDirectory() as dir_tmp_output:
        df.write_csv(f"{dir_tmp_output}/similarities.csv")
        np.save(f"{dir_tmp_output}/query_embeddings.npy", embeddings_query)
        np.save(f"{dir_tmp_output}/candidate_embeddings.npy", embeddings_candidate)

        logger.info(f"Uploading to {dir_gcs_output}...")
        subprocess.run(["gcloud", "storage", "rsync", "-r", dir_tmp_output, dir_gcs_output], check=True)
        logger.info(f"Uploaded to {dir_gcs_output}")


if __name__ == "__main__":
    tapify(main, description=__doc__)
