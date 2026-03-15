"""
Download a model from GCS, encode val+test data, save embeddings and similarities, and upload results to GCS.
"""

import logging
import os.path
import subprocess
import tempfile
import time

import numpy as np
import polars as pl
from sentence_transformers.util import pairwise_cos_sim
import torch
from tap import tapify

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def main(
    run_gcs_dir: str,
    df_path: str = "final_csvs/val_and_test.csv",
    truncate_dim: int | None = 64,
    batch_size: int = 2,
    sample_size: int | None = None,
):
    """
    Download a model from GCS, encode val+test pairs, and save embeddings + cosine similarities.

    Parameters
    ----------
    run_gcs_dir
        GCS path to the training run directory (e.g. gs://grouping-data/runs/my-run).
    df_path
        Path to the validation/test CSV file.
    truncate_dim
        Truncate embeddings to this many dimensions. None for full dimensionality.
    sample_size
        Number of rows to sample. None uses the full dataset.
    """
    gt.logging.configure_logging(process_type="save_embeddings")

    run_gcs_dir = run_gcs_dir.rstrip("/")
    path_gcs_inference = f"{run_gcs_dir}/inference"
    name_dataset = os.path.splitext(os.path.basename(df_path))[0]
    dir_gcs_output = f"{run_gcs_dir}/similarities/{name_dataset}"

    df = gt.data.load_val_df(path=df_path, sample_size=sample_size)
    logger.info(f"df shape: {df.shape}")

    with tempfile.TemporaryDirectory() as dir_tmp:
        logger.info(f"Downloading model from {path_gcs_inference} ...")
        subprocess.run(["gcloud", "storage", "rsync", "-r", path_gcs_inference, dir_tmp], check=True)

        kwargs_model = {}
        if torch.cuda.is_bf16_supported():
            kwargs_model = dict(dtype=torch.bfloat16, attn_implementation="sdpa")

        logger.info("Loading model...")
        start = time.monotonic()
        model = gt.utils.SentenceTransformer(
            dir_tmp,
            trust_remote_code=True,
            model_kwargs=kwargs_model,
        )
        logger.info(f"Model loaded in {time.monotonic() - start:.1f}s")

        _ = model.encode("warm up")
        logger.info(f"Warm up done in {time.monotonic() - start:.1f}s")

    logger.info("Encoding queries")
    texts_query = df["query_stacktrace_string"].to_list()
    embeddings_query = model.encode(texts_query, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)

    logger.info("Encoding candidates")
    texts_candidate = df["candidate_stacktrace_string"].to_list()
    embeddings_candidate = model.encode(
        texts_candidate, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )

    cos_sims = (
        pairwise_cos_sim(embeddings_query[..., :truncate_dim], embeddings_candidate[..., :truncate_dim])
        .detach()
        .cpu()
        .numpy()
    )
    df = df.with_columns(pl.Series(name="cos_sim", values=cos_sims))

    with tempfile.TemporaryDirectory() as dir_tmp_output:
        df.write_csv(f"{dir_tmp_output}/similarities.csv")
        np.save(f"{dir_tmp_output}/query_embeddings.npy", embeddings_query)
        np.save(f"{dir_tmp_output}/candidate_embeddings.npy", embeddings_candidate)

        logger.info(f"Uploading to {dir_gcs_output}...")
        subprocess.run(["gcloud", "storage", "rsync", "-r", dir_tmp_output, dir_gcs_output], check=True)
        logger.info(f"Uploaded to {dir_gcs_output}")


if __name__ == "__main__":
    tapify(main, description=__doc__)
