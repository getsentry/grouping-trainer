"""
Download a model from GCS, encode test data, save embeddings and similarities, and upload results to GCS.

For example to evaluate the baseline/prod model:

python eval/save_embeddings.py \
    --run_gcs_dir gs://grouping-data/runs/issue_grouping_v1 \
    --base_model "jinaai/jina-embeddings-v2-base-code" \
    --does_not_support_sdpa \
    --truncate_dims 64 128 256 512 768

To evaluate the finetuned model:

python eval/save_embeddings.py \
    --run_gcs_dir gs://grouping-data/runs/2026-04-07-11-56-28-large-con \
    --base_model "lightonai/modernbert-embed-large" \
    --truncate_dims 64 128 256 512 768

python eval/save_embeddings.py \
    --run_gcs_dir gs://grouping-data/runs/issue_grouping_v2 \
    --base_model "Alibaba-NLP/gte-modernbert-base" \
    --truncate_dims 64 128 256 512 768
"""

import json
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

_COLUMNS_PAIR = ("query_stacktrace_string", "candidate_stacktrace_string")


def _check_no_train_test_overlap(run_gcs_dir: str, df_test: pl.DataFrame) -> None:
    """
    Download training_config.json from GCS, load training data, and assert there is no overlap in projects or stacktrace
    pairs between training and test data.
    """
    path_gcs_config = f"{run_gcs_dir}/metadata/training_config.json"
    with tempfile.TemporaryDirectory() as dir_tmp:
        path_local_config = f"{dir_tmp}/training_config.json"
        result = subprocess.run(
            ["gcloud", "storage", "cp", path_gcs_config, path_local_config],
            capture_output=True,
        )
        if result.returncode != 0:
            logger.warning(f"No training_config.json found at {path_gcs_config}, skipping overlap check")
            return
        with open(path_local_config) as f:
            config = json.load(f)

    paths_train = tuple(config["training_csvs"])
    logger.info(f"Loading training data from {paths_train} to check for overlap...")
    df_train = gt.data.load_train_df(paths=paths_train)

    # Check project overlap
    projects_train = set(df_train["project_id"].unique().to_list())
    projects_test = set(df_test["project_id"].unique().to_list())
    projects_overlap = projects_train & projects_test
    assert not projects_overlap, (
        f"Train/test project overlap: {len(projects_overlap)} projects in common: {sorted(projects_overlap)[:10]}"
    )

    # Check pair overlap (canonicalize order since grouping is symmetric)
    cols_canonical = [
        pl.min_horizontal(_COLUMNS_PAIR).alias("_pair_first"),
        pl.max_horizontal(_COLUMNS_PAIR).alias("_pair_second"),
    ]
    pairs_train = set(df_train.select(cols_canonical).iter_rows())
    pairs_test = set(df_test.select(cols_canonical).iter_rows())
    pairs_overlap = pairs_train & pairs_test
    assert not pairs_overlap, f"Train/test pair overlap: {len(pairs_overlap)} pairs in common"

    logger.info(
        f"No overlap: {len(projects_train)} train projects, {len(projects_test)} test projects, "
        f"{len(pairs_train)} train pairs, {len(pairs_test)} test pairs"
    )


def main(
    run_gcs_dir: str,
    base_model: str,
    df_path: str = "final_csvs/test_full2.csv",
    truncate_dims: tuple[int, ...] | None = None,
    batch_size: int = 2,
    sample_size: int | None = None,
    does_not_support_sdpa: bool = False,
):
    """
    Download a model from GCS, encode val+test pairs, and save embeddings + cosine similarities.

    Parameters
    ----------
    run_gcs_dir
        GCS path to the training run directory (e.g. gs://grouping-data/runs/my-run).
    base_model
        Base model name to set on the model card, e.g. "Qwen/Qwen3-Embedding-0.6B"
    df_path
        Path to the validation/test CSV file.
    truncate_dims
        Grid of dimensions to truncate embeddings to. A cos_sim_{dim} column is added for each.
        None computes a single cos_sim column using the full dimensionality.
    sample_size
        Number of rows to sample. None uses the full dataset.
    does_not_support_sdpa
        If True, skip bfloat16 and SDPA attention for models that don't support it.
    """
    gt.logging.configure_logging(process_type="save_embeddings")

    run_gcs_dir = run_gcs_dir.rstrip("/")
    path_gcs_inference = f"{run_gcs_dir}/inference"
    name_dataset = os.path.splitext(os.path.basename(df_path))[0]
    dir_gcs_output = f"{run_gcs_dir}/similarities/{name_dataset}"

    df = gt.data.load_val_df(paths=(df_path,), sample_size=sample_size)
    logger.info(f"df shape: {df.shape}")

    _check_no_train_test_overlap(run_gcs_dir, df)

    with tempfile.TemporaryDirectory() as dir_tmp:
        logger.info(f"Downloading model from {path_gcs_inference} ...")
        subprocess.run(["gcloud", "storage", "rsync", "-r", path_gcs_inference, dir_tmp], check=True)

        model_kwargs = {}
        if not does_not_support_sdpa and torch.cuda.is_bf16_supported():
            model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")

        logger.info("Loading model...")
        start = time.monotonic()
        model = gt.utils.SentenceTransformer(
            dir_tmp,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
        )
        model.model_card_data.base_model = base_model
        logger.info(f"Model loaded in {time.monotonic() - start:.1f}s")

        _ = model.encode("warm up")
        logger.info(f"Warm up done in {time.monotonic() - start:.1f}s")

        logger.info("Encoding queries")
        texts_query = df["query_stacktrace_string"].to_list()
        embeddings_query: np.ndarray = model.encode(
            texts_query, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
        )

        logger.info("Encoding candidates")
        texts_candidate = df["candidate_stacktrace_string"].to_list()
        embeddings_candidate: np.ndarray = model.encode(
            texts_candidate, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
        )

    if truncate_dims is None:
        truncate_dims = (embeddings_query.shape[-1],)

    for dim in truncate_dims:
        cos_sims = pairwise_cos_sim(embeddings_query[..., :dim], embeddings_candidate[..., :dim]).detach().cpu().numpy()
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
