"""
Download a model from GCS, encode test data, save embeddings and similarities, and upload results to GCS.

After running this script for some `run_gcs_dir`, you can use it as a gcs_model1/gcs_model2 in eval/compare.py

For example to evaluate the baseline/prod model:

python eval/save_embeddings.py \
    --run_gcs_dir gs://grouping-data/runs/issue_grouping_v1 \
    --does_not_support_sdpa \
    --truncate_dims 64 128 256 512 768

To evaluate the finetuned model:

python eval/save_embeddings.py \
    --run_gcs_dir gs://grouping-data/runs/2026-04-10-12-39-45-large-no-prefix \
    --truncate_dims 64 128 256 512 768 \
    --use_compiled

python eval/save_embeddings.py \
    --run_gcs_dir gs://grouping-data/runs/issue_grouping_v2 \
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


def _check_no_overlap(df_train: pl.DataFrame, df_test: pl.DataFrame) -> None:
    # There should be 0 project overlap
    projects_train = set(df_train["project_id"].unique().to_list())
    projects_test = set(df_test["project_id"].unique().to_list())
    projects_overlap = projects_train & projects_test
    assert not projects_overlap, (
        f"Train/test project overlap: {len(projects_overlap)} projects in common. "
        f"Showing first 10 project IDs: {sorted(projects_overlap)[:10]}"
    )

    # There should be almost no pair overlap. Hash pairs to avoid materializing full stacktrace strings
    hash_expr = pl.concat_str(
        pl.min_horizontal(_COLUMNS_PAIR),  # canonicalize order since grouping is symmetric
        pl.max_horizontal(_COLUMNS_PAIR),
        separator="\x00",
    ).hash()
    hashes_train = set(df_train.select(hash_expr).to_series().to_list())
    hashes_test = df_test.select(hash_expr).to_series()
    n_overlap = hashes_test.is_in(hashes_train).sum()
    fraction_overlap = n_overlap / len(hashes_test)
    # When making this data I didn't dedupe pairs across train/test. There happens to be a tiny amount of overlap for a
    # handful of short iOS and Java stacktraces. Maybe this is b/c of projects migrating or generic stacktraces.
    #
    # 37 / 235_298 = 0.00016 overlap b/t DEFAULT_TRAIN_PATHS and test_full2.csv
    assert fraction_overlap < 0.0005, f"Train/test pair overlap weirdly high: {fraction_overlap:.2%}"

    logger.info(
        f"No overlap: {len(projects_train)} train projects, {len(projects_test)} test projects, "
        f"{len(hashes_train)} train pairs, {len(hashes_test)} test pairs"
    )


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
    cols_needed = ["project_id", *_COLUMNS_PAIR]
    logger.info(f"Loading training data columns {cols_needed} from {paths_train} to check for overlap w/ test data.")
    df_train = pl.concat(
        [pl.read_csv(path, columns=cols_needed).select(cols_needed) for path in paths_train],
    )
    _check_no_overlap(df_train, df_test)


def main(
    run_gcs_dir: str,
    prompt_prefix: str = "",
    df_path: str = "final_csvs/test_full2.csv",
    truncate_dims: tuple[int, ...] | None = None,
    batch_size: int = 2,
    sample_size: int | None = None,
    does_not_support_sdpa: bool = False,
    use_compiled: bool = False,
):
    """
    Download a model from GCS, encode df_path texts, and save embeddings + cosine similarities.

    Parameters
    ----------
    run_gcs_dir
        GCS path to the training run directory, e.g., gs://grouping-data/runs/my-run
    prompt_prefix
        String to prepend to every text before tokenization, e.g., for lightonai/modernbert-embed-large "clustering: "
    df_path
        Path to the validation/test CSV file.
    truncate_dims
        Grid of dimensions to truncate embeddings to. A cos_sim_{dim} column is added for each.
        None (default) computes a single cos_sim column using the full dimensionality.
    sample_size
        Number of rows to sample. None (default) uses the full dataset.
    does_not_support_sdpa
        If True, skip bfloat16 and SDPA attention for models that don't support it.
    use_compiled
        If True, compiles the model.
    """
    gt.logging.configure_logging(process_type="save_embeddings")

    if use_compiled and batch_size != 1:
        logger.warning(
            "use_compiled is currently only supported with batch_size=1, b/c that's what we use in prod. "
            "Re-setting batch_size=1."
        )
        batch_size = 1

    run_gcs_dir = run_gcs_dir.rstrip("/")
    path_gcs_inference = f"{run_gcs_dir}/inference"
    name_dataset = os.path.splitext(os.path.basename(df_path))[0]
    dir_gcs_output = f"{run_gcs_dir}/similarities/{name_dataset}"

    df = gt.data.load_val_df(paths=(df_path,), sample_size=sample_size)
    logger.info(f"Test df shape: {df.shape}")

    _check_no_train_test_overlap(run_gcs_dir, df)

    with tempfile.TemporaryDirectory() as dir_tmp:
        logger.info(f"Downloading model from {path_gcs_inference} ...")
        subprocess.run(["gcloud", "storage", "rsync", "-r", path_gcs_inference, dir_tmp], check=True)

        model_kwargs = {}
        if not does_not_support_sdpa and torch.cuda.is_bf16_supported():
            model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")

        logger.info("Loading model...")
        start = time.monotonic()

        st_class = gt.compiled.SentenceTransformer if use_compiled else gt.utils.SentenceTransformer
        model = st_class(
            dir_tmp,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            prompt_prefix=prompt_prefix,
        )
        logger.info(f"{st_class.__name__} loaded in {time.monotonic() - start:.1f}s")
        if use_compiled:
            model.compile_and_warm_up()
        else:
            _ = model.encode("warm up")
        logger.info(f"{st_class.__name__} loading and warming up done in {time.monotonic() - start:.1f}s")

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
