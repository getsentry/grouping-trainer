import os
import subprocess
from pathlib import Path

import polars as pl

import grouping_trainer as gt

COLUMNS_REQUIRED = (
    "query_seer_event_sent",
    "candidate_seer_event_sent",
    "distance",
    # - Cosine distance according to v1
    "query_group_id",
    "candidate_group_id",
    "query_hash",
    "candidate_hash",
    "query_grouphash_id",
    "candidate_grouphash_id",
    "query_grouphashmetadata_id",
    "candidate_grouphashmetadata_id",
    "query_seer_gr_id",
    "candidate_seer_gr_id",
    "query_error_type",
    "candidate_error_type",
    "project_id",
    "platform",
    "source",
    # - 'matched'   - https://github.com/getsentry/data-analysis/blob/main/grouping/data/query_bq.py
    # - 'unmatched' - https://github.com/getsentry/data-analysis/blob/main/grouping/data/query_bq.py
    # - 'synthetic-negative-semi-easy' - synthetic.py
    # - 'synthetic-positive-easy'      - synthetic.py
    # - 'synthetic-hard-negative-llm' - https://github.com/getsentry/data-analysis/blob/main/grouping/data/synthetic_hard_negatives.py
    # - 'synthetic-hard-positive-llm' - https://github.com/getsentry/data-analysis/blob/main/grouping/data/synthetic_hard_negatives.py
    "path",
    # - Path to the CSV file containing the pairs in the GCS bucket, e.g., Seer pairs which were not grouped
    #   by v1: 'dataset/org_{id}/project_{id}/2026-01-01-00-00-00/unmatched.csv'
    "query_stacktrace_string",
    "candidate_stacktrace_string",
    "label",
    "thinking_output",
    "response_output",
    "confidence_score",
    "prompt",
    "org_id",
)

# Loading functions

DEFAULT_TRAIN_PATHS: tuple[str, ...] = (
    "final_csvs/train.csv",
    "final_csvs/train_more.csv",
    "final_csvs/train_more2.csv",
    "final_csvs/synthetic-easy.csv",
)
DEFAULT_TRAIN_PATHS_NO_SYNTHETIC = tuple(path for path in DEFAULT_TRAIN_PATHS if "synthetic" not in path)

DEFAULT_VAL_PATHS: tuple[str, ...] = ("final_csvs/val.csv",)


def ensure_local(paths: tuple[str, ...]) -> None:
    """
    Download missing paths from `gs://$GROUPING_TRAINER_BUCKET/{path}` to `./{path}`. No-op on the remote VM, where
    bin/_startup.sh has already downloaded final_csvs/.
    """
    bucket = os.environ["GROUPING_TRAINER_BUCKET"]
    for path in paths:
        if Path(path).exists():
            continue
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", f"gs://{bucket}/{path}", path], check=True)


def _concat_check_dedupe(
    paths: tuple[str, ...],
    sample_size: int | None = None,
    n_rows_per_csv: int | None = None,
):
    df = gt.utils.concat_vertical_unordered(
        (pl.read_csv(path, n_rows=n_rows_per_csv) for path in paths), how="vertical_relaxed"
    )
    assert set(COLUMNS_REQUIRED).issubset(df.columns)
    assert df["label"].is_in(["GROUP", "SEPARATE"]).mean() == 1
    assert df["project_id"].is_null().sum() == 0
    assert (
        df.select(
            pl.col("query_stacktrace_string", "candidate_stacktrace_string").fill_null("").str.len_chars().gt(0).all()
        )
        .select(pl.all_horizontal(pl.all()))  # reduce over columns
        .item()
    ), "Some stacktraces are empty"
    df = gt.utils.deduplicate_pairs(df)
    if sample_size is not None:
        df = df.sample(n=sample_size, seed=42)
    return df


def load_val_df(paths: tuple[str, ...] = DEFAULT_VAL_PATHS, sample_size: int | None = None):
    return _concat_check_dedupe(paths, sample_size=sample_size)


def load_train_df(
    paths: tuple[str, ...] = DEFAULT_TRAIN_PATHS,
    sample_size: int | None = None,
    n_rows_per_csv: int | None = None,
):
    """
    `n_rows_per_csv` is a laptop-sanity knob: caps `pl.read_csv` rows per file. Prefix sample (not uniform), so don't
    use it for anything where distribution matters.
    """
    df = _concat_check_dedupe(paths, sample_size=sample_size, n_rows_per_csv=n_rows_per_csv)
    assert df.filter(pl.col("confidence_score").is_null())["source"].str.starts_with("synthetic-").all()
    return df
