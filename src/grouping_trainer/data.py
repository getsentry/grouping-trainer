from typing import TypedDict
from datasets import DatasetDict
import polars as pl
import torch

import grouping_trainer as gt


COLUMNS_REQUIRED = (
    "query_seer_event_sent",
    "candidate_seer_event_sent",
    "distance",
    # - Cosine distance according to the current grouping model, not the new one
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
    # - One of 'matched', 'unmatched', 'synthetic-negative-semi-easy'. See query_bq.ipynb in data-analysis for 'matched'
    #   and 'unmatched'. See synthetic.py for 'synthetic-negative-semi-easy'. I have a colab notebook for actually
    #   uploading and generating these. TODO: put that in this repo
    "path",
    # - Path to the CSV file containing the pairs in the grouping-data bucket, e.g., Seer pairs which were not grouped
    #   by the current grouping model: 'dataset/org_1/project_6178942/2025-08-13-18-22-03/unmatched.csv'
    "query_stacktrace_string",
    "candidate_stacktrace_string",
    "label",
    "thinking_output",
    "response_output",
    "confidence_score",
    "prompt",
    # - Sorry, this is mostly null after I started using the batch inference API and didn't care to populate this :-]
    "org_id",
)

# Loading functions

DEFAULT_TRAIN_PATHS = (
    "final_csvs/train.csv",
    "final_csvs/train_more.csv",
    "final_csvs/train_more2.csv",
    "final_csvs/synthetic-easy.csv",
)


def _concat_check_dedupe(paths: tuple[str, ...], sample_size: int | None = None):
    df = gt.utils.concat_vertical_unordered((pl.read_csv(path) for path in paths), how="vertical_relaxed")
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


def load_val_df(paths: tuple[str, ...] = ("final_csvs/val.csv",), sample_size: int | None = None):
    return _concat_check_dedupe(paths, sample_size=sample_size)


def load_train_df(paths: tuple[str, ...] = DEFAULT_TRAIN_PATHS, sample_size: int | None = None):
    df = _concat_check_dedupe(paths, sample_size=sample_size)
    assert df.filter(pl.col("confidence_score").is_null())["source"].str.starts_with("synthetic-").all()
    return df


def load_train_dataset_dict(
    sample_size: int | None = None,
    min_dataset_size: int | None = None,
    stress_test_min_pair_len: int | None = None,
    paths: tuple[str, ...] = DEFAULT_TRAIN_PATHS,
    source_to_sample_weight: dict[str, float] | None = None,
) -> tuple[DatasetDict, float]:
    """
    Args:
        stress_test_min_pair_len: If set, bypasses sample_size and instead keeps only pairs
            where (query + candidate character length) > this threshold. Useful for OOM stress testing.
        source_to_sample_weight: Maps source column values to sample weights. Sources not in the dict get weight 1.0.
    """
    if stress_test_min_pair_len is not None:
        df = load_train_df(paths=paths, sample_size=None)  # bypass sampling
        df = df.filter(
            (pl.col("query_stacktrace_string").str.len_chars() + pl.col("candidate_stacktrace_string").str.len_chars())
            > stress_test_min_pair_len
        )
    else:
        df = load_train_df(paths=paths, sample_size=sample_size)

    if source_to_sample_weight:
        df = df.with_columns(
            pl.col("source").replace_strict(source_to_sample_weight, default=1.0).alias("sample_weight")
        )
    else:
        df = df.with_columns(pl.lit(1.0).alias("sample_weight"))

    dataset_dict_train = gt.train.create_project_dataset_dict(df, min_dataset_size=min_dataset_size)
    frac_positive = (df["label"] == "GROUP").mean()
    return dataset_dict_train, frac_positive


# Training data types


class Record(TypedDict):
    query_stacktrace_string: str
    candidate_stacktrace_string: str
    label: int
    sample_weight: float


class Batch(TypedDict):
    query_stacktrace_string: list[str]
    candidate_stacktrace_string: list[str]
    label: torch.Tensor
    sample_weight: torch.Tensor


class Features(TypedDict):
    query_embeddings: torch.Tensor
    candidate_embeddings: torch.Tensor
