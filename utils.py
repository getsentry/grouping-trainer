from functools import lru_cache
from datasets import DatasetDict
import polars as pl

import grouping_trainer as gt


def _test_df(df: pl.DataFrame):
    assert df["label"].is_in(["GROUP", "SEPARATE"]).mean() == 1
    assert df["project_id"].is_null().sum() == 0
    assert (
        df.select(
            pl.col("query_stacktrace_string", "candidate_stacktrace_string").fill_null("").str.len_chars().gt(0).all()
        )
        .select(pl.all_horizontal(pl.all()))  # reduce over columns
        .item()
    )
    return gt.utils.deduplicate_pairs(df)


def load_val_df(path: str = "final_csvs/val.csv", sample_size: int | None = None):
    df = pl.read_csv(path)
    df = _test_df(df)

    if sample_size is not None:
        df = df.sample(n=sample_size, seed=42)

    return df


@lru_cache(maxsize=1)  # data too big for bigger cache, sorry
def _load_train_df(
    paths: tuple[str, ...] = ("final_csvs/train.csv", "final_csvs/synthetic-semi-easy-negatives.csv"),
):
    df = gt.utils.concat_vertical_unordered((pl.read_csv(path) for path in paths), how="vertical_relaxed")
    df = _test_df(df)
    assert (df.filter(pl.col("confidence_score").is_null())["source"] == "synthetic-negative-semi-easy").all()
    return df


def load_train_df(
    paths: tuple[str, ...] = ("final_csvs/train.csv", "final_csvs/synthetic-semi-easy-negatives.csv"),
    sample_size: int | None = None,
):
    df = _load_train_df(paths)
    if sample_size is not None:
        df = df.sample(n=sample_size, seed=42)
    return df


def load_train_dataset_dict(
    sample_size: int | None = None, min_dataset_size: int | None = None
) -> tuple[DatasetDict, float]:
    df = load_train_df(sample_size=sample_size)
    dataset_dict_train = gt.train.create_project_dataset_dict(df, min_dataset_size=min_dataset_size)
    frac_positive = (df["label"] == "GROUP").mean()
    return dataset_dict_train, frac_positive
