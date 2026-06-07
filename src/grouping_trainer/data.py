import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Literal, TypedDict

import polars as pl
import torch
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm

import grouping_trainer as gt

logger = logging.getLogger(__name__)

HoldoutMode = Literal["drop_platforms", "drop_random_match"]

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


def _apply_platform_holdout(
    df: pl.DataFrame,
    platforms_holdout: tuple[str, ...],
    holdout_mode: HoldoutMode,
    holdout_seed: int,
) -> pl.DataFrame:
    """
    Drop rows for an out-of-platform generalization experiment, volume-matched across arms.

    With `holdout_mode="drop_platforms"` (treatment) every row whose `platform` is in `platforms_holdout` is removed.
    With `holdout_mode="drop_random_match"` (control) the same *number* of rows is dropped uniformly at random, so the
    held-out platforms stay in-distribution but the training volume matches the treatment arm exactly. Both arms end at
    `df.height - n_holdout` rows. A no-op when `platforms_holdout` is empty.
    """
    if not platforms_holdout:
        return df

    platforms_present = set(df["platform"].unique().to_list())
    platforms_missing = sorted(set(platforms_holdout) - platforms_present)
    assert not platforms_missing, (
        f"platforms_holdout not found in data: {platforms_missing}. Present platforms: {sorted(platforms_present)}"
    )

    is_holdout = pl.col("platform").is_in(platforms_holdout)
    n_holdout = df.select(is_holdout.sum()).item()
    if holdout_mode == "drop_platforms":
        df_holdout = df.filter(~is_holdout)
    elif holdout_mode == "drop_random_match":
        df_holdout = df.sample(n=df.height - n_holdout, seed=holdout_seed)
    else:
        raise ValueError(f"Unknown holdout_mode: {holdout_mode!r}")

    counts_held = df.filter(is_holdout)["platform"].value_counts(sort=True)
    counts_by_platform = dict(zip(counts_held["platform"], counts_held["count"], strict=True))
    logger.info(
        f"Platform holdout ({holdout_mode}, seed={holdout_seed}): {df.height:,} -> {df_holdout.height:,} pairs "
        f"(dropped {n_holdout:,} matching {list(platforms_holdout)}; counts {counts_by_platform})"
    )
    return df_holdout


def _concat_check_dedupe(
    paths: tuple[str, ...],
    sample_size: int | None = None,
    n_rows_per_csv: int | None = None,
    platforms_holdout: tuple[str, ...] = (),
    holdout_mode: HoldoutMode = "drop_platforms",
    holdout_seed: int = 42,
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
    # Apply the holdout on the full deduped set so the dropped count is measured against the real training corpus,
    # before the CPU-sanity `sample_size` downsample.
    df = _apply_platform_holdout(df, platforms_holdout, holdout_mode, holdout_seed)
    if sample_size is not None:
        df = df.sample(n=sample_size, seed=42)
    return df


def load_val_df(paths: tuple[str, ...] = DEFAULT_VAL_PATHS, sample_size: int | None = None):
    return _concat_check_dedupe(paths, sample_size=sample_size)


def load_train_df(
    paths: tuple[str, ...] = DEFAULT_TRAIN_PATHS,
    sample_size: int | None = None,
    n_rows_per_csv: int | None = None,
    platforms_holdout: tuple[str, ...] = (),
    holdout_mode: HoldoutMode = "drop_platforms",
    holdout_seed: int = 42,
):
    """
    `n_rows_per_csv` is a laptop-sanity knob: caps `pl.read_csv` rows per file. Prefix sample (not uniform), so don't
    use it for anything where distribution matters.

    `platforms_holdout` / `holdout_mode` / `holdout_seed` drive the out-of-platform generalization experiment. See
    `_apply_platform_holdout`. These only apply to training data.
    """
    df = _concat_check_dedupe(
        paths,
        sample_size=sample_size,
        n_rows_per_csv=n_rows_per_csv,
        platforms_holdout=platforms_holdout,
        holdout_mode=holdout_mode,
        holdout_seed=holdout_seed,
    )
    assert df.filter(pl.col("confidence_score").is_null())["source"].str.starts_with("synthetic-").all()
    return df


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


def make_dummy_batch() -> Batch:
    return Batch(
        query_stacktrace_string=["dummy"],
        candidate_stacktrace_string=["dummy"],
        label=torch.tensor([0], dtype=torch.float32),
        sample_weight=torch.tensor([1.0], dtype=torch.float32),
    )


def _record_from_dict(record_dict: dict[str, Any]) -> Record:
    label = record_dict["label"]
    if label == "GROUP":
        label_int = 1
    elif label == "SEPARATE":
        label_int = 0
    else:
        raise ValueError(f"Unknown label: {label!r} (expected 'GROUP' or 'SEPARATE')")
    return Record(
        query_stacktrace_string=record_dict["query_stacktrace_string"],
        candidate_stacktrace_string=record_dict["candidate_stacktrace_string"],
        label=label_int,
        sample_weight=float(record_dict.get("sample_weight", 1.0)),
        # NOTE: cast to float b/c polars could read the data as a string if there were nulls in the CSV
    )


def df_to_dataset(
    df: pl.DataFrame,
    group_by_query_stacktrace_string: bool = True,
    shuffle_groups: bool = True,
    seed: int | None = None,
) -> Dataset:
    """
    Convert a DataFrame to a `Dataset`, grouping records by `query_stacktrace_string`.

    Records with the same `query_stacktrace_string` are kept together for cache hits in the forward pass. By default,
    the order of groups is randomized to avoid alphabetical ordering bias during training.
    """
    if not group_by_query_stacktrace_string:
        return Dataset.from_list(
            [_record_from_dict(record_dict) for record_dict in df.rows(named=True)]  # type: ignore[bad-argument-type]
        )

    query_group_dfs = [
        # Sort within each query group by candidate length so adjacent rows have similar token counts. When the loaded
        # batch is later split into sub-batches by token budget, this minimizes padding waste.
        group_df.sort(pl.col("candidate_stacktrace_string").str.len_chars())
        for _, group_df in df.group_by("query_stacktrace_string")
    ]
    # Sort deterministically first b/c polars group_by returns groups in arbitrary order, and DDP requires all processes
    # to have the same dataset ordering.
    query_group_dfs.sort(key=lambda query_group_df: query_group_df["query_stacktrace_string"][0])
    if shuffle_groups:
        rng = random.Random(seed if seed is not None else 42)
        rng.shuffle(query_group_dfs)

    return Dataset.from_list(
        [  # type: ignore[bad-argument-type]
            _record_from_dict(record_dict)
            for query_group_df in query_group_dfs
            for record_dict in query_group_df.rows(named=True)
        ]
    )


def create_project_dataset_dict(
    df: pl.DataFrame,
    min_dataset_size: int | None = None,
) -> DatasetDict:
    """
    Create a `DatasetDict` with one dataset per project. Projects below `min_dataset_size` are packed into a single
    dataset to avoid tiny batches. `min_dataset_size` can simply be set to the global/effective training batch size.
    """
    project_id_to_dataset: dict[str, Dataset] = {}
    small_project_dfs: list[pl.DataFrame] = []

    for (project_id,), df_project in tqdm(
        df.group_by("project_id"),
        total=len(df["project_id"].unique()),
        desc="Creating project datasets",
    ):
        project_id = str(project_id)
        # DatasetDict implements __getitem__ by accepting a mix of int and str. int is for array-like indexing so
        # that it can be used by torch dataloading, while the string is for whatever we want.

        if (min_dataset_size is not None) and (df_project.height < min_dataset_size):
            small_project_dfs.append(df_project)
        else:
            project_id_to_dataset[project_id] = df_to_dataset(df_project)

    if small_project_dfs:
        df_packed = pl.concat(small_project_dfs)
        project_id_to_dataset["__packed__"] = df_to_dataset(df_packed)

    return DatasetDict(project_id_to_dataset)  # type: ignore[no-matching-overload]


def _load_train_df(
    sample_size: int | None = None,
    stress_test_min_pair_len: int | None = None,
    paths: tuple[str, ...] = DEFAULT_TRAIN_PATHS,
    source_to_sample_weight: dict[str, float] | None = None,
    platforms_holdout: tuple[str, ...] = (),
    holdout_mode: HoldoutMode = "drop_platforms",
    holdout_seed: int = 42,
) -> tuple[pl.DataFrame, int]:
    holdout_kwargs = dict(platforms_holdout=platforms_holdout, holdout_mode=holdout_mode, holdout_seed=holdout_seed)
    if stress_test_min_pair_len is not None:  # used for OOM stress testing
        df = load_train_df(paths=paths, sample_size=None, **holdout_kwargs)
        df = df.filter(
            (pl.col("query_stacktrace_string").str.len_chars() + pl.col("candidate_stacktrace_string").str.len_chars())
            > stress_test_min_pair_len
        )
    else:
        df = load_train_df(paths=paths, sample_size=sample_size, **holdout_kwargs)

    if source_to_sample_weight:
        df = df.with_columns(
            pl.col("source").replace_strict(source_to_sample_weight, default=1.0).alias("sample_weight")
        )
    else:
        df = df.with_columns(pl.lit(1.0).alias("sample_weight"))

    num_projects = len(df["project_id"].unique())
    return df, num_projects


def load_train_dataset(
    sample_size: int | None = None,
    stress_test_min_pair_len: int | None = None,
    paths: tuple[str, ...] = DEFAULT_TRAIN_PATHS,
    source_to_sample_weight: dict[str, float] | None = None,
    platforms_holdout: tuple[str, ...] = (),
    holdout_mode: HoldoutMode = "drop_platforms",
    holdout_seed: int = 42,
) -> tuple[Dataset, float, int]:
    df, num_projects = _load_train_df(
        sample_size=sample_size,
        stress_test_min_pair_len=stress_test_min_pair_len,
        paths=paths,
        source_to_sample_weight=source_to_sample_weight,
        platforms_holdout=platforms_holdout,
        holdout_mode=holdout_mode,
        holdout_seed=holdout_seed,
    )
    dataset_train = df_to_dataset(df, group_by_query_stacktrace_string=False)
    frac_positive = float((df["label"] == "GROUP").mean())  # type: ignore[arg-type]
    return dataset_train, frac_positive, num_projects


def load_train_dataset_dict(
    sample_size: int | None = None,
    stress_test_min_pair_len: int | None = None,
    paths: tuple[str, ...] = DEFAULT_TRAIN_PATHS,
    source_to_sample_weight: dict[str, float] | None = None,
    min_dataset_size: int | None = None,
    platforms_holdout: tuple[str, ...] = (),
    holdout_mode: HoldoutMode = "drop_platforms",
    holdout_seed: int = 42,
) -> tuple[DatasetDict, float, int]:
    df, num_projects = _load_train_df(
        sample_size=sample_size,
        stress_test_min_pair_len=stress_test_min_pair_len,
        paths=paths,
        source_to_sample_weight=source_to_sample_weight,
        platforms_holdout=platforms_holdout,
        holdout_mode=holdout_mode,
        holdout_seed=holdout_seed,
    )
    dataset_dict_train = create_project_dataset_dict(df, min_dataset_size=min_dataset_size)
    frac_positive = float((df["label"] == "GROUP").mean())  # type: ignore[arg-type]
    return dataset_dict_train, frac_positive, num_projects
