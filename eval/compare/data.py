"""Loading and joining similarity CSVs from GCS, plus schema validation."""

import subprocess
from pathlib import Path

import polars as pl

import grouping_trainer as gt

COLS_JOIN = ("query_stacktrace_string", "candidate_stacktrace_string")
COLUMNS_EXPECTED = (*gt.data.COLUMNS_REQUIRED, "is_grouped")
COLUMNS_ANONYMIZED_DENYLIST = ("path",)


def _resolve_cos_sim(df: pl.DataFrame, dim: int) -> tuple[str, str]:
    """Find the cos_sim_{dim} column and return (column_name, dim_label)."""
    col = f"cos_sim_{dim}"
    if col not in df.columns:
        cols_cos_sim = [name for name in df.columns if name.startswith("cos_sim")]
        raise ValueError(f"Column {col} not found. Available: {cols_cos_sim}")
    return col, str(dim)


def _check_no_duplicate_pairs(df: pl.DataFrame, source: Path) -> None:
    """
    Raise if the DataFrame has duplicate stacktrace pairs. The join in _load_and_join assumes 1:1 rows.
    eval/save_embeddings.py should have already deduplicated pairs when loading the test data.
    """
    n_dupes = len(df) - df.select(COLS_JOIN).n_unique()
    if n_dupes > 0:
        raise ValueError(f"{source} has {n_dupes} duplicate stacktrace pairs")


def _check_expected_columns(df: pl.DataFrame, source: Path) -> None:
    """Raise unless `df.columns` exactly matches COLUMNS_EXPECTED (cos_sim_{dim} columns excluded)
    and every cos_sim_{dim} column is float-typed.
    """
    actual = {name for name in df.columns if not name.startswith("cos_sim_")}
    expected = set(COLUMNS_EXPECTED)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{source} column mismatch: missing={missing}, extra={extra}")

    cos_sim_non_float = [name for name in df.columns if name.startswith("cos_sim_") and not df[name].dtype.is_float()]
    if cos_sim_non_float:
        raise ValueError(f"{source} cos_sim_* columns must be float-typed: non-float columns = {cos_sim_non_float}")


def _load_and_join(
    path_model1: Path,
    path_model2: Path,
    dim_model1: int,
    dim_model2: int,
    name_model1: str,
    name_model2: str,
) -> tuple[pl.DataFrame, str, str]:
    """Load similarity CSVs, select cos_sim columns, join into a single DataFrame.

    Returns (joined_df, dim_label1, dim_label2).
    """
    df1 = pl.read_csv(path_model1)
    col1, label_dim1 = _resolve_cos_sim(df1, dim_model1)

    _check_no_duplicate_pairs(df1, path_model1)

    if path_model1.resolve() == path_model2.resolve():
        col2, label_dim2 = _resolve_cos_sim(df1, dim_model2)
        if col1 == col2:
            raise ValueError(f"Both models resolve to the same column: {col1}")
        df = df1.rename({col1: f"cos_sim_{name_model1}", col2: f"cos_sim_{name_model2}"})
    else:
        df2 = pl.read_csv(path_model2)
        col2, label_dim2 = _resolve_cos_sim(df2, dim_model2)

        _check_no_duplicate_pairs(df2, path_model2)

        # Validate pair sets match
        pairs1 = df1.select(COLS_JOIN)
        pairs2 = df2.select(COLS_JOIN)
        if not pairs1.equals(pairs2):
            only1 = pairs1.join(pairs2, on=COLS_JOIN, how="anti")
            only2 = pairs2.join(pairs1, on=COLS_JOIN, how="anti")
            raise ValueError(
                f"Pair mismatch: model1 has {len(df1)} rows, model2 has {len(df2)}. "
                f"{len(only1)} only in model1, {len(only2)} only in model2."
            )
        # Rename before join to avoid suffix collision when both use the same dim
        df2_subset = df2.select([*COLS_JOIN, col2]).rename({col2: f"cos_sim_{name_model2}"})
        df = df1.join(df2_subset, on=COLS_JOIN)

    # Rename model1's cos_sim column
    if col1 in df.columns:
        df = df.rename({col1: f"cos_sim_{name_model1}"})

    # Drop any remaining cos_sim columns that aren't the two we care about
    cols_keep = {f"cos_sim_{name_model1}", f"cos_sim_{name_model2}"}
    cols_drop = [name for name in df.columns if name.startswith("cos_sim") and name not in cols_keep]
    df = df.drop(cols_drop)
    return df, label_dim1, label_dim2


def _sync_gcs(gcs_dir: str) -> Path:
    """Sync a GCS similarities directory to a local cache and return the local similarities.csv path.

    Maps e.g. ``gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v1/similarities/test_full``
    to ``eval/similarities/issue_grouping_v1/test_full/``.
    """
    gcs_dir = gcs_dir.rstrip("/")
    # Expected structure: gs://bucket/runs/{run_name}/similarities/{dataset}
    parts = gcs_dir.split("/")
    idx_similarities = parts.index("similarities")
    name_run = parts[idx_similarities - 1]
    name_dataset = parts[idx_similarities + 1]
    dir_local = Path("eval/similarities") / name_run / name_dataset
    dir_local.mkdir(parents=True, exist_ok=True)
    print(f"Syncing {gcs_dir} -> {dir_local}")
    subprocess.run(["gcloud", "storage", "rsync", "-r", gcs_dir, str(dir_local)], check=True)
    return dir_local / "similarities.csv"
