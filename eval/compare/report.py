"""Markdown report buffering, anonymization-aware pl.read_csv shim, table formatting."""

from pathlib import Path

import polars as pl

from .data import COLUMNS_ANONYMIZED_DENYLIST, _check_expected_columns

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)

IS_ANONYMIZED: bool = True

# Saved reference to the un-patched pl.read_csv so the anonymizing wrapper can delegate.
polars_read_csv = pl.read_csv

_report_lines: list[str] = []


def _df_to_markdown(df: pl.DataFrame) -> str:
    """Render a Polars DataFrame as a GitHub-flavored markdown table."""
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=10000,
        fmt_str_lengths=1000,
    ):
        return str(df)


def emit(*args, **print_kwargs):
    """Print to console and buffer for the markdown report."""
    if not IS_ANONYMIZED:
        print(*args, **print_kwargs)

    parts = []
    for arg in args:
        if isinstance(arg, pl.DataFrame):
            parts.append(_df_to_markdown(arg))
        else:
            parts.append(str(arg))
    _report_lines.append(" ".join(parts))


def emit_table(df: pl.DataFrame) -> None:
    """Emit a DataFrame untruncated. The scoped pl.Config affects the console-print side of `emit()`
    (`emit` itself routes DataFrames through `_df_to_markdown`, which already disables truncation).
    """
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=1000):
        emit(df)


def emit_plot(title: str, filename: str) -> None:
    """Buffer a collapsible image embed for the markdown report."""
    emit(f"\n<details>\n<summary>{title}</summary>\n\n![{title}]({filename})\n</details>\n")


def save_report(path: Path) -> None:
    """Write buffered report lines to a markdown file."""
    path.write_text("\n".join(_report_lines) + "\n")
    print(f"Report saved to {path}")


def _make_anonymizing_read_csv(real_id_to_synthetic_str: dict):
    """Return a `pl.read_csv` replacement that maps real org/project IDs to synthetic ones."""
    anonymized_columns = (
        "org_id",
        "project_id",
    )

    def read_csv_anonymized(source, *args, **kwargs):
        df = polars_read_csv(source, *args, **kwargs)
        _check_expected_columns(df, Path(source))
        for column in anonymized_columns:
            # drop_nulls so a null org_id (documented as common after batch inference) isn't mapped
            # to a synthetic value; default=None on replace_strict below preserves the null.
            for real_id in df[column].drop_nulls().unique().sort().to_list():
                real_id_to_synthetic_str.setdefault(real_id, f"id_{len(real_id_to_synthetic_str) + 1}")
            df = df.with_columns(
                pl.col(column).replace_strict(real_id_to_synthetic_str, return_dtype=pl.String, default=None)
            )
        return df.drop(COLUMNS_ANONYMIZED_DENYLIST)

    return read_csv_anonymized
