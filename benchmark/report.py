"""
Analyze a benchmark/run.py run: produce a markdown report comparing compiled vs base per-call encode latency.

Reads `times.csv` (one row per unique query stacktrace, columns
`query_stacktrace_string, num_tokens, time_compiled_sec, time_base_sec`) from a gs:// dir and writes
`./benchmark/README.md` by default (overwriting the previous run's report — raw data lives in GCS).

Example:

python benchmark/report.py \
    --run_gcs_dir gs://grouping-data/perf/2026-04-24-23-44-59-2026-04-10-12-39-45-large-no-prefix/test_full2
"""

import logging
import os.path
import subprocess
import tempfile
from pathlib import Path

import polars as pl
from tap import tapify

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def _bucketize(df: pl.DataFrame, edges: tuple[int, ...]) -> pl.DataFrame:
    """Add a `bucket` column whose categories are <=e0, e0+1..e1, ..., >e_last."""
    labels = [f"<={edges[0]}"]
    for i in range(1, len(edges)):
        labels.append(f"{edges[i - 1] + 1}-{edges[i]}")
    labels.append(f">{edges[-1]}")
    return df.with_columns(
        bucket=pl.col("num_tokens").cut(breaks=list(edges), labels=labels),
    )


def _summary_per_bucket(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("bucket", maintain_order=False)
        .agg(
            pl.len().alias("n"),
            pl.col("num_tokens").median().alias("tok_p50"),
            (pl.col("time_compiled_sec").median() * 1000).round(2).alias("compiled_ms_p50"),
            (pl.col("time_base_sec").median() * 1000).round(2).alias("base_ms_p50"),
            (pl.col("time_compiled_sec").quantile(0.9) * 1000).round(2).alias("compiled_ms_p90"),
            (pl.col("time_base_sec").quantile(0.9) * 1000).round(2).alias("base_ms_p90"),
            (pl.col("time_base_sec").median() / pl.col("time_compiled_sec").median())
            .round(2)
            .alias("speedup_p50"),
        )
        .sort("tok_p50")
    )


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


def _write_report(
    df: pl.DataFrame,
    summary: pl.DataFrame,
    run_txt: str,
    edges: tuple[int, ...],
    path_out: Path,
) -> None:
    speedup = df["time_base_sec"] / df["time_compiled_sec"]
    p10 = float(speedup.quantile(0.1))
    p50 = float(speedup.median())
    p90 = float(speedup.quantile(0.9))
    fraction_wins = float((speedup >= 1.0).mean())

    median_compiled_ms = float(df["time_compiled_sec"].median()) * 1000
    median_base_ms = float(df["time_base_sec"].median()) * 1000

    df_worst = (
        df.with_columns(speedup=pl.col("time_base_sec") / pl.col("time_compiled_sec"))
        .sort("speedup")
        .head(5)
        .with_columns(
            pl.col("time_compiled_sec").mul(1000).round(2).alias("compiled_ms"),
            pl.col("time_base_sec").mul(1000).round(2).alias("base_ms"),
            pl.col("speedup").round(3),
        )
        .select("num_tokens", "compiled_ms", "base_ms", "speedup")
    )

    lines = [
        "# benchmark_compiled report",
        "",
        "## Run",
        "",
        "```",
        run_txt.strip() or "(no run.txt found)",
        "```",
        "",
        f"- Token bucket boundaries used for analysis: `{edges}`",
        f"- Rows: {len(df):,}",
        "",
        "## Headline",
        "",
        f"- Median compiled: **{median_compiled_ms:.1f} ms**",
        f"- Median base:     **{median_base_ms:.1f} ms**",
        f"- Per-row speedup p10/p50/p90: **{p10:.2f}x / {p50:.2f}x / {p90:.2f}x**",
        f"- Compiled wins on **{fraction_wins:.1%}** of rows",
        "",
        "## Per-bucket",
        "",
        _df_to_markdown(summary),
        "",
        "## Worst 5 rows for compiled",
        "",
        _df_to_markdown(df_worst),
        "",
    ]
    path_out.write_text("\n".join(lines))


def main(
    run_gcs_dir: str,
    output_path: str = "./benchmark/README.md",
    token_buckets: tuple[int, ...] = (64, 128, 256, 512, 1024),
):
    """
    Parameters
    ----------
    run_gcs_dir
        gs:// path to a directory containing times.csv and run.txt, e.g.,
        gs://grouping-data/perf/{stamp}-{run_id}/{dataset_name}.
    output_path
        Local path to write the markdown report to. Overwrites if it exists. Defaults to ./benchmark/README.md
        — the report is intended to reflect the latest run, not be archived (raw data lives in GCS).
    token_buckets
        Bucket boundaries used to group rows for the per-bucket table. Should match the buckets the compiled
        model was warmed up with. Defaults to the current `gt.compiled.SentenceTransformer` defaults.
    """
    gt.logging.configure_logging(process_type="benchmark_compiled_report")

    if not run_gcs_dir.startswith("gs://"):
        raise ValueError(f"run_gcs_dir must start with gs://, got: {run_gcs_dir}")

    with tempfile.TemporaryDirectory(prefix="benchmark_report_") as dir_tmp:
        logger.info(f"Downloading {run_gcs_dir} -> {dir_tmp}")
        subprocess.run(["gcloud", "storage", "rsync", "-r", run_gcs_dir, dir_tmp], check=True)

        path_csv = os.path.join(dir_tmp, "times.csv")
        if not os.path.exists(path_csv):
            raise FileNotFoundError(f"times.csv not found at {run_gcs_dir}")
        path_runtxt = os.path.join(dir_tmp, "run.txt")
        run_txt = Path(path_runtxt).read_text() if os.path.exists(path_runtxt) else ""

        df = pl.read_csv(path_csv)

    df = _bucketize(df, edges=token_buckets)
    summary = _summary_per_bucket(df)

    path_out = Path(output_path)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    _write_report(df, summary, run_txt=run_txt, edges=token_buckets, path_out=path_out)

    logger.info(f"Wrote report to {path_out.resolve()}")
    print(_df_to_markdown(summary))


if __name__ == "__main__":
    tapify(main, description=__doc__)
