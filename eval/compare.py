"""
Head-to-head comparison b/t 2 models on held out data.

Example usage:

python eval/compare.py \
    --name_model1 v1 \
    --name_model2 large-con \
    --gcs_model1 gs://grouping-data/runs/issue_grouping_v1/similarities/test_full2 \
    --gcs_model2 gs://grouping-data/runs/2026-04-07-11-56-28-large-con/similarities/test_full2 \
    --threshold_model1 0.99 \
    --threshold_model2 0.90 \
    --dim_model2 64

python eval/compare.py \
    --name_model1 v2 \
    --name_model2 large-con \
    --gcs_model1 gs://grouping-data/runs/issue_grouping_v2/similarities/test_full2 \
    --gcs_model2 gs://grouping-data/runs/2026-04-07-11-56-28-large-con/similarities/test_full2 \
    --threshold_model1 0.90 \
    --threshold_model2 0.90 \
    --dim_model2 64
"""

from itertools import zip_longest
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import gspread
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import yaml
from google.auth import default as google_auth_default
from tap import tapify
from tqdm.auto import tqdm

sns.set_theme(style="darkgrid")


@dataclass
class CompareResult:
    """Result from compare_models."""

    df: pl.DataFrame
    """DataFrame with pred_{model_name} columns added."""

    model_names: list[str]
    """List of model names in order."""

    project_metrics: pl.DataFrame
    """Per-project metrics with columns: org_id, project_id, {model}_{metric}."""

    projects: list[dict]
    """Projects with group_rate_increase >= threshold, sorted descending."""

    more_issues_projects: list[dict]
    """Projects where model2 creates more issues (groups less) by >= min_group_rate_decrease."""


pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)

# Consistent colors: model1 = blue, model2 = orange
MODEL_COLORS = ["#1f77b4", "#ff7f0e"]  # matplotlib default blue and orange

_report_lines: list[str] = []


def report(*args, **kwargs):
    """Print to console AND buffer for the report file."""
    output = " ".join(str(a) for a in args)
    _report_lines.append(output)
    print(*args, **kwargs)


def save_report(path: Path) -> None:
    """Write buffered report lines to a text file."""
    path.write_text("\n".join(_report_lines) + "\n")
    print(f"Report saved to {path}")


def stratify_round_robin(df: pl.DataFrame, group_name: str, target_num_rows: int) -> pl.DataFrame:
    groups = (list(df_group.rows(named=True)) for _, df_group in df.group_by(group_name, maintain_order=True))
    records = []
    for records_across_groups in zip_longest(*groups, fillvalue=None):
        for record in records_across_groups:
            if record is not None:
                records.append(record)
                if len(records) == target_num_rows:
                    break
        else:
            continue
        break

    return pl.DataFrame(records)


def _projects_to_display_df(projects: list[dict]) -> pl.DataFrame:
    """Convert project dicts to a display DataFrame, excluding internal _-prefixed keys."""
    display_cols = [k for k in projects[0] if not k.startswith("_")]
    return pl.DataFrame([{k: v for k, v in p.items() if k in display_cols} for p in projects]).with_columns(
        pl.col(pl.Float64).round(2)
    )


def _upload_df_to_sheet(spreadsheet: gspread.Spreadsheet, sheet_name: str, df: pl.DataFrame) -> None:
    """Upload a polars DataFrame to a new sheet in the spreadsheet."""
    # Delete existing sheet with same name if it exists
    try:
        existing = spreadsheet.worksheet(sheet_name)
        spreadsheet.del_worksheet(existing)
    except gspread.WorksheetNotFound:
        pass

    worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=len(df) + 1, cols=len(df.columns))
    data = [df.columns] + df.to_numpy().tolist()
    worksheet.update(values=data, range_name="A1")

    # Freeze and bold header row, enable text wrapping
    worksheet.freeze(rows=1)
    worksheet.format("1:1", {"textFormat": {"bold": True}})
    worksheet.format(f"1:{len(df) + 1}", {"wrapStrategy": "WRAP"})

    # Find column indices for styling
    columns = list(df.columns)
    visible_cols = {"platform", "query_stacktrace_string", "candidate_stacktrace_string"}
    wide_cols = {"query_stacktrace_string", "candidate_stacktrace_string", "thinking_output", "response_output"}

    # Build batch update requests for column widths and hiding
    requests = []
    sheet_id = worksheet.id
    for i, col in enumerate(columns):
        if col in wide_cols:
            # Set wide columns to 400 pixels
            requests.append(
                {
                    "updateDimensionProperties": {
                        "range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
                        "properties": {"pixelSize": 400},
                        "fields": "pixelSize",
                    }
                }
            )
        if col not in visible_cols:
            # Hide other columns
            requests.append(
                {
                    "updateDimensionProperties": {
                        "range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
                        "properties": {"hiddenByUser": True},
                        "fields": "hiddenByUser",
                    }
                }
            )

    if requests:
        spreadsheet.batch_update({"requests": requests})

    # Rate limit: ~60 write requests/min, we make ~8 per sheet
    time.sleep(8)


def print_projects(
    projects: list[dict],
    description: str,
    max_projects: int | None = None,
    stratify_by: str | None = None,
) -> list[dict]:
    """Print a table of filtered projects, optionally stratified and limited.

    Args:
        projects: Project dicts to display.
        description: Label for the print header.
        max_projects: If set, limit to this many projects (after stratification).
        stratify_by: Column to stratify by when limiting projects (e.g. "platform").

    Returns:
        The (possibly filtered) list of projects that were printed.
    """
    if max_projects is not None and len(projects) > max_projects:
        projects_df = _projects_to_display_df(projects)
        if stratify_by:
            projects_df = stratify_round_robin(projects_df, stratify_by, max_projects)
        else:
            projects_df = projects_df.head(max_projects)
        selected_keys = set((row["org_id"], row["project_id"]) for row in projects_df.iter_rows(named=True))
        projects = [p for p in projects if (p["org_id"], p["project_id"]) in selected_keys]

    stratify_msg = f", stratified by {stratify_by}" if stratify_by else ""
    report(f"\n=== {description}:{stratify_msg} ===")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=1000):
        report(_projects_to_display_df(projects))

    return projects


def _upload_projects_to_sheets(
    projects: list[dict],
    description: str,
    sort_by: list[tuple[str, bool]] | None = None,
) -> None:
    """Upload merged/new data for projects to a new Google Sheet.

    Creates a new spreadsheet named with the description and current timestamp.
    Call print_projects() first to filter/display projects before uploading.

    Args:
        projects: Project dicts to upload (already filtered by print_projects if desired).
        description: Used for the spreadsheet title.
        sort_by: Optional list of (column, descending) tuples to sort each sheet by before uploading.
    """
    creds, _ = google_auth_default(
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    )
    client = gspread.authorize(creds)
    title = f"{description} ({time.strftime('%Y-%m-%d %H:%M')})"
    spreadsheet = client.create(title)

    # Build list of (sheet_name, df) to upload
    uploads = []
    for project in projects:
        prefix = f"org_{project['org_id']}|project_{project['project_id']}"
        if project["_new_df"] is not None:
            uploads.append((f"{prefix}|new", project["_new_df"]))
        if project["_merged_df"] is not None:
            uploads.append((f"{prefix}|merged", project["_merged_df"]))

    if sort_by:
        cols = [s[0] for s in sort_by]
        descending = [s[1] for s in sort_by]
        uploads = [(name, df.sort(cols, descending=descending)) for name, df in uploads]

    for sheet_name, df in tqdm(uploads, desc="Uploading to Google Sheets"):
        _upload_df_to_sheet(spreadsheet, sheet_name, df)

    # Remove the default "Sheet1" now that other sheets exist
    try:
        spreadsheet.del_worksheet(spreadsheet.worksheet("Sheet1"))
    except gspread.WorksheetNotFound:
        pass

    report(f"Done! View at: {spreadsheet.url}")


def _compute_metrics_for_model(df: pl.DataFrame, model_name: str) -> dict:
    """Compute metrics for a single model on the given dataframe."""
    pred_col = f"pred_{model_name}"
    pred_group = df.filter(pl.col(pred_col) == "GROUP")
    pred_separate = df.filter(pl.col(pred_col) == "SEPARATE")
    label_group = df.filter(pl.col("label") == "GROUP")
    label_separate = df.filter(pl.col("label") == "SEPARATE")
    return {
        "pred_GROUP_rate": (df[pred_col] == "GROUP").mean(),
        "precision_GROUP": (pred_group["label"] == "GROUP").mean() if len(pred_group) > 0 else float("nan"),
        "precision_SEPARATE": (pred_separate["label"] == "SEPARATE").mean() if len(pred_separate) > 0 else float("nan"),
        "recall_GROUP": (label_group[pred_col] == "GROUP").mean() if len(label_group) > 0 else float("nan"),
        "recall_SEPARATE": (label_separate[pred_col] == "SEPARATE").mean() if len(label_separate) > 0 else float("nan"),
    }


def _compute_metrics(df: pl.DataFrame, model_names: list[str]) -> pl.DataFrame:
    """Compute metrics for each model on the given dataframe."""
    metrics_rows = []
    for model_name in model_names:
        metrics = _compute_metrics_for_model(df, model_name)
        metrics_rows.append({"model": model_name, **metrics})
    return pl.DataFrame(metrics_rows).with_columns(pl.col(pl.Float64).round(2))


def plot_metrics_by_platform(df: pl.DataFrame, model_names: list[str]) -> plt.Figure:
    """
    Create bar plots comparing 2 models grouped by platform.

    Args:
        df: DataFrame with prediction columns (pred_{model_name}) already added.
        model_names: List of model names (expects exactly 2).

    Returns:
        Figure with one subplot per metric.
    """
    # Compute metrics per platform for each model, averaged over projects
    metrics_rows = []
    metrics_to_plot = None
    for (platform,), platform_df in df.group_by("platform"):
        for model_name in model_names:
            project_metrics_list = []
            for _, proj_df in platform_df.group_by("project_id"):
                project_metrics_list.append(_compute_metrics_for_model(proj_df, model_name))
            if metrics_to_plot is None:
                metrics_to_plot = list(project_metrics_list[0].keys())
            avg_metrics = {
                k: sum(m[k] for m in project_metrics_list if m[k] == m[k])
                / sum(1 for m in project_metrics_list if m[k] == m[k])
                for k in project_metrics_list[0]
            }
            avg_metrics["platform"] = platform
            avg_metrics["model"] = model_name
            metrics_rows.append(avg_metrics)

    metrics_df = pl.DataFrame(metrics_rows)

    # Convert to pandas and pivot for plotting
    metrics_pd = metrics_df.to_pandas()
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5))
    axes: list[plt.Axes] = list(axes)

    for ax, metric in zip(axes, metrics_to_plot):
        pivot_df = metrics_pd.pivot(index="platform", columns="model", values=metric)
        pivot_df = pivot_df[model_names]  # ensure consistent column order
        pivot_df.plot(kind="bar", ax=ax, rot=45, legend=False, color=MODEL_COLORS)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylim(0, 1)

    # Single legend for the whole figure (top center)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(model_names), bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # make room for legend on top
    return fig


def plot_similarity_distribution(
    df: pl.DataFrame,
    model_name: str,
    bins: int = 50,
) -> plt.Figure:
    """
    Plot histograms of cosine similarity distribution, one subplot per platform.

    Args:
        df: DataFrame with cos_sim_{model_name} and platform columns.
        model_name: Model name (used to find cos_sim_ column).
        bins: Number of histogram bins.

    Returns:
        Figure with vertically stacked subplots.
    """
    sim_col = f"cos_sim_{model_name}"
    platforms = sorted(df["platform"].unique().to_list())
    n = len(platforms)

    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, platform in zip(axes, platforms):
        data = df.filter(pl.col("platform") == platform)[sim_col].to_numpy()
        ax.hist(data, bins=bins, edgecolor="none", alpha=0.8)
        ax.set_ylabel(platform, rotation=0, labelpad=60, ha="right")
        ax.tick_params(left=False, labelleft=False)

    axes[-1].set_xlabel(f"Cosine Similarity ({model_name})")
    fig.suptitle(f"Similarity Distribution by Platform ({model_name})", fontsize=14)
    plt.tight_layout()
    return fig


def plot_dumbbell_by_project(
    project_metrics_df: pl.DataFrame, model_names: list[str], metrics: list[str] | None = None
) -> plt.Figure:
    """
    Create dumbbell plots comparing 2 models across all projects.

    Args:
        project_metrics_df: DataFrame with columns like {model}_{metric} for each project.
        model_names: List of model names (expects exactly 2).
        metrics: List of metrics to plot. If None, plots all available metrics.

    Returns:
        Figure with one dumbbell subplot per metric.
    """
    model1, model2 = model_names

    # Get available metrics
    if metrics is None:
        metrics = [c.replace(f"{model1}_", "") for c in project_metrics_df.columns if c.startswith(f"{model1}_")]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(8, len(project_metrics_df) * 0.15)))
    if n_metrics == 1:
        axes = [axes]
    axes: list[plt.Axes] = list(axes)

    # Sort once by pred_GROUP_rate delta, use same order for all subplots
    group_rate_col1 = f"{model1}_pred_GROUP_rate"
    group_rate_col2 = f"{model2}_pred_GROUP_rate"
    sorted_df = project_metrics_df.with_columns(
        (pl.col(group_rate_col2) - pl.col(group_rate_col1)).alias("_delta")
    ).sort("_delta")
    y_labels = [f"{row['org_id']}|{row['project_id']}" for row in sorted_df.iter_rows(named=True)]

    for ax, metric in zip(axes, metrics):
        col1 = f"{model1}_{metric}"
        col2 = f"{model2}_{metric}"

        x1 = sorted_df[col1].to_numpy()
        x2 = sorted_df[col2].to_numpy()
        y = range(len(sorted_df))

        # Draw lines colored by direction
        for i, (v1, v2) in enumerate(zip(x1, x2)):
            color = "green" if v2 >= v1 else "red"
            ax.hlines(y=i, xmin=min(v1, v2), xmax=max(v1, v2), color=color, alpha=0.6)

        # Draw dots
        ax.scatter(x1, y, color=MODEL_COLORS[0], label=model1, zorder=3, s=20)
        ax.scatter(x2, y, color=MODEL_COLORS[1], label=model2, zorder=3, s=20)

        ax.set_yticks(list(y))
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.set_xlim(0, 1)

    # Single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(model_names), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Metrics by Project (org_id|project_id)", fontsize=14, y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def compare_models(
    df: pl.DataFrame,
    thresholds: dict[str, float | dict[str, float]],
    output_dir: Path | None = None,
    min_group_rate_increase: float | None = 0.3,
    min_group_rate_decrease: float | None = None,
    write_csvs: bool = True,
    display_names: dict[str, str] | None = None,
) -> CompareResult:
    """
    Compare two models' grouping decisions and split data by (org_id, project_id).

    Args:
        df: DataFrame with cos_sim_{model-name} columns and a label column.
        thresholds: Dict mapping model-name to threshold. Value can be:
            - float: single threshold for all platforms.
            - dict[str, float]: per-platform thresholds. Must include a "default" key
              used for platforms not explicitly listed.
            First key = model1 (baseline), second key = model2 (new model).
        output_dir: Directory for writing CSVs. Required if write_csvs is True.
        min_group_rate_increase: Track projects where model2 GROUP rate is >= this value higher than model1. None to skip.
        min_group_rate_decrease: Track projects where model2 GROUP rate is >= this value lower than model1 (absolute).
            E.g., 0.10 means model2 has at least 10pp lower GROUP rate. None to skip.
        write_csvs: If True, write new.csv and merged.csv files for each project.
        display_names: Optional mapping from model names to display names for charts/tables.

    Outputs are written to output_dir / org_{org_id} / project_{project_id} /
    """
    if len(thresholds) != 2:
        raise ValueError(f"Expected exactly 2 models in thresholds, got {len(thresholds)}")
    if write_csvs and output_dir is None:
        raise ValueError("output_dir is required when write_csvs is True")

    report("Thresholds:", json.dumps(thresholds, indent=2))
    report(df["distance"].describe())
    report(f"GROUP rate: {(df['label'] == 'GROUP').mean():.2%}")

    # Print platform stats
    platform_stats = (
        df.group_by("platform")
        .agg(
            pl.len().alias("n_pairs"),
            pl.col("project_id").n_unique().alias("n_projects"),
            (pl.col("label") == "GROUP").mean().round(2).alias("label_GROUP_rate"),
        )
        .sort("platform")
        .with_columns((pl.col("n_pairs") / pl.col("n_pairs").sum()).round(2).alias("proportion"))
    )
    report("\nPlatform stats:")
    report(platform_stats)

    # Get model names in order
    model_names = list(thresholds.keys())
    model1, model2 = model_names[0], model_names[1]

    # Create prediction columns based on thresholds
    for model_name, threshold in thresholds.items():
        sim_col = f"cos_sim_{model_name}"
        pred_col = f"pred_{model_name}"

        if sim_col not in df.columns:
            raise ValueError(f"Column {sim_col} not found in dataframe. Available: {df.columns}")

        if isinstance(threshold, dict):
            # Per-platform thresholds: build a when/then chain
            threshold_default = threshold["default"]
            expr = pl.lit(threshold_default)
            for platform, thresh in threshold.items():
                if platform == "default":
                    continue
                expr = pl.when(pl.col("platform") == platform).then(pl.lit(thresh)).otherwise(expr)
            df = df.with_columns(
                pl.when(pl.col(sim_col) > expr).then(pl.lit("GROUP")).otherwise(pl.lit("SEPARATE")).alias(pred_col)
            )
        else:
            df = df.with_columns(
                pl.when(pl.col(sim_col) > threshold).then(pl.lit("GROUP")).otherwise(pl.lit("SEPARATE")).alias(pred_col)
            )

    pred1_col = f"pred_{model1}"
    pred2_col = f"pred_{model2}"

    # Compute and print overall metrics
    report(_compute_metrics(df, model_names))

    # Conditional probabilities: P(model2 GROUP | model1 prediction)
    prod_group = df.filter(pl.col(pred1_col) == "GROUP")
    prod_separate = df.filter(pl.col(pred1_col) == "SEPARATE")
    p_group_given_group = (prod_group[pred2_col] == "GROUP").mean() if len(prod_group) > 0 else float("nan")
    p_group_given_separate = (prod_separate[pred2_col] == "GROUP").mean() if len(prod_separate) > 0 else float("nan")
    report(f"\nP({model2} GROUP | {model1} GROUP)    = {p_group_given_group:.4f}")
    report(f"P({model2} GROUP | {model1} SEPARATE) = {p_group_given_separate:.4f}")

    # Conditional probability for close pairs (distance < 0.005)
    df_close = df.filter(pl.col("distance") < 0.005)
    close_group = df_close.filter(pl.col(pred1_col) == "GROUP")
    p_close = (close_group[pred2_col] == "GROUP").mean() if len(close_group) > 0 else float("nan")
    report(f"\nP({model2} GROUP | {model1} GROUP, distance < 0.005) = {p_close:.4f}  (n={len(close_group)})")

    # Columns to keep in output
    output_cols = [
        "platform",
        "query_stacktrace_string",
        "candidate_stacktrace_string",
        "label",
        "thinking_output",
        "response_output",
        "confidence_score",
        f"cos_sim_{model1}",
        f"cos_sim_{model2}",
        pred1_col,
        pred2_col,
    ]

    # Group by (org_id, project_id) and write split CSVs
    if not write_csvs:
        print("\n(Skipping CSV writes)")

    projects = []
    more_issues_projects = []
    all_project_metrics = []  # for computing project-averaged metrics
    total_projects = 0
    df_sorted = df.sort(["org_id", "project_id"])
    for (org_id, project_id), group_df in df_sorted.group_by(["org_id", "project_id"], maintain_order=True):
        total_projects += 1
        proj_dir = output_dir / f"org_{org_id}" / f"project_{project_id}"

        # Compute metrics for each model on this project
        model1_metrics = _compute_metrics_for_model(group_df, model1)
        model2_metrics = _compute_metrics_for_model(group_df, model2)
        model1_group_rate = model1_metrics["pred_GROUP_rate"]
        model2_group_rate = model2_metrics["pred_GROUP_rate"]
        delta = model2_group_rate - model1_group_rate

        # Store for project-averaged metrics and dumbbell plots
        all_project_metrics.append({"org_id": org_id, "project_id": project_id})
        all_project_metrics[-1].update({f"{model1}_{k}": v for k, v in model1_metrics.items()})
        all_project_metrics[-1].update({f"{model2}_{k}": v for k, v in model2_metrics.items()})

        # Compute merged/new dataframes
        new_df = group_df.filter((pl.col(pred1_col) == "GROUP") & (pl.col(pred2_col) == "SEPARATE"))
        merged_df = group_df.filter((pl.col(pred1_col) == "SEPARATE") & (pl.col(pred2_col) == "GROUP"))

        # Base project info reused for filtered project lists
        base_project_info = {
            "org_id": org_id,
            "project_id": project_id,
            "platform": group_df["platform"][0],
            "n_pairs": len(group_df),
            "label_GROUP_rate": (group_df["label"] == "GROUP").mean(),
            f"{model1}_GROUP_rate": model1_group_rate,
            f"{model1}_prec": model1_metrics["precision_GROUP"],
            f"{model1}_rec": model1_metrics["recall_GROUP"],
            f"{model2}_GROUP_rate": model2_group_rate,
            f"{model2}_prec": model2_metrics["precision_GROUP"],
            f"{model2}_rec": model2_metrics["recall_GROUP"],
            "_new_df": new_df.select(output_cols) if len(new_df) > 0 else None,
            "_merged_df": merged_df.select(output_cols) if len(merged_df) > 0 else None,
        }

        if min_group_rate_increase is not None and delta >= min_group_rate_increase:
            projects.append({**base_project_info, "group_rate_increase": delta})

        # Track projects where model2 creates more issues (groups less)
        if min_group_rate_decrease is not None:
            group_rate_decrease = -delta  # positive when model2 groups less
            if group_rate_decrease >= min_group_rate_decrease:
                more_issues_projects.append({**base_project_info, "group_rate_decrease": group_rate_decrease})

        if not write_csvs:
            continue

        # new.csv: model1 says GROUP but model2 says SEPARATE
        # (things that get split apart by model2 - new issues created)
        if len(new_df) > 0:
            proj_dir.mkdir(parents=True, exist_ok=True)
            new_path = proj_dir / "new.csv"
            new_df.select(output_cols).write_csv(new_path)

        # merged.csv: model1 says SEPARATE but model2 says GROUP
        # (things that get merged together by model2)
        if len(merged_df) > 0:
            proj_dir.mkdir(parents=True, exist_ok=True)
            merged_path = proj_dir / "merged.csv"
            merged_df.select(output_cols).write_csv(merged_path)

    # Print project-averaged metrics (skip NaNs when averaging)
    project_metrics_df = pl.DataFrame(all_project_metrics)
    report(f"\n=== Project-averaged metrics ({total_projects} projects) ===")
    avg_metrics = []
    for model_name in model_names:
        model_cols = [c for c in project_metrics_df.columns if c.startswith(f"{model_name}_")]
        avg = {c.replace(f"{model_name}_", ""): project_metrics_df[c].drop_nans().mean() for c in model_cols}
        avg_metrics.append({"model": model_name, **avg})
    report(pl.DataFrame(avg_metrics).with_columns(pl.col(pl.Float64).round(2)))

    # Apply display names for charts
    display_names = display_names or {}
    if display_names:
        # Rename columns in project_metrics_df
        rename_map = {}
        for col in project_metrics_df.columns:
            for old, new in display_names.items():
                if col.startswith(f"{old}_"):
                    rename_map[col] = col.replace(f"{old}_", f"{new}_", 1)
        project_metrics_df = project_metrics_df.rename(rename_map)
        # Rename pred_ and cos_sim_ columns in df
        df_rename_map = {}
        for col in df.columns:
            for old, new in display_names.items():
                if col == f"pred_{old}" or col == f"cos_sim_{old}":
                    df_rename_map[col] = col.replace(old, new)
        df = df.rename(df_rename_map)
    display_model_names = [display_names.get(n, n) for n in model_names]

    # Sort filtered project lists (printing deferred to upload step)
    if projects:
        projects.sort(key=lambda p: p["group_rate_increase"], reverse=True)
    if more_issues_projects:
        more_issues_projects.sort(key=lambda p: p["group_rate_decrease"], reverse=True)

    return CompareResult(
        df=df,
        model_names=display_model_names,
        project_metrics=project_metrics_df,
        projects=projects,
        more_issues_projects=more_issues_projects,
    )


def _add_token_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add token estimate columns to dataframe."""
    return df.with_columns(
        (pl.col("query_stacktrace_string").str.len_chars() // 4).alias("query_tokens"),
        (pl.col("candidate_stacktrace_string").str.len_chars() // 4).alias("candidate_tokens"),
    )


def compute_stacktrace_token_percentiles(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute percentile metrics for stacktrace token counts in the test set.

    Uses len(stacktrace) // 4 as a rough token count approximation.
    Computes stats for query, candidate, and combined (max of the two per pair).
    """
    df = _add_token_columns(df)

    # Add combined metrics
    df = df.with_columns(
        pl.max_horizontal("query_tokens", "candidate_tokens").alias("max_tokens"),
        (pl.col("query_tokens") + pl.col("candidate_tokens")).alias("total_tokens"),
    )

    # Compute percentiles for each token column
    token_cols = ["query_tokens", "candidate_tokens", "max_tokens", "total_tokens"]
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    rows = []
    for col in token_cols:
        row = {"metric": col}
        row["min"] = df[col].min()
        row["mean"] = df[col].mean()
        for p in percentiles:
            row[f"p{int(p * 100)}"] = df[col].quantile(p)
        row["max"] = df[col].max()
        rows.append(row)

    result = pl.DataFrame(rows).with_columns(pl.col(pl.Float64).cast(pl.Int64))

    report(f"\n=== Stacktrace Token Percentiles ({len(df)} pairs) ===")
    report("(Using len(stacktrace) // 4 as token approximation)")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        report(result)

    return result


def sweep_thresholds(
    df: pl.DataFrame,
    model_name: str,
    thresholds: list[float] = [0.80, 0.85, 0.87, 0.90],
) -> pl.DataFrame:
    """
    Show metrics for a single model at multiple similarity thresholds.

    Args:
        df: DataFrame with a cos_sim_{model_name} column and a label column.
        model_name: Model name (used to find cos_sim_ column).
        thresholds: List of similarity thresholds to evaluate.

    Returns:
        DataFrame with one row per threshold and metric columns.
    """
    sim_col = f"cos_sim_{model_name}"
    rows = []
    for thresh in thresholds:
        df_t = df.with_columns(
            pl.when(pl.col(sim_col) > thresh)
            .then(pl.lit("GROUP"))
            .otherwise(pl.lit("SEPARATE"))
            .alias(f"pred_{model_name}")
        )
        metrics = _compute_metrics_for_model(df_t, model_name)
        rows.append({"threshold": thresh, **metrics})

    result = pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(2))
    report(f"\n=== Threshold sweep for {model_name} ===")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        report(result)
    return result


def sweep_thresholds_by_project(
    df: pl.DataFrame,
    model_name: str,
    thresholds: list[float] = [0.80, 0.85, 0.87, 0.90],
    precision_floor: float = 0.8,
    harm_threshold: float = 0.05,
    thresholds_platform: dict[str, float] | None = None,
    baseline_model: str | None = None,
    baseline_threshold: float | None = None,
) -> None:
    """
    Show per-platform precision stats for platform-specific thresholds vs a baseline.

    Args:
        df: DataFrame with cos_sim_{model_name}, label, org_id, project_id columns.
        model_name: Model name (used to find cos_sim_ column).
        thresholds: List of similarity thresholds (unused when thresholds_platform is provided).
        precision_floor: Count projects with precision_GROUP below this absolute value.
        harm_threshold: Count projects where precision_GROUP drops by >= this vs baseline.
        thresholds_platform: Per-platform thresholds (platform -> threshold).
            Must include a "default" key for platforms not explicitly listed.
        baseline_model: If set, use this model at baseline_threshold as the baseline
            for computing deltas. Otherwise uses the highest threshold in the sweep.
        baseline_threshold: Threshold for the baseline model.
    """
    sim_col = f"cos_sim_{model_name}"
    pred_col = f"pred_{model_name}"
    thresholds_sorted = sorted(thresholds, reverse=True)

    def _compute_project_precisions(model: str, threshold: float) -> pl.DataFrame:
        """Compute per-project precision_GROUP for a model at a single flat threshold."""
        sc = f"cos_sim_{model}"
        pc = f"pred_{model}"
        df_t = df.with_columns(
            pl.when(pl.col(sc) > threshold).then(pl.lit("GROUP")).otherwise(pl.lit("SEPARATE")).alias(pc)
        )
        rows_project = []
        for (org_id, project_id), group_df in df_t.group_by(["org_id", "project_id"]):
            pred_group = group_df.filter(pl.col(pc) == "GROUP")
            prec = (pred_group["label"] == "GROUP").mean() if len(pred_group) > 0 else None
            rows_project.append(
                {
                    "org_id": org_id,
                    "project_id": project_id,
                    "platform": group_df["platform"][0],
                    "precision_GROUP": prec,
                }
            )
        return pl.DataFrame(rows_project)

    def _compute_project_precisions_per_platform(thresholds_platform: dict[str, float]) -> pl.DataFrame:
        """Compute per-project precision_GROUP using each platform's own threshold."""
        rows_project = []
        for (org_id, project_id), group_df in df.group_by(["org_id", "project_id"]):
            platform = group_df["platform"][0]
            thresh = thresholds_platform.get(platform, thresholds_platform["default"])
            df_t = group_df.with_columns(
                pl.when(pl.col(sim_col) > thresh).then(pl.lit("GROUP")).otherwise(pl.lit("SEPARATE")).alias(pred_col)
            )
            pred_group = df_t.filter(pl.col(pred_col) == "GROUP")
            prec = (pred_group["label"] == "GROUP").mean() if len(pred_group) > 0 else None
            rows_project.append(
                {"org_id": org_id, "project_id": project_id, "platform": platform, "precision_GROUP": prec}
            )
        return pl.DataFrame(rows_project)

    # Compute per-project precision_GROUP at each threshold
    project_precisions: dict[str, pl.DataFrame] = {}
    if thresholds_platform is not None:
        project_precisions["platform-specific"] = _compute_project_precisions_per_platform(thresholds_platform)
    for thresh in thresholds_sorted:
        project_precisions[str(thresh)] = _compute_project_precisions(model_name, thresh)

    # Baseline: external model if provided, else highest flat threshold
    if baseline_model is not None:
        baseline_key = f"{baseline_model}@{baseline_threshold}"
        project_precisions[baseline_key] = _compute_project_precisions(baseline_model, baseline_threshold)
    else:
        baseline_key = str(thresholds_sorted[0])
    baseline_df = project_precisions[baseline_key].rename({"precision_GROUP": "baseline_prec"})

    # Show per-platform stats: compare platform-specific thresholds vs baseline
    if thresholds_platform is not None and "platform-specific" in project_precisions:
        merged = project_precisions["platform-specific"].join(baseline_df, on=["org_id", "project_id"])
        merged = merged.with_columns((pl.col("precision_GROUP") - pl.col("baseline_prec")).alias("delta"))
        # Add per-project pair counts
        pairs_per_project = df.group_by(["org_id", "project_id"]).agg(pl.len().alias("n_pairs"))
        merged = merged.join(pairs_per_project, on=["org_id", "project_id"])
        rows_by_platform = []
        for (platform,), platform_df in merged.group_by("platform"):
            prec = platform_df["precision_GROUP"].drop_nulls().drop_nans()
            delta = platform_df["delta"].drop_nulls().drop_nans()
            rows_by_platform.append(
                {
                    "platform": platform,
                    "n_projects": len(prec),
                    "median_pairs": int(platform_df["n_pairs"].median()),
                    "mean": prec.mean(),
                    "p5": prec.quantile(0.05),
                    "p10": prec.quantile(0.10),
                    "p25": prec.quantile(0.25),
                    "median": prec.quantile(0.50),
                    f"below_{precision_floor}": (prec < precision_floor).mean(),
                    f"harmed_{harm_threshold:.0%}": (delta <= -harm_threshold).mean(),
                    "delta_mean": delta.mean(),
                    "delta_p5": delta.quantile(0.05),
                    "delta_p10": delta.quantile(0.10),
                }
            )
        by_platform = pl.DataFrame(rows_by_platform).sort("platform").with_columns(pl.col(pl.Float64).round(2))
        report(f"\n=== Per-project precision_GROUP: platform-specific vs {baseline_key} by platform ===")
        with pl.Config(tbl_rows=-1, tbl_cols=-1):
            report(by_platform)


def metrics_by_platform(
    df: pl.DataFrame,
    model_name: str,
    threshold: float | dict[str, float] = 0.99,
) -> pl.DataFrame:
    """
    Show metrics per platform at a given threshold.

    Args:
        df: DataFrame with cos_sim_{model_name}, label, and platform columns.
        model_name: Model name (used to find cos_sim_ column).
        threshold: Similarity threshold. Either a single float or a dict mapping
            platform -> threshold (with a "default" key for unlisted platforms).

    Returns:
        DataFrame with one row per platform and metric columns.
    """
    sim_col = f"cos_sim_{model_name}"
    pred_col = f"pred_{model_name}"

    if isinstance(threshold, dict):
        threshold_default = threshold["default"]
        expr = pl.lit(threshold_default)
        for platform, thresh in threshold.items():
            if platform == "default":
                continue
            expr = pl.when(pl.col("platform") == platform).then(pl.lit(thresh)).otherwise(expr)
        df_t = df.with_columns(
            pl.when(pl.col(sim_col) > expr).then(pl.lit("GROUP")).otherwise(pl.lit("SEPARATE")).alias(pred_col)
        )
    else:
        df_t = df.with_columns(
            pl.when(pl.col(sim_col) > threshold).then(pl.lit("GROUP")).otherwise(pl.lit("SEPARATE")).alias(pred_col)
        )

    rows = []
    for (platform,), platform_df in df_t.group_by("platform"):
        # Average metrics over projects to avoid large projects dominating
        project_metrics_list = []
        for _, proj_df in platform_df.group_by("project_id"):
            project_metrics_list.append(_compute_metrics_for_model(proj_df, model_name))
        avg_metrics = {
            k: sum(m[k] for m in project_metrics_list if m[k] == m[k])
            / sum(1 for m in project_metrics_list if m[k] == m[k])
            for k in project_metrics_list[0]
        }
        platform_threshold = threshold.get(platform, threshold["default"]) if isinstance(threshold, dict) else threshold
        rows.append(
            {
                "platform": platform,
                "n_pairs": len(platform_df),
                "n_projects": platform_df["project_id"].n_unique(),
                "label_GROUP_rate": (platform_df["label"] == "GROUP").mean(),
                "min_threshold": platform_threshold,
                **avg_metrics,
            }
        )

    result = pl.DataFrame(rows).sort("platform").with_columns(pl.col(pl.Float64).round(3))

    threshold_label = "platform-specific" if isinstance(threshold, dict) else str(threshold)
    report(f"\n=== Metrics by platform, avg over projects ({model_name} @ {threshold_label}) ===")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        report(result)

    return result


def find_threshold_by_platform(
    df: pl.DataFrame,
    model_name: str,
    min_precision: float | dict[str, float] = 0.95,
    thresholds: list[float] | None = None,
) -> pl.DataFrame:
    """
    Find the minimum threshold that achieves >= min_precision per platform.

    Args:
        df: DataFrame with cos_sim_{model_name}, label, and platform columns.
        model_name: Model name (used to find cos_sim_ column).
        min_precision: Minimum precision_GROUP required. Can be a single float
            or a dict mapping platform -> precision target.
        thresholds: Thresholds to sweep. Defaults to 0.50 to 0.99 in steps of 0.01.

    Returns:
        DataFrame with one row per platform showing the minimum threshold,
        plus metrics at that threshold.
    """
    if thresholds is None:
        thresholds = [round(x * 0.01, 2) for x in range(50, 100)]

    sim_col = f"cos_sim_{model_name}"
    pred_col = f"pred_{model_name}"
    thresholds_sorted = sorted(thresholds)

    precision_by_platform = min_precision if isinstance(min_precision, dict) else None

    rows = []
    for (platform,), platform_df in df.group_by("platform"):
        n_pairs = len(platform_df)
        n_projects = platform_df["project_id"].n_unique()
        label_group_rate = (platform_df["label"] == "GROUP").mean()
        threshold_found = None
        target_precision = precision_by_platform[platform] if precision_by_platform else min_precision

        # Walk thresholds from low to high; first one meeting precision is the minimum
        # Precision is averaged over projects to avoid large projects dominating
        for thresh in thresholds_sorted:
            df_t = platform_df.with_columns(
                pl.when(pl.col(sim_col) > thresh).then(pl.lit("GROUP")).otherwise(pl.lit("SEPARATE")).alias(pred_col)
            )
            # Compute per-project precision, then average
            project_precisions = []
            for _, proj_df in df_t.group_by("project_id"):
                pred_group = proj_df.filter(pl.col(pred_col) == "GROUP")
                if len(pred_group) > 0:
                    project_precisions.append((pred_group["label"] == "GROUP").mean())
            if not project_precisions:
                continue
            precision = sum(project_precisions) / len(project_precisions)
            if precision >= target_precision:
                metrics = _compute_metrics_for_model(df_t, model_name)
                threshold_found = thresh
                rows.append(
                    {
                        "platform": platform,
                        "n_pairs": n_pairs,
                        "n_projects": n_projects,
                        "label_GROUP_rate": label_group_rate,
                        "min_threshold": thresh,
                        **metrics,
                    }
                )
                break

        if threshold_found is None:
            rows.append(
                {
                    "platform": platform,
                    "n_pairs": n_pairs,
                    "n_projects": n_projects,
                    "label_GROUP_rate": label_group_rate,
                    "min_threshold": None,
                    "pred_GROUP_rate": None,
                    "precision_GROUP": None,
                    "precision_SEPARATE": None,
                    "recall_GROUP": None,
                    "recall_SEPARATE": None,
                }
            )

    result = pl.DataFrame(rows).sort("platform").with_columns(pl.col(pl.Float64).round(3))

    precision_label = "per-platform" if precision_by_platform else f"{min_precision:.0%}"
    report(f"\n=== Min threshold for >= {precision_label} avg project precision_GROUP by platform ({model_name}) ===")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        report(result)

    return result


def compare_metrics_by_stacktrace_length(
    df: pl.DataFrame,
    model_names: list[str],
    token_col: str = "query_tokens",
) -> None:
    """
    Compare model metrics for short (<=p10) and long (>=p90) stacktraces.

    Args:
        df: DataFrame with pred_{model_name} columns already added.
        model_names: List of model names to compare.
        token_col: Column to use for filtering (default: query_tokens).
    """
    # Add token columns if not present
    if token_col not in df.columns:
        df = _add_token_columns(df)

    # Compute percentile thresholds
    p10 = df[token_col].quantile(0.10)
    p90 = df[token_col].quantile(0.90)

    # Filter to short and long stacktraces
    short_df = df.filter(pl.col(token_col) <= p10)
    long_df = df.filter(pl.col(token_col) >= p90)

    # Print metrics for each bucket
    report(f"\n=== Short stacktraces ({token_col} <= p10 = {p10:.0f} tokens, {len(short_df)} pairs) ===")
    report(f"label GROUP rate: {(short_df['label'] == 'GROUP').mean():.2%}")
    report(_compute_metrics(short_df, model_names))

    report(f"\n=== Long stacktraces ({token_col} >= p90 = {p90:.0f} tokens, {len(long_df)} pairs) ===")
    report(f"label GROUP rate: {(long_df['label'] == 'GROUP').mean():.2%}")
    report(_compute_metrics(long_df, model_names))


COLS_JOIN = ["query_stacktrace_string", "candidate_stacktrace_string"]


def _load_thresholds(path_model: Path, threshold_default: float) -> float | dict[str, float]:
    """Load platform-specific thresholds from a YAML file next to the similarities CSV.

    If thresholds.yaml exists, returns a dict with platform keys and a "default" key.
    Otherwise returns threshold_default as a flat float.
    """
    path_yaml = path_model.parent / "thresholds.yaml"
    if not path_yaml.exists():
        return threshold_default
    with open(path_yaml) as f:
        thresholds = yaml.safe_load(f)
    thresholds.setdefault("default", threshold_default)
    print(f"Loaded thresholds from {path_yaml}")
    return thresholds


def _resolve_cos_sim(df: pl.DataFrame, dim: int) -> tuple[str, str]:
    """Find the cos_sim_{dim} column and return (column_name, dim_label)."""
    col = f"cos_sim_{dim}"
    if col not in df.columns:
        cols_cos_sim = [c for c in df.columns if c.startswith("cos_sim")]
        raise ValueError(f"Column {col} not found. Available: {cols_cos_sim}")
    return col, str(dim)


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

    if path_model1.resolve() == path_model2.resolve():
        col2, label_dim2 = _resolve_cos_sim(df1, dim_model2)
        if col1 == col2:
            raise ValueError(f"Both models resolve to the same column: {col1}")
        df = df1.rename({col1: f"cos_sim_{name_model1}", col2: f"cos_sim_{name_model2}"})
    else:
        df2 = pl.read_csv(path_model2)
        col2, label_dim2 = _resolve_cos_sim(df2, dim_model2)

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
    cols_drop = [c for c in df.columns if c.startswith("cos_sim") and c not in cols_keep]
    df = df.drop(cols_drop)
    return df, label_dim1, label_dim2


def _sync_gcs(gcs_dir: str) -> Path:
    """Sync a GCS similarities directory to a local cache and return the local similarities.csv path.

    Maps e.g. ``gs://grouping-data/runs/issue_grouping_v1/similarities/test_full``
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
    print(f"Syncing {gcs_dir} → {dir_local}")
    subprocess.run(["gcloud", "storage", "rsync", "-r", gcs_dir, str(dir_local)], check=True)
    return dir_local / "similarities.csv"


def main(
    gcs_model1: str,
    gcs_model2: str,
    name_model1: str,
    name_model2: str,
    dim_model1: int = 768,
    dim_model2: int = 768,
    threshold_model1: float = 0.99,
    threshold_model2: float = 0.90,
    min_group_rate_increase: float = 0.15,
    min_group_rate_decrease: float = 0.10,
    max_display_projects: int = 30,
    upload_sheets: bool = False,
    overwrite: bool = False,
):
    """
    Compare two models' grouping decisions on held-out data.

    Downloads similarity CSVs from GCS, joins them on stacktrace pairs, and runs
    a head-to-head comparison.

    Parameters
    ----------
    gcs_model1
        GCS path to model 1's similarities directory
        (e.g. gs://grouping-data/runs/issue_grouping_v1/similarities/test_full).
    gcs_model2
        GCS path to model 2's similarities directory
        (e.g. gs://grouping-data/runs/issue_grouping_v2/similarities/test_full).
    dim_model1
        Which cos_sim_{dim} column to use from model 1's CSV.
    dim_model2
        Which cos_sim_{dim} column to use from model 2's CSV.
    name_model1
        Short alias for model 1 used in output columns and file names.
    name_model2
        Short alias for model 2 used in output columns and file names.
    threshold_model1
        Default cosine similarity threshold for model 1. Overridden per-platform
        if a thresholds.yaml exists next to the CSV.
    threshold_model2
        Default cosine similarity threshold for model 2. Overridden per-platform
        if a thresholds.yaml exists next to the CSV.
    min_group_rate_increase
        Flag projects where model2 GROUP rate exceeds model1 by at least this amount.
    min_group_rate_decrease
        Flag projects where model2 GROUP rate is lower than model1 by at least this amount.
    max_display_projects
        Maximum number of flagged projects to display.
    upload_sheets
        If True, upload flagged projects to Google Sheets. Prompts you to authenticate using a secret JSON. Ask Kush for
        it or just make on yourself. TODO: put it in a secret or something easy.
    overwrite
        Allow overwriting an existing output directory. Without this flag the
        script exits with an error if the output directory already exists.
    """
    if upload_sheets:
        subprocess.run(
            [
                "gcloud",
                "auth",
                "application-default",
                "login",
                "--client-id-file=client_secret.json",
                "--scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive",
            ],
            check=True,
        )

    path1 = _sync_gcs(gcs_model1)
    path2 = _sync_gcs(gcs_model2)

    df, label_dim1, label_dim2 = _load_and_join(
        path1,
        path2,
        dim_model1,
        dim_model2,
        name_model1,
        name_model2,
    )
    print(f"Loaded {len(df)} pairs: {name_model1} (dim={label_dim1}) vs {name_model2} (dim={label_dim2})")

    name_dataset = path1.parent.name
    dir_output = (
        Path("eval/comparisons") / name_dataset / f"{name_model1}_dim{label_dim1}_vs_{name_model2}_dim{label_dim2}"
    )
    if dir_output.exists() and not overwrite:
        print()
        raise SystemExit(
            f"Output directory already exists: {dir_output}\n"
            "You're using the same dim_model1, dim_model2, name_model1, name_model2 values as a previous run.\n"
            "Pass --overwrite to replace it, or use a different name."
        )
    dir_output.mkdir(parents=True, exist_ok=True)

    thresholds = {
        name_model1: _load_thresholds(path1, threshold_model1),
        name_model2: _load_thresholds(path2, threshold_model2),
    }

    result = compare_models(
        df=df,
        thresholds=thresholds,
        output_dir=dir_output,
        min_group_rate_increase=min_group_rate_increase,
        min_group_rate_decrease=min_group_rate_decrease,
    )

    # Per-platform metrics for each model
    for name in [name_model1, name_model2]:
        metrics_by_platform(df, name, thresholds[name])

    # Find minimum threshold per platform for each model
    for name in [name_model1, name_model2]:
        find_threshold_by_platform(df, name)

    # Threshold sweep for model2
    sweep_thresholds(df, name_model2)
    threshold2 = thresholds[name_model2]
    sweep_thresholds_by_project(
        df,
        name_model2,
        thresholds_platform=threshold2 if isinstance(threshold2, dict) else None,
        baseline_model=name_model1,
        baseline_threshold=threshold_model1,
    )

    # Metrics by stacktrace length
    compare_metrics_by_stacktrace_length(result.df, result.model_names)

    # Plots
    fig = plot_metrics_by_platform(result.df, result.model_names)
    fig.savefig(dir_output / "metrics_by_platform.png", dpi=150, bbox_inches="tight")
    print(f"Saved {dir_output / 'metrics_by_platform.png'}")

    fig = plot_dumbbell_by_project(result.project_metrics, result.model_names)
    fig.savefig(dir_output / "dumbbell_by_project.png", dpi=150, bbox_inches="tight")
    print(f"Saved {dir_output / 'dumbbell_by_project.png'}")

    for name in result.model_names:
        fig = plot_similarity_distribution(result.df, name)
        path_plot = dir_output / f"similarity_distribution_{name}.png"
        fig.savefig(path_plot, dpi=150, bbox_inches="tight")
        print(f"Saved {path_plot}")

    if result.projects:
        print_projects(
            result.projects,
            description=f">= {min_group_rate_increase:.0%} group rate increase",
            max_projects=max_display_projects,
            stratify_by="platform",
        )

    if result.more_issues_projects:
        print_projects(
            result.more_issues_projects,
            description=f">= {min_group_rate_decrease:.0%} group rate decrease",
        )

    if upload_sheets:
        if result.projects:
            _upload_projects_to_sheets(
                result.projects,
                description=f"{name_model1} vs {name_model2} — group rate increase >= {min_group_rate_increase:.0%}",
            )
        if result.more_issues_projects:
            _upload_projects_to_sheets(
                result.more_issues_projects,
                description=f"{name_model1} vs {name_model2} — group rate decrease >= {min_group_rate_decrease:.0%}",
            )

    save_report(dir_output / "report.txt")
    print(f"\nResults written to {dir_output}")


if __name__ == "__main__":
    tapify(main, description=__doc__)
