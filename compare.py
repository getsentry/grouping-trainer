"""
gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/spreadsheets,https://www.googleapis.com/auth/drive
"""

import time
from dataclasses import dataclass
from pathlib import Path

import gspread
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from google.auth import default as google_auth_default
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

    high_delta_projects: list[dict]
    """Projects with group_rate_increase >= threshold, sorted descending."""

    more_issues_projects: list[dict]
    """Projects where model2 creates more issues (groups less) by >= min_group_rate_decrease."""


pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)

# Consistent colors: model1 (prod) = blue, model2 (gte-finetuned) = orange
MODEL_COLORS = ["#1f77b4", "#ff7f0e"]  # matplotlib default blue and orange


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


def _upload_high_delta_projects_to_sheets(spreadsheet_id: str, high_delta_projects: list[dict]) -> None:
    """Upload merged/new data for high-delta projects to Google Sheets."""
    creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/spreadsheets"])
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(spreadsheet_id)

    # Build list of (sheet_name, df) to upload
    uploads = []
    for project in high_delta_projects:
        prefix = f"org_{project['org_id']}|project_{project['project_id']}"
        if project["_new_df"] is not None:
            uploads.append((f"{prefix}|new", project["_new_df"]))
        if project["_merged_df"] is not None:
            uploads.append((f"{prefix}|merged", project["_merged_df"]))

    for sheet_name, df in tqdm(uploads, desc="Uploading to Google Sheets"):
        _upload_df_to_sheet(spreadsheet, sheet_name, df)

    print(f"Done! View at: {spreadsheet.url}")


def _compute_metrics_for_model(df: pl.DataFrame, model_name: str) -> dict:
    """Compute metrics for a single model on the given dataframe."""
    pred_col = f"pred_{model_name}"
    pred_group = df.filter(pl.col(pred_col) == "GROUP")
    pred_separate = df.filter(pl.col(pred_col) == "SEPARATE")
    label_group = df.filter(pl.col("label") == "GROUP")
    label_separate = df.filter(pl.col("label") == "SEPARATE")
    return {
        "pred_GROUP_rate": (df[pred_col] == "GROUP").mean(),
        "accuracy": (df[pred_col] == df["label"]).mean(),
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
        metrics["model"] = model_name
        metrics_rows.append(metrics)
    return pl.DataFrame(metrics_rows).with_columns(pl.col(pl.Float64).round(2))


def plot_metrics_by_platform(df: pl.DataFrame, model_names: list[str]) -> plt.Figure:
    """
    Create bar plots comparing 2 models grouped by platform.

    Args:
        df: DataFrame with prediction columns (pred_{model_name}) already added.
        model_names: List of model names (expects exactly 2).

    Returns:
        Figure with 3 subplots: precision_GROUP, recall_GROUP, accuracy.
    """
    # Compute metrics per platform for each model
    metrics_rows = []
    metrics_to_plot = None
    for (platform,), platform_df in df.group_by("platform"):
        for model_name in model_names:
            metrics = _compute_metrics_for_model(platform_df, model_name)
            if metrics_to_plot is None:
                metrics_to_plot = list(metrics.keys())
            metrics["platform"] = platform
            metrics["model"] = model_name
            metrics_rows.append(metrics)

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
    csv_path: Path,
    thresholds: dict[str, float],
    min_group_rate_increase: float | None = 0.3,
    min_group_rate_decrease: float | None = None,
    write_csvs: bool = True,
    display_names: dict[str, str] | None = None,
) -> CompareResult:
    """
    Compare two models' grouping decisions and split data by (org_id, project_id).

    Args:
        csv_path: Path to CSV with cos_sim_{model-name} columns.
        thresholds: Dict mapping model-name to cos_sim_threshold.
            First key = model1 (baseline), second key = model2 (new model).
        min_group_rate_increase: Track projects where model2 GROUP rate is >= this value higher than model1. None to skip.
        min_group_rate_decrease: Track projects where model2 GROUP rate is >= this value lower than model1 (absolute).
            E.g., 0.10 means model2 has at least 10pp lower GROUP rate. None to skip.
        write_csvs: If True, write new.csv and merged.csv files for each project.
        display_names: Optional mapping from model names to display names for charts/tables.

    Outputs are written to csv_path.parent / org_{org_id} / project_{project_id} /
    """
    if len(thresholds) != 2:
        raise ValueError(f"Expected exactly 2 models in thresholds, got {len(thresholds)}")

    df = pl.read_csv(csv_path)
    output_dir = csv_path.parent

    print("Thresholds:", ", ".join(f"{model}={thresh}" for model, thresh in thresholds.items()))
    print(df["distance"].describe())
    print(f"GROUP rate: {(df['label'] == 'GROUP').mean():.2%}")

    # Print platform stats
    platform_stats = (
        df.group_by("platform")
        .agg(
            pl.len().alias("n_pairs"),
            pl.col("project_id").n_unique().alias("n_projects"),
        )
        .sort("platform")
    )
    print("\nPlatform stats:")
    print(platform_stats)

    # Get model names in order
    model_names = list(thresholds.keys())
    model1, model2 = model_names[0], model_names[1]

    # Create prediction columns based on thresholds
    for model_name, threshold in thresholds.items():
        sim_col = f"cos_sim_{model_name}"
        pred_col = f"pred_{model_name}"

        if sim_col not in df.columns:
            raise ValueError(f"Column {sim_col} not found in dataframe. Available: {df.columns}")

        df = df.with_columns(
            pl.when(pl.col(sim_col) > threshold).then(pl.lit("GROUP")).otherwise(pl.lit("SEPARATE")).alias(pred_col)
        )

    pred1_col = f"pred_{model1}"
    pred2_col = f"pred_{model2}"

    # Compute and print overall metrics
    print(_compute_metrics(df, model_names))

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

    high_delta_projects = []
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
            high_delta_projects.append({**base_project_info, "group_rate_increase": delta})

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
    print(f"\n=== Project-averaged metrics ({total_projects} projects) ===")
    avg_metrics = []
    for model_name in model_names:
        model_cols = [c for c in project_metrics_df.columns if c.startswith(f"{model_name}_")]
        avg = {c.replace(f"{model_name}_", ""): project_metrics_df[c].drop_nans().mean() for c in model_cols}
        avg["model"] = model_name
        avg_metrics.append(avg)
    print(pl.DataFrame(avg_metrics).with_columns(pl.col(pl.Float64).round(2)))

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

    # Helper to rename columns for display
    def _apply_display_names(df: pl.DataFrame) -> pl.DataFrame:
        if not display_names:
            return df
        rename_map = {}
        for col in df.columns:
            for old, new in display_names.items():
                if old in col:
                    rename_map[col] = col.replace(old, new)
        return df.rename(rename_map)

    # Print projects with increased grouping
    if high_delta_projects:
        # Sort by group_rate_increase descending
        high_delta_projects.sort(key=lambda p: p["group_rate_increase"], reverse=True)

        # Exclude internal dataframe columns for display
        display_cols = [k for k in high_delta_projects[0] if not k.startswith("_")]
        high_delta_df = _apply_display_names(
            pl.DataFrame([{k: v for k, v in p.items() if k in display_cols} for p in high_delta_projects]).with_columns(
                pl.col(pl.Float64).round(2)
            )
        )
        print(
            f"\n=== Projects with >= {min_group_rate_increase:.0%} increase in grouping ({len(high_delta_projects)}/{total_projects} projects) ==="
        )
        with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=1000):
            print(high_delta_df)

    # Print projects where model2 creates more issues
    if more_issues_projects:
        more_issues_projects.sort(key=lambda p: p["group_rate_decrease"], reverse=True)

        display_cols = [k for k in more_issues_projects[0] if not k.startswith("_")]
        more_issues_df = _apply_display_names(
            pl.DataFrame([{k: v for k, v in p.items() if k in display_cols} for p in more_issues_projects])
            .with_columns(pl.col(pl.Float64).round(2))
            .rename({"group_rate_decrease": "decrease"})
        )
        print(
            f"\n=== Projects with >= {min_group_rate_decrease:.0%} decrease in grouping "
            f"({len(more_issues_projects)}/{total_projects} projects) ==="
        )
        with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=1000):
            print(more_issues_df)

    return CompareResult(
        df=df,
        model_names=display_model_names,
        project_metrics=project_metrics_df,
        high_delta_projects=high_delta_projects,
        more_issues_projects=more_issues_projects,
    )


if __name__ == "__main__":
    # csv_path = Path("similarities/2026-01-08-13-28-41-sentry/similarities.csv")
    # thresholds = {
    #     "prod": 0.99,
    #     "gte-finetuned": 0.60,
    # }
    # min_project_size = None
    # max_model1_group_rate = None

    csv_path = Path("similarities/2026-01-08-13-19-08-test/similarities.csv")
    thresholds = {
        "prod": 0.99,
        "gte-finetuned": 0.85,
    }

    result = compare_models(
        csv_path=csv_path,
        thresholds=thresholds,
        min_group_rate_increase=0.3,
        min_group_rate_decrease=0.15,
        write_csvs=True,
        display_names={"gte-finetuned": "new"},
    )

    fig = plot_metrics_by_platform(result.df, result.model_names)
    fig.savefig(csv_path.parent / "metrics_by_platform.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {csv_path.parent / 'metrics_by_platform.png'}")

    fig = plot_dumbbell_by_project(result.project_metrics, result.model_names)
    fig.savefig(csv_path.parent / "dumbbell_by_project.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {csv_path.parent / 'dumbbell_by_project.png'}")

    # Upload to Google Sheets (slowest step, do last)
    # spreadsheet_id = "1-aHK2-ZO8WwmuHyP4gRRCtiPWQtYyZr4qcWkVa4Ptjw"
    # if spreadsheet_id and result.high_delta_projects:
    #     _upload_high_delta_projects_to_sheets(spreadsheet_id, result.high_delta_projects)

    # spreadsheet_id_more_issues = "1u59V6D0G8WSidg9Jc43KcE22bLRrRs4B0twCINS7IbA"
    # if spreadsheet_id_more_issues and result.more_issues_projects:
    #     _upload_high_delta_projects_to_sheets(spreadsheet_id_more_issues, result.more_issues_projects)
