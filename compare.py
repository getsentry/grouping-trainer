import time
from pathlib import Path

import gspread
import matplotlib.pyplot as plt
import polars as pl
from google.auth import default as google_auth_default
from tqdm.auto import tqdm

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)


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
    wide_cols = {"query_stacktrace_string", "candidate_stacktrace_string"}

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
        pivot_df.plot(kind="bar", ax=ax, rot=45, legend=False)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylim(0, 1)

    # Single legend for the whole figure (top center)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(model_names), bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # make room for legend on top
    return fig


def compare_models(
    csv_path: Path,
    thresholds: dict[str, float],
    min_delta: float | None = 0.3,
    write_csvs: bool = True,
    spreadsheet_id: str | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """
    Compare two models' grouping decisions and split data by (org_id, project_id).

    Args:
        csv_path: Path to CSV with cos_sim_{model-name} columns.
        thresholds: Dict mapping model-name to cos_sim_threshold.
            First key = model1 (baseline), second key = model2 (new model).
        min_delta: Print projects where absolute delta in pred_GROUP_rate >= this value. None to skip.
        write_csvs: If True, write new.csv and merged.csv files for each project.
        spreadsheet_id: If provided, upload merged/new data for high-delta projects to this Google Sheet.

    Returns:
        Tuple of (df with pred_{model_name} columns added, list of model names).

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

        # Store for project-averaged metrics
        all_project_metrics.append({f"{model1}_{k}": v for k, v in model1_metrics.items()})
        all_project_metrics[-1].update({f"{model2}_{k}": v for k, v in model2_metrics.items()})

        # Compute merged/new dataframes
        new_df = group_df.filter((pl.col(pred1_col) == "GROUP") & (pl.col(pred2_col) == "SEPARATE"))
        merged_df = group_df.filter((pl.col(pred1_col) == "SEPARATE") & (pl.col(pred2_col) == "GROUP"))

        if min_delta is not None and abs(delta) >= min_delta:
            high_delta_projects.append(
                {
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
                    "delta": delta,
                    # "path": str(proj_dir),
                    "_new_df": new_df.select(output_cols) if len(new_df) > 0 else None,
                    "_merged_df": merged_df.select(output_cols) if len(merged_df) > 0 else None,
                }
            )

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

    # Print high delta projects
    if high_delta_projects:
        # Exclude internal dataframe columns for display
        display_cols = [k for k in high_delta_projects[0] if not k.startswith("_")]
        high_delta_df = (
            pl.DataFrame([{k: v for k, v in p.items() if k in display_cols} for p in high_delta_projects])
            .sort("delta", descending=True)
            .with_columns(pl.col(pl.Float64).round(2))
        )
        print(
            f"\n=== Projects with |delta| >= {min_delta:.0%} ({len(high_delta_projects)}/{total_projects} projects) ==="
        )
        with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=1000):
            print(high_delta_df)

        # Upload to Google Sheets if requested
        if spreadsheet_id:
            _upload_high_delta_projects_to_sheets(spreadsheet_id, high_delta_projects)

    return df, model_names


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
    spreadsheet_id = "1-aHK2-ZO8WwmuHyP4gRRCtiPWQtYyZr4qcWkVa4Ptjw"

    df, model_names = compare_models(
        csv_path=csv_path,
        thresholds=thresholds,
        min_delta=0.3,
        write_csvs=True,
        spreadsheet_id=spreadsheet_id,
    )

    fig = plot_metrics_by_platform(df, model_names)
    fig.savefig(csv_path.parent / "metrics_by_platform.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot to {csv_path.parent / 'metrics_by_platform.png'}")
