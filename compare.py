from pathlib import Path

import polars as pl

pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)


def _compute_metrics(df: pl.DataFrame, model_names: list[str], thresholds: dict[str, float]) -> pl.DataFrame:
    """Compute metrics for each model on the given dataframe."""
    metrics_rows = []
    for model_name in model_names:
        pred_col = f"pred_{model_name}"
        pred_group = df.filter(pl.col(pred_col) == "GROUP")
        pred_separate = df.filter(pl.col(pred_col) == "SEPARATE")
        label_group = df.filter(pl.col("label") == "GROUP")
        label_separate = df.filter(pl.col("label") == "SEPARATE")
        metrics_rows.append(
            {
                "model": model_name,
                "pred_GROUP_rate": (df[pred_col] == "GROUP").mean(),
                "accuracy": (df[pred_col] == df["label"]).mean(),
                "PPV": (pred_group["label"] == "GROUP").mean() if len(pred_group) > 0 else float("nan"),
                "NPV": (pred_separate["label"] == "SEPARATE").mean() if len(pred_separate) > 0 else float("nan"),
                "recall_GROUP": (label_group[pred_col] == "GROUP").mean() if len(label_group) > 0 else float("nan"),
                "recall_SEPARATE": (label_separate[pred_col] == "SEPARATE").mean()
                if len(label_separate) > 0
                else float("nan"),
            }
        )
    return pl.DataFrame(metrics_rows).with_columns(pl.col(pl.Float64).round(2))


def compare_models(
    csv_path: Path,
    thresholds: dict[str, float],
    per_project_metrics: bool = False,
    min_project_size: int | None = 2000,
    max_model1_group_rate: float | None = None,
    write_csvs: bool = True,
) -> None:
    """
    Compare two models' grouping decisions and split data by (org_id, project_id).

    Args:
        csv_path: Path to CSV with cos_sim_{model-name} columns.
        thresholds: Dict mapping model-name to cos_sim_threshold.
            First key = model1 (baseline), second key = model2 (new model).
        per_project_metrics: If True, print metrics for each project in addition to overall.
        min_project_size: Minimum number of rows in a project to show per-project metrics. None for no filter.
        max_model1_group_rate: Only show projects where model1's pred_GROUP_rate <= this value. None for no filter.
        write_csvs: If True, write new.csv and merged.csv files for each project.

    Outputs are written to csv_path.parent / org_{org_id} / project_{project_id} /
    """
    if len(thresholds) != 2:
        raise ValueError(f"Expected exactly 2 models in thresholds, got {len(thresholds)}")

    df = pl.read_csv(csv_path)
    output_dir = csv_path.parent

    print("Thresholds:", ", ".join(f"{model}={thresh}" for model, thresh in thresholds.items()))
    print(df["distance"].describe())
    print(f"GROUP rate: {(df['label'] == 'GROUP').mean():.2%}")

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
    print(_compute_metrics(df, model_names, thresholds))

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
    df_sorted = df.sort(["org_id", "project_id"])
    for (org_id, project_id), group_df in df_sorted.group_by(["org_id", "project_id"], maintain_order=True):
        if per_project_metrics and (min_project_size is None or len(group_df) >= min_project_size):
            model1_group_rate = (group_df[pred1_col] == "GROUP").mean()
            if max_model1_group_rate is None or model1_group_rate <= max_model1_group_rate:
                group_rate = (group_df["label"] == "GROUP").mean()
                print(f"\n=== org_{org_id} / project_{project_id} (n={len(group_df)}, {group_rate:.0%} GROUP) ===")
                print(_compute_metrics(group_df, model_names, thresholds))

        if not write_csvs:
            continue

        proj_dir = output_dir / f"org_{org_id}" / f"project_{project_id}"
        proj_dir.mkdir(parents=True, exist_ok=True)

        # new.csv: model1 says GROUP but model2 says SEPARATE
        # (things that get split apart by model2 - new issues created)
        new_df = group_df.filter((pl.col(pred1_col) == "GROUP") & (pl.col(pred2_col) == "SEPARATE"))
        if len(new_df) > 0:
            new_path = proj_dir / "new.csv"
            new_df.select(output_cols).write_csv(new_path)
            print(f"Wrote to {new_path}")

        # merged.csv: model1 says SEPARATE but model2 says GROUP
        # (things that get merged together by model2)
        merged_df = group_df.filter((pl.col(pred1_col) == "SEPARATE") & (pl.col(pred2_col) == "GROUP"))
        if len(merged_df) > 0:
            merged_path = proj_dir / "merged.csv"
            merged_df.select(output_cols).write_csv(merged_path)
            print(f"Wrote to {merged_path}")


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
    min_project_size = 2_000  # may be biased towards severe undergrouping
    max_model1_group_rate = 0.2  # to see only projects where prod undergroups

    compare_models(
        csv_path=csv_path,
        thresholds=thresholds,
        min_project_size=min_project_size,
        max_model1_group_rate=max_model1_group_rate,
        per_project_metrics=True,
        write_csvs=False,
    )
