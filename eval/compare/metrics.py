"""Metric computation, threshold sweeps, and the head-to-head `compare_models` orchestrator."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .report import emit, emit_table


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


def _threshold_pred_expr(model_name: str, threshold: float | dict[str, float]) -> pl.Expr:
    """Build a polars expression that labels each row "GROUP" or "SEPARATE" based on threshold.

    `threshold` may be a single float or a per-platform dict (with a "default" key) — for the
    dict case the expression resolves the platform-specific cutoff via a when/then chain.
    """
    sim_col = f"cos_sim_{model_name}"
    if isinstance(threshold, dict):
        cutoff = pl.lit(threshold["default"])
        for platform, thresh in threshold.items():
            if platform == "default":
                continue
            cutoff = pl.when(pl.col("platform") == platform).then(pl.lit(thresh)).otherwise(cutoff)
    else:
        cutoff = pl.lit(threshold)
    return pl.when(pl.col(sim_col) > cutoff).then(pl.lit("GROUP")).otherwise(pl.lit("SEPARATE"))


def _apply_threshold(df: pl.DataFrame, model_name: str, threshold: float | dict[str, float]) -> pl.DataFrame:
    """Return a copy of `df` with a `pred_{model_name}` column added."""
    sim_col = f"cos_sim_{model_name}"
    if sim_col not in df.columns:
        raise ValueError(f"Column {sim_col} not found in dataframe. Available: {df.columns}")
    return df.with_columns(_threshold_pred_expr(model_name, threshold).alias(f"pred_{model_name}"))


def _compute_metrics_for_model(df: pl.DataFrame, model_name: str) -> dict:
    """Compute metrics for a single model on the given dataframe."""
    pred_col = f"pred_{model_name}"
    pred_is_group = pl.col(pred_col) == "GROUP"
    label_is_group = pl.col("label") == "GROUP"
    # Single-pass aggregation; .mean() on a boolean expression filtered to an empty subset returns null,
    # which we then map to NaN below to preserve the legacy nan-on-empty contract.
    row = df.select(
        pred_is_group.mean().alias("pred_GROUP_rate"),
        label_is_group.filter(pred_is_group).mean().alias("precision_GROUP"),
        (~label_is_group).filter(~pred_is_group).mean().alias("precision_SEPARATE"),
        pred_is_group.filter(label_is_group).mean().alias("recall_GROUP"),
        (~pred_is_group).filter(~label_is_group).mean().alias("recall_SEPARATE"),
    ).row(0, named=True)
    return {key: (float("nan") if value is None else value) for key, value in row.items()}


def _compute_metrics_avg_over_projects(df: pl.DataFrame, model_name: str) -> dict[str, float]:
    """Compute metrics averaged over projects so large projects don't dominate."""
    metrics_per_project = []
    for _, df_project in df.group_by("project_id"):
        metrics_per_project.append(_compute_metrics_for_model(df_project, model_name))
    avg_metrics = {}
    for key in metrics_per_project[0]:
        values_valid = [metrics[key] for metrics in metrics_per_project if metrics[key] == metrics[key]]
        avg_metrics[key] = sum(values_valid) / len(values_valid) if values_valid else float("nan")
    return avg_metrics


def _compute_metrics(df: pl.DataFrame, model_names: list[str]) -> pl.DataFrame:
    """Compute metrics for each model on the given dataframe."""
    metrics_rows = []
    for model_name in model_names:
        metrics = _compute_metrics_for_model(df, model_name)
        metrics_rows.append({"model": model_name, **metrics})
    return pl.DataFrame(metrics_rows).with_columns(pl.col(pl.Float64).round(2))


def _compute_conditional_probabilities(df: pl.DataFrame, model1: str, model2: str) -> tuple[float, float, float, int]:
    """Return (P(m2=G | m1=G), P(m2=G | m1=S), P(m2=G | m1=G, distance<0.005), n_close)."""
    pred1_col = f"pred_{model1}"
    pred2_col = f"pred_{model2}"
    pred1_group = df.filter(pl.col(pred1_col) == "GROUP")
    pred1_separate = df.filter(pl.col(pred1_col) == "SEPARATE")
    p_group_given_group = (
        float((pred1_group[pred2_col] == "GROUP").mean()) if len(pred1_group) > 0 else float("nan")  # type: ignore[arg-type]
    )
    p_group_given_separate = (
        float((pred1_separate[pred2_col] == "GROUP").mean()) if len(pred1_separate) > 0 else float("nan")  # type: ignore[arg-type]
    )
    close_group = df.filter((pl.col("distance") < 0.005) & (pl.col(pred1_col) == "GROUP"))
    p_close = float((close_group[pred2_col] == "GROUP").mean()) if len(close_group) > 0 else float("nan")  # type: ignore[arg-type]
    return p_group_given_group, p_group_given_separate, p_close, len(close_group)


_OUTPUT_COLS_BASE = (
    "platform",
    "query_stacktrace_string",
    "candidate_stacktrace_string",
    "label",
    "thinking_output",
    "response_output",
    "confidence_score",
)


def _iterate_projects(
    df: pl.DataFrame,
    model1: str,
    model2: str,
    output_dir: Path | None,
    write_csvs: bool,
    min_group_rate_increase: float | None,
    min_group_rate_decrease: float | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Walk per-project groups, compute metrics, optionally write new/merged CSVs.

    Returns (all_project_metrics, projects_more_grouping, projects_more_issues).
    """
    pred1_col = f"pred_{model1}"
    pred2_col = f"pred_{model2}"
    output_cols = [*_OUTPUT_COLS_BASE, f"cos_sim_{model1}", f"cos_sim_{model2}", pred1_col, pred2_col]

    all_project_metrics: list[dict] = []
    projects_more_grouping: list[dict] = []
    projects_more_issues: list[dict] = []
    df_sorted = df.sort(["org_id", "project_id"])
    for (org_id, project_id), group_df in df_sorted.group_by(["org_id", "project_id"], maintain_order=True):
        proj_dir = output_dir / f"org_{org_id}" / f"project_{project_id}" if output_dir is not None else None

        metrics_model1 = _compute_metrics_for_model(group_df, model1)
        metrics_model2 = _compute_metrics_for_model(group_df, model2)
        group_rate_model1 = metrics_model1["pred_GROUP_rate"]
        group_rate_model2 = metrics_model2["pred_GROUP_rate"]
        delta = group_rate_model2 - group_rate_model1

        row_metrics = {"org_id": org_id, "project_id": project_id}
        row_metrics.update({f"{model1}_{metric}": value for metric, value in metrics_model1.items()})
        row_metrics.update({f"{model2}_{metric}": value for metric, value in metrics_model2.items()})
        all_project_metrics.append(row_metrics)

        new_df = group_df.filter((pl.col(pred1_col) == "GROUP") & (pl.col(pred2_col) == "SEPARATE"))
        merged_df = group_df.filter((pl.col(pred1_col) == "SEPARATE") & (pl.col(pred2_col) == "GROUP"))

        base_project_info = {
            "org_id": org_id,
            "project_id": project_id,
            "platform": group_df["platform"][0],
            "n_pairs": len(group_df),
            "label_GROUP_rate": (group_df["label"] == "GROUP").mean(),
            f"{model1}_GROUP_rate": group_rate_model1,
            f"{model1}_prec": metrics_model1["precision_GROUP"],
            f"{model1}_rec": metrics_model1["recall_GROUP"],
            f"{model2}_GROUP_rate": group_rate_model2,
            f"{model2}_prec": metrics_model2["precision_GROUP"],
            f"{model2}_rec": metrics_model2["recall_GROUP"],
            "_new_df": new_df.select(output_cols) if len(new_df) > 0 else None,
            "_merged_df": merged_df.select(output_cols) if len(merged_df) > 0 else None,
        }

        if min_group_rate_increase is not None and delta >= min_group_rate_increase:
            projects_more_grouping.append({**base_project_info, "group_rate_increase": delta})
        if min_group_rate_decrease is not None and -delta >= min_group_rate_decrease:
            projects_more_issues.append({**base_project_info, "group_rate_decrease": -delta})

        if not write_csvs or proj_dir is None:
            continue
        # new.csv: model1 says GROUP but model2 says SEPARATE
        # (things that get split apart by model2 - new issues created)
        if len(new_df) > 0:
            proj_dir.mkdir(parents=True, exist_ok=True)
            new_df.select(output_cols).write_csv(proj_dir / "new.csv")
        # merged.csv: model1 says SEPARATE but model2 says GROUP
        # (things that get merged together by model2)
        if len(merged_df) > 0:
            proj_dir.mkdir(parents=True, exist_ok=True)
            merged_df.select(output_cols).write_csv(proj_dir / "merged.csv")

    return all_project_metrics, projects_more_grouping, projects_more_issues


def _rename_for_display(
    df: pl.DataFrame, project_metrics_df: pl.DataFrame, display_names: dict[str, str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rename pred_/cos_sim_/per-project metric columns from internal model names to display names."""
    if not display_names:
        return df, project_metrics_df

    rename_metrics = {}
    for col in project_metrics_df.columns:
        for old, new in display_names.items():
            if col.startswith(f"{old}_"):
                rename_metrics[col] = col.replace(f"{old}_", f"{new}_", 1)
    project_metrics_df = project_metrics_df.rename(rename_metrics)

    rename_df = {}
    for old, new in display_names.items():
        for prefix in ("pred_", "cos_sim_"):
            col = f"{prefix}{old}"
            if col in df.columns:
                rename_df[col] = f"{prefix}{new}"
    df = df.rename(rename_df)
    return df, project_metrics_df


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
        min_group_rate_increase: Track projects where model2 GROUP rate is >= this value higher than model1. None skips.
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

    model_names = list(thresholds.keys())
    model1, model2 = model_names

    for model_name, threshold in thresholds.items():
        df = _apply_threshold(df, model_name, threshold)

    metrics_overall = _compute_metrics(df, model_names)
    p_group_given_group, p_group_given_separate, p_close, n_close = _compute_conditional_probabilities(
        df, model1, model2
    )

    if not write_csvs:
        print("\n(Skipping CSV writes)")

    all_project_metrics, projects_more_grouping, projects_more_issues = _iterate_projects(
        df,
        model1,
        model2,
        output_dir if write_csvs else None,
        write_csvs,
        min_group_rate_increase,
        min_group_rate_decrease,
    )
    total_projects = len(all_project_metrics)
    project_metrics_df = pl.DataFrame(all_project_metrics)

    # --- Report sections (order matters for the markdown document) ---
    emit("\n### Overall metrics\n")
    emit(metrics_overall)

    emit(f"\n### Project-averaged metrics ({total_projects} projects)\n")
    avg_rows = []
    for model_name in model_names:
        cols_model = [col for col in project_metrics_df.columns if col.startswith(f"{model_name}_")]
        avg = {col.replace(f"{model_name}_", ""): project_metrics_df[col].drop_nans().mean() for col in cols_model}
        avg_rows.append({"model": model_name, **avg})
    emit(pl.DataFrame(avg_rows).with_columns(pl.col(pl.Float64).round(2)))

    emit("\n### Conditional probabilities\n")
    emit(f"P({model2} GROUP | {model1} GROUP)    = {p_group_given_group:.4f}\n")
    emit(f"P({model2} GROUP | {model1} SEPARATE) = {p_group_given_separate:.4f}\n")
    emit(f"P({model2} GROUP | {model1} GROUP, distance < 0.005) = {p_close:.4f}  (n={n_close})")

    emit("\n### Thresholds\n")
    emit("```json\n" + json.dumps(thresholds, indent=2) + "\n```")

    emit("\n### Distance distribution\n")
    emit(df["distance"].describe())
    emit(f"\nGROUP rate: {float((df['label'] == 'GROUP').mean()):.2%}")  # type: ignore[arg-type]

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
    emit("\n### Platform stats\n")
    emit(platform_stats)

    display_names = display_names or {}
    df, project_metrics_df = _rename_for_display(df, project_metrics_df, display_names)
    display_model_names = [display_names.get(name, name) for name in model_names]

    if projects_more_grouping:
        projects_more_grouping.sort(key=lambda project: project["group_rate_increase"], reverse=True)
    if projects_more_issues:
        projects_more_issues.sort(key=lambda project: project["group_rate_decrease"], reverse=True)

    return CompareResult(
        df=df,
        model_names=display_model_names,
        project_metrics=project_metrics_df,
        projects=projects_more_grouping,
        more_issues_projects=projects_more_issues,
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
    df = df.with_columns(
        pl.max_horizontal("query_tokens", "candidate_tokens").alias("max_tokens"),
        (pl.col("query_tokens") + pl.col("candidate_tokens")).alias("total_tokens"),
    )

    token_cols = ["query_tokens", "candidate_tokens", "max_tokens", "total_tokens"]
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

    rows = []
    for col in token_cols:
        row: dict[str, Any] = {"metric": col}
        row["min"] = df[col].min()
        row["mean"] = df[col].mean()
        for percentile in percentiles:
            row[f"p{int(percentile * 100)}"] = df[col].quantile(percentile)
        row["max"] = df[col].max()
        rows.append(row)

    result = pl.DataFrame(rows).with_columns(pl.col(pl.Float64).cast(pl.Int64))

    emit(f"\n### Stacktrace token percentiles ({len(df)} pairs)\n")
    emit("(Using len(stacktrace) // 4 as token approximation)")
    emit_table(result)

    return result


def sweep_thresholds(
    df: pl.DataFrame,
    model_name: str,
    thresholds: list[float] | None = None,
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
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.87, 0.90]
    rows = []
    for thresh in thresholds:
        df_t = _apply_threshold(df, model_name, thresh)
        metrics = _compute_metrics_for_model(df_t, model_name)
        rows.append({"threshold": thresh, **metrics})

    result = pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(2))
    emit(f"\n### Threshold sweep for {model_name}\n")
    emit_table(result)
    return result


def _project_precisions(df: pl.DataFrame, model_name: str, threshold: float | dict[str, float]) -> pl.DataFrame:
    """Per-project precision_GROUP at a single threshold (flat or per-platform)."""
    pred_col = f"pred_{model_name}"
    df_t = _apply_threshold(df, model_name, threshold)
    rows = []
    for (org_id, project_id), group_df in df_t.group_by(["org_id", "project_id"]):
        pred_group = group_df.filter(pl.col(pred_col) == "GROUP")
        prec = (pred_group["label"] == "GROUP").mean() if len(pred_group) > 0 else None
        rows.append(
            {
                "org_id": org_id,
                "project_id": project_id,
                "platform": group_df["platform"][0],
                "precision_GROUP": prec,
            }
        )
    return pl.DataFrame(rows)


def sweep_thresholds_by_project(
    df: pl.DataFrame,
    model_name: str,
    thresholds: list[float] | None = None,
    precision_floor: float = 0.8,
    harm_threshold: float = 0.05,
    thresholds_platform: dict[str, float] | None = None,
    baseline_model: str | None = None,
    baseline_threshold: float | dict[str, float] | None = None,
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
        baseline_threshold: Threshold for the baseline model. Can be a float or a
            per-platform dict (with a "default" key), same format as thresholds_platform.
    """
    if thresholds is None:
        thresholds = [0.80, 0.85, 0.87, 0.90]
    thresholds_sorted = sorted(thresholds, reverse=True)

    project_precisions: dict[str, pl.DataFrame] = {}
    if thresholds_platform is not None:
        project_precisions["platform-specific"] = _project_precisions(df, model_name, thresholds_platform)
    for thresh in thresholds_sorted:
        project_precisions[str(thresh)] = _project_precisions(df, model_name, thresh)

    if baseline_model is not None:
        if isinstance(baseline_threshold, dict):
            baseline_key = f"{baseline_model}@platform-specific"
        else:
            assert isinstance(baseline_threshold, float)
            baseline_key = f"{baseline_model}@{baseline_threshold}"
        project_precisions[baseline_key] = _project_precisions(df, baseline_model, baseline_threshold)
    else:
        baseline_key = str(thresholds_sorted[0])
    baseline_df = project_precisions[baseline_key].rename({"precision_GROUP": "baseline_prec"})

    if thresholds_platform is not None and "platform-specific" in project_precisions:
        merged = project_precisions["platform-specific"].join(baseline_df, on=["org_id", "project_id"])
        merged = merged.with_columns((pl.col("precision_GROUP") - pl.col("baseline_prec")).alias("delta"))
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
                    "median_pairs": int(platform_df["n_pairs"].median()),  # type: ignore[arg-type]
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
        emit(f"\n### Per-project precision_GROUP: platform-specific vs {baseline_key} by platform\n")
        emit_table(by_platform)


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
    df_t = _apply_threshold(df, model_name, threshold)

    rows = []
    for (platform_obj,), platform_df in df_t.group_by("platform"):
        platform = str(platform_obj)
        avg_metrics = _compute_metrics_avg_over_projects(platform_df, model_name)
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

    threshold_label = "platform-specific" if isinstance(threshold, dict) else f"threshold={threshold}"
    emit(f"\n### Metrics by platform, avg over projects ({model_name}, {threshold_label})\n")
    emit_table(result)

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
        thresholds = [round(step * 0.01, 2) for step in range(50, 100)]

    pred_col = f"pred_{model_name}"
    thresholds_sorted = sorted(thresholds)
    precision_by_platform = min_precision if isinstance(min_precision, dict) else None

    rows = []
    for (platform_obj,), platform_df in df.group_by("platform"):
        platform = str(platform_obj)
        n_pairs = len(platform_df)
        n_projects = platform_df["project_id"].n_unique()
        label_group_rate = (platform_df["label"] == "GROUP").mean()
        threshold_found = None
        target_precision: float = (
            precision_by_platform[platform] if precision_by_platform else min_precision  # type: ignore[assignment]
        )

        # Walk thresholds from low to high; first one meeting precision is the minimum
        # Precision is averaged over projects to avoid large projects dominating
        for thresh in thresholds_sorted:
            df_t = _apply_threshold(platform_df, model_name, thresh)
            project_precisions: list[float] = []
            for _, proj_df in df_t.group_by("project_id"):
                pred_group = proj_df.filter(pl.col(pred_col) == "GROUP")
                if len(pred_group) > 0:
                    project_precisions.append(float((pred_group["label"] == "GROUP").mean()))  # type: ignore[arg-type]
            if not project_precisions:
                continue
            precision = sum(project_precisions) / len(project_precisions)
            if precision >= target_precision:
                avg_metrics = _compute_metrics_avg_over_projects(df_t, model_name)
                threshold_found = thresh
                rows.append(
                    {
                        "platform": platform,
                        "n_pairs": n_pairs,
                        "n_projects": n_projects,
                        "label_GROUP_rate": label_group_rate,
                        "min_threshold": thresh,
                        **avg_metrics,
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
    emit(f"\n### Min threshold for >= {precision_label} avg project precision_GROUP by platform ({model_name})\n")
    emit_table(result)

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
    if token_col not in df.columns:
        df = _add_token_columns(df)

    p10 = df[token_col].quantile(0.10)
    p90 = df[token_col].quantile(0.90)
    short_df = df.filter(pl.col(token_col) <= p10)
    long_df = df.filter(pl.col(token_col) >= p90)

    emit(f"\n### Short stacktraces ({token_col} <= p10 = {p10:.0f} tokens, {len(short_df)} pairs)\n")
    emit(f"label GROUP rate: {float((short_df['label'] == 'GROUP').mean()):.2%}")  # type: ignore[arg-type]
    emit(_compute_metrics(short_df, model_names))

    emit(f"\n### Long stacktraces ({token_col} >= p90 = {p90:.0f} tokens, {len(long_df)} pairs)\n")
    emit(f"label GROUP rate: {float((long_df['label'] == 'GROUP').mean()):.2%}")  # type: ignore[arg-type]
    emit(_compute_metrics(long_df, model_names))
