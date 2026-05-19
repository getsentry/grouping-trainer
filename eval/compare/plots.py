"""Matplotlib figures comparing two models head-to-head."""

from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from .metrics import _compute_metrics_avg_over_projects

sns.set_theme(style="darkgrid")

# Consistent colors: model1 = blue, model2 = orange
MODEL_COLORS = ["#1f77b4", "#ff7f0e"]  # matplotlib default blue and orange


def plot_metrics_by_platform(df: pl.DataFrame, model_names: list[str]) -> plt.Figure:
    """
    Create bar plots comparing 2 models grouped by platform.

    Args:
        df: DataFrame with prediction columns (pred_{model_name}) already added.
        model_names: List of model names (expects exactly 2).

    Returns:
        Figure with one subplot per metric.
    """
    metrics_to_plot: list[str] | None = None
    metrics_rows: list[dict[str, Any]] = []
    for (platform_obj,), platform_df in df.group_by("platform"):
        platform = str(platform_obj)
        for model_name in model_names:
            avg_metrics = _compute_metrics_avg_over_projects(platform_df, model_name)
            if metrics_to_plot is None:
                metrics_to_plot = list(avg_metrics.keys())
            row: dict[str, Any] = {**avg_metrics, "platform": platform, "model": model_name}
            metrics_rows.append(row)

    assert metrics_to_plot is not None, "No platforms in df"
    metrics_df = pl.DataFrame(metrics_rows)
    metrics_pd = metrics_df.to_pandas()
    fig, axes_arr = plt.subplots(1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 5))
    axes: list[plt.Axes] = list(axes_arr)

    for ax, metric in zip(axes, metrics_to_plot, strict=True):
        pivot_df = metrics_pd.pivot(index="platform", columns="model", values=metric)
        pivot_df = pivot_df[model_names]  # ensure consistent column order
        pivot_df.plot(kind="bar", ax=ax, rot=45, legend=False, color=MODEL_COLORS)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylim(0, 1)

    # Single legend for the whole figure (top center)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(model_names), bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # make room for legend on top
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
    n_platforms = len(platforms)

    fig, axes_arr = plt.subplots(n_platforms, 1, figsize=(10, 2 * n_platforms), sharex=True)
    axes = [axes_arr] if n_platforms == 1 else list(axes_arr)

    for ax, platform in zip(axes, platforms, strict=True):
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

    if metrics is None:
        metrics = [col.replace(f"{model1}_", "") for col in project_metrics_df.columns if col.startswith(f"{model1}_")]

    n_metrics = len(metrics)
    fig, axes_arr = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, max(8, len(project_metrics_df) * 0.15)))
    axes: list[plt.Axes] = [axes_arr] if n_metrics == 1 else list(axes_arr)

    # Sort once by pred_GROUP_rate delta, use same order for all subplots
    group_rate_col1 = f"{model1}_pred_GROUP_rate"
    group_rate_col2 = f"{model2}_pred_GROUP_rate"
    sorted_df = project_metrics_df.with_columns(
        (pl.col(group_rate_col2) - pl.col(group_rate_col1)).alias("_delta")
    ).sort("_delta")
    y_labels = [f"{row['org_id']}|{row['project_id']}" for row in sorted_df.iter_rows(named=True)]

    for ax, metric in zip(axes, metrics, strict=True):
        col1 = f"{model1}_{metric}"
        col2 = f"{model2}_{metric}"

        x1 = sorted_df[col1].to_numpy()
        x2 = sorted_df[col2].to_numpy()
        y = range(len(sorted_df))

        for project_idx, (val1, val2) in enumerate(zip(x1, x2, strict=True)):
            color = "green" if val2 >= val1 else "red"
            ax.hlines(y=project_idx, xmin=min(val1, val2), xmax=max(val1, val2), color=color, alpha=0.6)

        ax.scatter(x1, y, color=MODEL_COLORS[0], label=model1, zorder=3, s=20)
        ax.scatter(x2, y, color=MODEL_COLORS[1], label=model2, zorder=3, s=20)

        ax.set_yticks(list(y))
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_xlabel(metric)
        ax.set_title(metric)
        ax.set_xlim(0, 1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(model_names), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Metrics by Project (org_id|project_id)", fontsize=14, y=1.05)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    return fig
