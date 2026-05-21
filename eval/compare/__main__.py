"""
Head-to-head comparison b/t 2 models on held out data. Writes a markdown report and optionally uploads the most impacted
projects to Google Sheets.

Assumes you've run save_embeddings.py for both models.

Example usage:

python -m eval.compare \
    --name_model1 v1 \
    --name_model2 large-no-prefix \
    --gcs_model1 gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v1/similarities/test_full3 \
    --gcs_model2 gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix/similarities/test_full3 \
    --threshold_model1 0.99 \
    --threshold_model2 0.90 \
    --dim_model2 64

# Platform-specific thresholds (comma-separated platform=value, must include "default"):
python -m eval.compare \
    --name_model1 v1 \
    --name_model2 large-no-prefix \
    --gcs_model1 gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v1/similarities/test_full3 \
    --gcs_model2 gs://$GROUPING_TRAINER_BUCKET/runs/2026-04-10-12-39-45-large-no-prefix/similarities/test_full3 \
    --threshold_model1 0.99 \
    --threshold_model2 default=0.92,cocoa=0.94,native=0.94,ruby=0.94 \
    --dim_model2 64

Similarity accuracy is computed over already-labeled pairs. An alternative is to have an LLM cluster each project's
stacktraces. I don't think there's much of a benefit to measuring clustering accuracy, as the labeled pairs were sampled
to be around v1's decision boundary. So the comparison over new pairs introduced in a clustering dataset prolly isn't
interesting. The prod service also isn't doing batch clustering: it makes a new grouping record if a new stacktrace
isn't similar to existing records, o.w. it returns the match w/ no side effect. So there aren't as many transitive
dependencies b/t matches in prod as there would be in a batch or online k-means clustering service.

B/c of intentional sampling biases, `pred_GROUP_rate` is underestimated. Differences b/t models may be amplified when
conditioned on project or platform. (Prolly not in the aggregate b/c javascript is undersampled.) There are many easy
positives missing from the test set. The only reliable way to estimate merge rate offline is to randomly sample a
continguous stream of stacktraces for each project and simulate the grouping service using the load test:
https://github.com/getsentry/seer/tree/main/benchmark
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import polars as pl
from tap import Tap, tapify

from . import report
from .data import _load_and_join, _sync_gcs
from .metrics import (
    compare_metrics_by_stacktrace_length,
    compare_models,
    find_threshold_by_platform,
    metrics_by_platform,
    sweep_thresholds,
    sweep_thresholds_by_project,
)
from .plots import plot_dumbbell_by_project, plot_metrics_by_platform, plot_similarity_distribution
from .report import emit, emit_plot, save_report
from .sheets import _authenticate_for_sheets_upload, _upload_projects_to_sheets, print_projects


def _parse_threshold_list(value: str | None) -> list[float] | None:
    """Parse a comma-separated list of floats, e.g. "10,15,20,25" -> [10.0, 15.0, 20.0, 25.0].

    None / empty string returns None (caller falls back to the function's default).
    """
    if not value:
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_threshold(value: str) -> float | dict[str, float]:
    """Parse a threshold CLI argument.

    Accepts either a plain float (e.g. "0.99") or comma-separated platform=value
    pairs (e.g. "default=0.92,cocoa=0.80,csharp=0.75"). The latter must include
    a "default" key.
    """
    if "=" not in value:
        return float(value)
    thresholds = {}
    for part in value.split(","):
        platform, thresh = part.split("=", 1)
        thresholds[platform.strip()] = float(thresh.strip())
    if "default" not in thresholds:
        raise ValueError(f"Platform-specific thresholds must include a 'default' key, got: {value}")
    return thresholds


def _run_once(
    *,
    gcs_model1: str,
    gcs_model2: str,
    name_model1: str,
    name_model2: str,
    dim_model1: int,
    dim_model2: int,
    threshold_model1: str,
    threshold_model2: str,
    sweep_thresholds_model1: str | None,
    sweep_thresholds_model2: str | None,
    min_group_rate_increase: float,
    min_group_rate_decrease: float,
    max_display_projects: int,
    upload_sheets: bool,
    overwrite: bool,
    dir_output_base: Path,
) -> str:
    """Run a single comparison pass against the live filesystem. Returns the relative comparison_dir."""
    if upload_sheets:
        _authenticate_for_sheets_upload()

    path1 = _sync_gcs(gcs_model1)
    path2 = _sync_gcs(gcs_model2)

    df, label_dim1, label_dim2 = _load_and_join(path1, path2, dim_model1, dim_model2, name_model1, name_model2)
    print(f"Loaded {len(df)} pairs: {name_model1} (dim={label_dim1}) vs {name_model2} (dim={label_dim2})")

    name_dataset = path1.parent.name
    comparison_name = f"{name_model1}_dim{label_dim1}_vs_{name_model2}_dim{label_dim2}"
    dir_output = dir_output_base / name_dataset / comparison_name
    if dir_output.exists() and not overwrite:
        print()
        raise SystemExit(
            f"Output directory already exists: {dir_output}\n"
            "You're using the same dim_model1, dim_model2, name_model1, name_model2 values as a previous run.\n"
            "Pass --overwrite to replace it, or use a different name."
        )
    dir_output.mkdir(parents=True, exist_ok=True)

    thresholds = {
        name_model1: _parse_threshold(threshold_model1),
        name_model2: _parse_threshold(threshold_model2),
    }

    emit(f"# {name_model1} (dim={label_dim1}) vs {name_model2} (dim={label_dim2}), dataset: {name_dataset}\n")
    # `Tap.get_reproducibility_info` captures sys.argv verbatim — for `python -m eval.compare` that's the
    # absolute path to __main__.py. Rewrite to the module-invocation form so the README stays portable.
    raw_cmd = Tap.get_reproducibility_info()["command_line"]
    cmd = re.sub(r"^python\s+\S+", "python -m eval.compare", raw_cmd).replace(" --", " \\\n    --")
    expected_gcs_bucket = f"gs://{os.environ['GROUPING_TRAINER_BUCKET']}/"
    assert expected_gcs_bucket in cmd
    cmd = cmd.replace(expected_gcs_bucket, "gs://$GROUPING_TRAINER_BUCKET/")
    emit("Command to repro:\n\n```bash\n" + cmd + "\n```\n")

    emit(
        "### Column definitions\n\n"
        "- **model**: The name of the model being evaluated.\n"
        "- **pred_GROUP_rate**: The fraction of pairs this model groups together"
        "—lower means more separate issues are created."
        " It's smaller than prod b/c the test dataset contains far more borderline cases;"
        " it's missing pairs that are very close. "
        " This bias also means precision_GROUP is lower than what it'd be in prod.\n"
        "- **precision_GROUP**: When the model groups a pair, how often is it correct?"
        " Higher = less over-grouping.\n"
        "- **precision_SEPARATE**: When the model separates a pair, how often is it correct?\n"
        "- **recall_GROUP**: Of all pairs that should be grouped,"
        " what fraction does the model correctly group? Higher = less under-grouping.\n"
        "- **recall_SEPARATE**: Of all pairs that should be separate,"
        " what fraction does the model correctly separate?\n"
    )

    emit("## Aggregate results\n")

    result = compare_models(
        df=df,
        thresholds=thresholds,
        output_dir=dir_output,
        min_group_rate_increase=min_group_rate_increase,
        min_group_rate_decrease=min_group_rate_decrease,
    )

    compare_metrics_by_stacktrace_length(result.df, result.model_names)

    emit("\n## Threshold sweep\n")

    sweep_list_model1 = _parse_threshold_list(sweep_thresholds_model1)
    sweep_list_model2 = _parse_threshold_list(sweep_thresholds_model2)
    sweep_lists_by_name = {name_model1: sweep_list_model1, name_model2: sweep_list_model2}

    sweep_thresholds(df, name_model1, thresholds=sweep_list_model1)
    sweep_thresholds(df, name_model2, thresholds=sweep_list_model2)
    threshold1_parsed = thresholds[name_model1]
    threshold2_parsed = thresholds[name_model2]
    sweep_thresholds_by_project(
        df,
        name_model2,
        thresholds=sweep_list_model2,
        thresholds_platform=threshold2_parsed if isinstance(threshold2_parsed, dict) else None,
        baseline_model=name_model1,
        baseline_threshold=threshold1_parsed,
    )

    emit("\n## Platform-level results\n")

    for name in [name_model1, name_model2]:
        metrics_by_platform(df, name, thresholds[name])

    for name in [name_model1, name_model2]:
        find_threshold_by_platform(df, name, thresholds=sweep_lists_by_name[name])

    fig = plot_metrics_by_platform(result.df, result.model_names)
    fig.savefig(dir_output / "metrics_by_platform.png", dpi=150, bbox_inches="tight")
    print(f"Saved {dir_output / 'metrics_by_platform.png'}")
    emit_plot("Metrics by platform", "metrics_by_platform.png")

    for name in result.model_names:
        fig = plot_similarity_distribution(result.df, name)
        filename = f"similarity_distribution_{name}.png"
        fig.savefig(dir_output / filename, dpi=150, bbox_inches="tight")
        print(f"Saved {dir_output / filename}")
        emit_plot(f"Similarity distribution ({name})", filename)

    emit("\n## Project-level results\n")

    model1_display, model2_display = result.model_names
    project_metrics = result.project_metrics
    wins = project_metrics.filter(
        (pl.col(f"{model2_display}_precision_GROUP") > pl.col(f"{model1_display}_precision_GROUP"))
        & (pl.col(f"{model2_display}_recall_GROUP") > pl.col(f"{model1_display}_recall_GROUP"))
    )
    emit(
        f"**Project win rate for {model2_display}**: {len(wins)}/{len(project_metrics)}"
        f" ({len(wins) / len(project_metrics):.0%}) projects where both precision_GROUP and recall_GROUP are higher\n"
    )

    fig = plot_dumbbell_by_project(result.project_metrics, result.model_names)
    fig.savefig(dir_output / "dumbbell_by_project.png", dpi=150, bbox_inches="tight")
    print(f"Saved {dir_output / 'dumbbell_by_project.png'}")
    emit_plot("Dumbbell by project", "dumbbell_by_project.png")

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
        emit(
            f"\n- Sheets suffixed w/ `|new` contain pairs that {name_model1} groups together,"
            f" but {name_model2} separates (creating new issues).\n"
            f"- Sheets suffixed w/ `|merged` contain pairs that {name_model1} separates,"
            f" but {name_model2} would group together (merging issues).\n"
            "- If you click the button in the top right corner of the table,"
            " you'll see what the LLM thought about the pair and the model's similarity scores"
        )

    save_report(dir_output / "README.md")
    print(f"\nResults written to {dir_output}")

    return f"{name_dataset}/{comparison_name}"


def main(
    gcs_model1: str,
    gcs_model2: str,
    name_model1: str,
    name_model2: str,
    dim_model1: int = 768,
    dim_model2: int = 768,
    threshold_model1: str = "0.99",
    threshold_model2: str = "0.90",
    sweep_thresholds_model1: str | None = None,
    sweep_thresholds_model2: str | None = None,
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
        GCS path to model 1's similarities directory,
        e.g., gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v1/similarities/test_full3
    gcs_model2
        GCS path to model 2's similarities directory,
        e.g., gs://$GROUPING_TRAINER_BUCKET/runs/issue_grouping_v2/similarities/test_full3
    dim_model1
        Which cos_sim_{dim} column to use from model 1's CSV.
    dim_model2
        Which cos_sim_{dim} column to use from model 2's CSV.
    name_model1
        Short alias for model 1 used in output columns and file names.
    name_model2
        Short alias for model 2 used in output columns and file names.
    threshold_model1
        Cosine similarity threshold for model 1. Either a plain float (e.g. "0.99")
        or comma-separated platform=value pairs (e.g. "default=0.92,cocoa=0.80,node=0.90").
    threshold_model2
        Cosine similarity threshold for model 2. Same format as threshold_model1.
    sweep_thresholds_model1
        Comma-separated thresholds to sweep for model 1's per-threshold metrics and per-platform threshold finder,
        e.g. "0.95,0.97,0.99". Override the cosine-range defaults when model 1's score isn't in [0, 1].
    sweep_thresholds_model2
        Comma-separated thresholds to sweep for model 2. Same format as sweep_thresholds_model1.
    min_group_rate_increase
        Flag projects where model2 GROUP rate exceeds model1 by at least this amount.
    min_group_rate_decrease
        Flag projects where model2 GROUP rate is lower than model1 by at least this amount.
    max_display_projects
        Maximum number of flagged projects to display.
    upload_sheets
        If True, upload flagged projects to Google Sheets. Fetches the OAuth client JSON from GCP Secret Manager
        (secret name given by the `OAUTH_CLIENT_SECRET_NAME` env var) and prompts you to authenticate in a browser.
    overwrite
        Allow overwriting an existing output directory. Without this flag the
        script exits with an error if the output directory already exists.
    """
    common_kwargs = dict(
        gcs_model1=gcs_model1,
        gcs_model2=gcs_model2,
        name_model1=name_model1,
        name_model2=name_model2,
        dim_model1=dim_model1,
        dim_model2=dim_model2,
        threshold_model1=threshold_model1,
        threshold_model2=threshold_model2,
        sweep_thresholds_model1=sweep_thresholds_model1,
        sweep_thresholds_model2=sweep_thresholds_model2,
        min_group_rate_increase=min_group_rate_increase,
        min_group_rate_decrease=min_group_rate_decrease,
        max_display_projects=max_display_projects,
        overwrite=overwrite,
    )

    report.IS_ANONYMIZED = True
    report._report_lines.clear()
    print("Generating anonymized report (committed to repo)")
    real_id_to_synthetic_str: dict = {}
    with patch.object(pl, "read_csv", report._make_anonymizing_read_csv(real_id_to_synthetic_str)):
        comparison_dir = _run_once(
            **common_kwargs,
            upload_sheets=False,
            dir_output_base=Path("eval/comparisons"),
        )

    gcs_path_templated = f"gs://$GROUPING_TRAINER_BUCKET/eval/comparisons/{comparison_dir}/"
    path_readme_committed = Path("eval/comparisons") / comparison_dir / "README.md"
    # Append a templated pointer so the bucket name stays out of the committed README; readers with
    # $GROUPING_TRAINER_BUCKET set can shell-substitute it to get the real path.
    with path_readme_committed.open("a") as f:
        f.write(f"\n\n---\n\n_Real report with original org/project IDs at_ `{gcs_path_templated}`\n")

    report.IS_ANONYMIZED = False
    report._report_lines.clear()
    print("Generating real report (uploaded to GCS)")
    gcs_path = f"gs://{os.environ['GROUPING_TRAINER_BUCKET']}/eval/comparisons/{comparison_dir}/"
    with tempfile.TemporaryDirectory() as temp_dir:
        _run_once(
            **common_kwargs,
            upload_sheets=upload_sheets,
            dir_output_base=Path(temp_dir),
        )
        print(f"Uploading real report to {gcs_path}")
        # Source is the subdir matching comparison_dir so contents land directly under gcs_path
        # (rsyncing temp_dir itself would duplicate the {dataset}/{model_pair} segment in GCS).
        # --no-user-output-enabled mutes the per-file "Copying ..." lines; errors still surface.
        subprocess.run(
            [
                "gcloud",
                "--no-user-output-enabled",
                "storage",
                "rsync",
                "-r",
                f"{temp_dir}/{comparison_dir}/",
                gcs_path,
            ],
            check=True,
        )


if __name__ == "__main__":
    tapify(main, description=__doc__)
