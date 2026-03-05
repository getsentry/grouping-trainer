"""
Polls GCS for new training checkpoints and runs evaluation on each one.
Run on a separate machine (e.g., L4) so that the training GPU stays fully utilized.

Usage:
    python eval/eval_poller.py --gcs-dir gs://grouping-data/runs/... --wandb-run-id abc123
"""

import argparse
import logging
import re
import subprocess
import tempfile
import time

import wandb

import grouping_trainer as gt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def list_ready_checkpoints(gcs_dir: str) -> list[tuple[int, str]]:
    """Return (step, gcs_path) for checkpoints that have a .done sentinel, sorted by step."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", f"{gcs_dir}/"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.warning(f"Failed to list {gcs_dir}/")
        return []

    checkpoints = []
    for line in result.stdout.strip().splitlines():
        line = line.rstrip("/")
        match = re.search(r"checkpoint-(\d+)$", line)
        if not match:
            continue
        step = int(match.group(1))
        gcs_path = line

        # Check for .done sentinel
        sentinel = f"{gcs_path}/.done"
        check = subprocess.run(["gsutil", "ls", sentinel], capture_output=True)
        if check.returncode == 0:
            checkpoints.append((step, gcs_path))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def training_done(gcs_dir: str) -> bool:
    result = subprocess.run(["gsutil", "ls", f"{gcs_dir}/.training_done"], capture_output=True)
    return result.returncode == 0


def download_checkpoint(gcs_path: str, local_dir: str):
    subprocess.run(["gsutil", "-m", "rsync", "-r", gcs_path, local_dir], check=True)


def main():
    parser = argparse.ArgumentParser(description="Poll GCS for checkpoints and run evaluation")
    parser.add_argument("--gcs-dir", required=True, help="GCS directory containing checkpoints")
    parser.add_argument("--wandb-run-id", required=True, help="W&B run ID to log metrics to")
    parser.add_argument("--wandb-project", default="grouping-trainer", help="W&B project name")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between polls")
    parser.add_argument("--sample-val", type=int, default=8000, help="Number of validation samples")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Eval batch size")
    parser.add_argument("--truncate-dims", default="64,768", help="Comma-separated truncation dims")
    args = parser.parse_args()

    truncate_dims = tuple(int(d) for d in args.truncate_dims.split(","))

    wandb.login()
    wandb.init(id=args.wandb_run_id, project=args.wandb_project, resume="allow")
    wandb.define_metric("eval_step")
    wandb.define_metric("eval_val_*", step_metric="eval_step")

    # Load validation data once
    dataset_val = gt.train.df_to_dataset(gt.data.load_val_df(sample_size=args.sample_val))
    evaluator = gt.evaluator.MinPrecisionEvaluator(
        sentences1=list(dataset_val["query_stacktrace_string"]),
        sentences2=list(dataset_val["candidate_stacktrace_string"]),
        labels=[int(record["label"]) for record in dataset_val],
        name="val",
        show_progress_bar=True,
        batch_size=args.eval_batch_size,
        truncate_dims=truncate_dims,
    )

    evaluated_steps: set[int] = set()

    while True:
        checkpoints = list_ready_checkpoints(args.gcs_dir)
        new_checkpoints = [(step, path) for step, path in checkpoints if step not in evaluated_steps]

        for step, gcs_path in new_checkpoints:
            logger.info(f"Evaluating checkpoint at step {step}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                download_checkpoint(gcs_path, tmp_dir)
                # rsync syncs contents directly into tmp_dir
                model = gt.danger.SentenceTransformer(tmp_dir)
                model.warmup_and_compile()

                metrics = evaluator(model)

            # Prefix metrics for async eval namespace
            log_dict = {"eval_step": step}
            for key, value in metrics.items():
                log_dict[f"eval_{key}"] = value

            wandb.log(log_dict)
            evaluated_steps.add(step)
            logger.info(f"Step {step} eval complete: {metrics}")

        if training_done(args.gcs_dir) and not new_checkpoints:
            logger.info("Training done and all checkpoints evaluated. Exiting.")
            break

        if not new_checkpoints:
            logger.info(f"No new checkpoints. Sleeping {args.poll_interval}s...")
            time.sleep(args.poll_interval)

    wandb.finish()


if __name__ == "__main__":
    main()
