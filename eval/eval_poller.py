"""
Polls GCS for new training checkpoints and runs evaluation on each one. Run this script on an L4. Idempotent.
"""

import logging
import re
import subprocess
import tempfile
import time
from typing import Literal, overload

import polars as pl
from pydantic import BaseModel
from tap import tapify
import torch
import wandb

import grouping_trainer as gt

logger = logging.getLogger(__name__)


class CheckpointInfo(BaseModel):
    step: int
    gcs_path: str
    is_eval_done: bool


def gcloud_storage_ls(gcs_path: str) -> str | None:
    """
    Returns stdout if path exists, None if not found.
    """
    try:
        result = subprocess.run(["gcloud", "storage", "ls", gcs_path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        if "matched no objects" in e.stderr:
            return None
        raise


def list_done_checkpoints(run_gcs_dir: str) -> list[CheckpointInfo]:
    """
    Return checkpoints that have a .checkpoint_done sentinel, sorted by step.
    """
    output = gcloud_storage_ls(f"{run_gcs_dir}/")
    if output is None:
        return []
    checkpoint_dirs = output.strip().splitlines()

    checkpoints: list[CheckpointInfo] = []
    for line in checkpoint_dirs:
        line = line.rstrip("/")
        match = re.search(r"checkpoint-(\d+)$", line)
        if not match:
            continue
        checkpoint_gcs_path = line

        if gcloud_storage_ls(f"{checkpoint_gcs_path}/{gt.sentinels.CHECKPOINT_DONE}") is None:
            continue

        checkpoints.append(
            CheckpointInfo(
                step=int(match.group(1)),
                gcs_path=checkpoint_gcs_path,
                is_eval_done=gcloud_storage_ls(f"{checkpoint_gcs_path}/{gt.sentinels.EVAL_DONE}") is not None,
            )
        )

    checkpoints.sort(key=lambda checkpoint_info: checkpoint_info.step)
    return checkpoints


def download_checkpoint(checkpoint_gcs_path: str, local_dir: str):
    """
    Download the checkpoint from GCS to a local directory.
    """
    subprocess.run(["gcloud", "storage", "rsync", "-r", checkpoint_gcs_path, local_dir], check=True)


def make_evaluator(
    sample_val: int | None, truncate_dims: tuple[int, ...], use_simple_precisions: bool = False
) -> gt.evaluator.MinPrecisionEvaluator:
    dataset_val = gt.train.df_to_dataset(gt.data.load_val_df(sample_size=sample_val))
    return gt.evaluator.MinPrecisionEvaluator(
        sentences1=list(dataset_val["query_stacktrace_string"]),
        sentences2=list(dataset_val["candidate_stacktrace_string"]),
        labels=[int(record["label"]) for record in dataset_val],
        name="val",
        show_progress_bar=True,
        batch_size=1,  # pls use the CUDA graph-cached model
        truncate_dims=truncate_dims,
        target_precisions=[0.7, 0.8] if use_simple_precisions else None,
    )


@overload
def make_encoder(base_model: str, use_auto_detected_device: Literal[True]) -> gt.utils.SentenceTransformer: ...


@overload
def make_encoder(base_model: str, use_auto_detected_device: Literal[False] = ...) -> gt.danger.SentenceTransformer: ...


def make_encoder(
    base_model: str, use_auto_detected_device: bool = False
) -> gt.utils.SentenceTransformer | gt.danger.SentenceTransformer:
    if use_auto_detected_device:
        return gt.utils.SentenceTransformer(base_model)
    encoder = gt.danger.SentenceTransformer(
        base_model,
        model_kwargs=dict(
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ),
    )
    encoder.warmup_and_compile()
    return encoder


def _format_metrics(metrics: dict[str, float]) -> str:
    """
    Format eval metrics as a polars table, parsed from flat key structure.
    """
    rows = []
    for key, value in metrics.items():
        parts = key.split("_", 2)  # e.g. ["val", "dim64", "pr85_threshold"]
        if len(parts) >= 3:
            rows.append({"dim": parts[1], "metric": parts[2], "value": value})
        else:
            rows.append({"dim": "", "metric": key, "value": value})

    df = pl.DataFrame(rows).pivot(on="metric", index="dim", values="value")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        return str(df)


def log_eval_metrics(step: int, metrics: dict[str, float]):
    wandb.log({"train/global_step": step, **{f"eval_{key}": value for key, value in metrics.items()}})
    logger.info(f"Step {step} eval:\n{_format_metrics(metrics)}")


def evaluate_checkpoint(
    step: int,
    checkpoint_gcs_path: str,
    encoder: gt.danger.SentenceTransformer,
    evaluator: gt.evaluator.MinPrecisionEvaluator,
):
    """
    Download the checkpoint to a temp dir, evaluate it, log to wandb, and write the eval done sentinel to GCS.
    """
    logger.info(f"Evaluating checkpoint at step {step}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        download_checkpoint(checkpoint_gcs_path, tmp_dir)
        model = gt.train.ModelForTraining.from_checkpoint(checkpoint_dir=tmp_dir, encoder=encoder)
        metrics = evaluator(model)

    log_eval_metrics(step, metrics)
    subprocess.run(
        ["gcloud", "storage", "cp", "-", f"{checkpoint_gcs_path}/{gt.sentinels.EVAL_DONE}"],
        input=b"",
        check=True,
    )


def evaluate_baseline(
    run_gcs_dir: str,
    encoder: gt.danger.SentenceTransformer,
    evaluator: gt.evaluator.MinPrecisionEvaluator,
):
    """
    Evaluate the base model (before any fine-tuning) and log metrics at step 0.
    Writes a sentinel so the baseline is not re-evaluated on restart.
    """
    sentinel_path = f"{run_gcs_dir}/{gt.sentinels.BASELINE_EVAL_DONE}"
    if gcloud_storage_ls(sentinel_path) is not None:
        logger.info("Baseline already evaluated. Skipping.")
        return

    logger.info("Evaluating base model.")
    model_baseline = gt.train.ModelForTraining(encoder=encoder, loss=gt.loss.SigmoidPairwiseLoss())
    metrics_baseline = evaluator(model_baseline)
    log_eval_metrics(step=0, metrics=metrics_baseline)
    subprocess.run(["gcloud", "storage", "cp", "-", sentinel_path], input=b"", check=True)


def backfill(run_gcs_dir: str, encoder: gt.danger.SentenceTransformer, evaluator: gt.evaluator.MinPrecisionEvaluator):
    """
    Evaluate all unevaluated checkpoints.
    """
    checkpoints = list_done_checkpoints(run_gcs_dir)
    unevaluated = [checkpoint_info for checkpoint_info in checkpoints if not checkpoint_info.is_eval_done]
    logger.info(f"Backfill: {len(unevaluated)} out of {len(checkpoints)} checkpoints are unevaluated")
    for checkpoint_info in unevaluated:
        evaluate_checkpoint(checkpoint_info.step, checkpoint_info.gcs_path, encoder, evaluator)


def poll(
    run_gcs_dir: str,
    poll_interval_sec: int,
    encoder: gt.danger.SentenceTransformer,
    evaluator: gt.evaluator.MinPrecisionEvaluator,
):
    """
    Poll for new checkpoints until training is done, then do a final backfill pass.
    """
    evaluated_steps: set[int] = set()

    while True:
        checkpoints = list_done_checkpoints(run_gcs_dir)
        # Pre-populate from .eval_done sentinels (survives restarts)
        evaluated_steps.update(checkpoint_info.step for checkpoint_info in checkpoints if checkpoint_info.is_eval_done)
        new_checkpoints = [
            checkpoint_info for checkpoint_info in checkpoints if checkpoint_info.step not in evaluated_steps
        ]

        for checkpoint_info in new_checkpoints:
            evaluate_checkpoint(checkpoint_info.step, checkpoint_info.gcs_path, encoder, evaluator)
            evaluated_steps.add(checkpoint_info.step)

        if gcloud_storage_ls(f"{run_gcs_dir}/{gt.sentinels.TRAINING_DONE}") is not None:
            logger.info("Training done. Running final backfill pass.")
            backfill(run_gcs_dir, encoder, evaluator)
            break

        if not new_checkpoints:
            logger.info(f"No new checkpoints. Sleeping {poll_interval_sec}s...")
            time.sleep(poll_interval_sec)


def main(
    run_gcs_dir: str,
    wandb_run_id: str,
    base_model: str = "Alibaba-NLP/gte-modernbert-base",
    wandb_project: str = "grouping-trainer",
    poll_interval_sec: int = 60 * 5,
    sample_val: int | None = 20_000,
    truncate_dims: tuple[int, ...] = (64, 768),
    use_auto_detected_device: bool = False,
    use_simple_precisions: bool = False,
):
    """
    Poll GCS for new training checkpoints and evaluate each one.

    Parameters
    ----------
    run_gcs_dir
        GCS path to the training run directory (e.g. gs://grouping-data/runs/...).
    wandb_run_id
        W&B run ID to resume logging into (from the training job).
    base_model
        HuggingFace model ID for the base encoder. Used to load architecture before applying checkpoint weights.
    wandb_project
        W&B project name.
    poll_interval_sec
        Seconds to sleep between polling cycles when no new checkpoints are found.
    sample_val
        Number of validation examples to sample. None uses the full val set.
    truncate_dims
        Matryoshka dimensions to evaluate at.
    use_auto_detected_device
        Leave the model on its auto-detected device and use a plain SentenceTransformer w/o compilation.
    use_simple_precisions
        Use a simpler set of target precisions—[0.7, 0.8]—for faster eval.
    """
    run_name = run_gcs_dir.rstrip("/").rsplit("/", 1)[-1]
    gt.logging.configure_logging(
        run_name=run_name,
        process_type="eval_poller",
    )

    if not use_auto_detected_device:
        assert torch.cuda.is_available(), "Run this on a GPU or pass --use_auto_detected_device"

    wandb.login()
    wandb.init(id=wandb_run_id, project=wandb_project, resume="allow")
    wandb.define_metric("train/global_step")
    wandb.define_metric("eval_*", step_metric="train/global_step")

    evaluator = make_evaluator(sample_val, truncate_dims, use_simple_precisions=use_simple_precisions)
    encoder = make_encoder(base_model, use_auto_detected_device=use_auto_detected_device)

    evaluate_baseline(run_gcs_dir, encoder, evaluator)
    poll(run_gcs_dir, poll_interval_sec, encoder, evaluator)

    wandb.finish()


if __name__ == "__main__":
    tapify(main, description=__doc__)
