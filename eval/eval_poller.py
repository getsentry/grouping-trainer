"""
Polls GCS for new training checkpoints and runs evaluation on each one. Run this script on an L4. Idempotent.
"""

import logging
import re
import subprocess
import tempfile
import time
from typing import Literal

import wandb
from pydantic import BaseModel
from tap import tapify

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
    sample_val: int | None,
    truncate_dims: tuple[int, ...],
    use_simple_precisions: bool = False,
    use_confidence_score: bool = False,
    confidence_score_floor: float = 0.9,
) -> gt.evaluator.MinPrecisionEvaluator:
    dataset_val = gt.train.df_to_dataset(
        gt.data.load_val_df(paths=("final_csvs/val.csv",), sample_size=sample_val),
        use_confidence_score=use_confidence_score,
        confidence_score_floor=confidence_score_floor,
    )
    return gt.evaluator.MinPrecisionEvaluator(
        sentences1=list(dataset_val["query_stacktrace_string"]),
        sentences2=list(dataset_val["candidate_stacktrace_string"]),
        labels=[int(record["label"]) for record in dataset_val],
        confidence_scores=None if not use_confidence_score else [record["confidence_score"] for record in dataset_val],
        name="val",
        show_progress_bar=True,
        batch_size=2,
        truncate_dims=truncate_dims,
        target_precisions=[0.7, 0.8] if use_simple_precisions else None,
    )


def _format_metrics(metrics: dict[str, float]) -> str:
    """
    Format eval metrics as plain-text lines for GCP-friendly logging. It doesn't like showing a polars table.
    """
    lines = [f"  {key}: {value:.6f}" for key, value in sorted(metrics.items())]
    return "\n".join(lines)


def log_eval_metrics(step: int, metrics: dict[str, float]):
    wandb.log({"train/global_step": step, **{f"eval/{key}": value for key, value in metrics.items()}})
    logger.info(f"Step {step} eval:\n{_format_metrics(metrics)}")


def evaluate_checkpoint(
    step: int,
    checkpoint_gcs_path: str,
    encoder: gt.utils.SentenceTransformer,
    evaluator: gt.evaluator.MinPrecisionEvaluator,
    loss_type: Literal["sigmoid", "contrastive"] = "contrastive",
    contrastive_margin: float = 0.5,
):
    """
    Download the checkpoint to a temp dir, evaluate it, log to wandb, and write the eval done sentinel to GCS.
    """
    logger.info(f"Evaluating checkpoint at step {step}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        download_checkpoint(checkpoint_gcs_path, tmp_dir)
        model = gt.train.ModelForTraining.from_checkpoint(
            checkpoint_dir=tmp_dir, encoder=encoder, loss_type=loss_type, contrastive_margin=contrastive_margin
        )
        metrics = evaluator(model)

    log_eval_metrics(step, metrics)
    subprocess.run(
        ["gcloud", "storage", "cp", "-", f"{checkpoint_gcs_path}/{gt.sentinels.EVAL_DONE}"],
        input=b"",
        check=True,
    )


def evaluate_baseline(
    run_gcs_dir: str,
    encoder: gt.utils.SentenceTransformer,
    evaluator: gt.evaluator.MinPrecisionEvaluator,
    loss_type: Literal["sigmoid", "contrastive"] = "contrastive",
    contrastive_margin: float = 0.5,
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
    if loss_type == "sigmoid":
        loss = gt.loss.SigmoidPairwiseLoss()
    elif loss_type == "contrastive":
        loss = gt.loss.ContrastiveLoss(margin=contrastive_margin)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    model_baseline = gt.train.ModelForTraining(encoder=encoder, loss=loss)
    metrics_baseline = evaluator(model_baseline)
    log_eval_metrics(step=0, metrics=metrics_baseline)
    subprocess.run(["gcloud", "storage", "cp", "-", sentinel_path], input=b"", check=True)


def backfill(
    run_gcs_dir: str,
    encoder: gt.utils.SentenceTransformer,
    evaluator: gt.evaluator.MinPrecisionEvaluator,
    loss_type: Literal["sigmoid", "contrastive"] = "contrastive",
    contrastive_margin: float = 0.5,
):
    """
    Evaluate all unevaluated checkpoints.
    """
    checkpoints = list_done_checkpoints(run_gcs_dir)
    unevaluated = [checkpoint_info for checkpoint_info in checkpoints if not checkpoint_info.is_eval_done]
    logger.info(f"Backfill: {len(unevaluated)} out of {len(checkpoints)} checkpoints are unevaluated")
    for checkpoint_info in unevaluated:
        evaluate_checkpoint(
            checkpoint_info.step,
            checkpoint_info.gcs_path,
            encoder,
            evaluator,
            loss_type=loss_type,
            contrastive_margin=contrastive_margin,
        )


def poll(
    run_gcs_dir: str,
    poll_interval_sec: int,
    encoder: gt.utils.SentenceTransformer,
    evaluator: gt.evaluator.MinPrecisionEvaluator,
    loss_type: Literal["sigmoid", "contrastive"] = "contrastive",
    contrastive_margin: float = 0.5,
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
            evaluate_checkpoint(
                checkpoint_info.step,
                checkpoint_info.gcs_path,
                encoder,
                evaluator,
                loss_type=loss_type,
                contrastive_margin=contrastive_margin,
            )
            evaluated_steps.add(checkpoint_info.step)

        if gcloud_storage_ls(f"{run_gcs_dir}/{gt.sentinels.TRAINING_DONE}") is not None:
            logger.info("Training done. Running final backfill pass.")
            backfill(run_gcs_dir, encoder, evaluator, loss_type=loss_type, contrastive_margin=contrastive_margin)
            break

        if not new_checkpoints:
            logger.info(f"No new checkpoints. Sleeping {poll_interval_sec}s...")
            time.sleep(poll_interval_sec)


def main(
    run_gcs_dir: str,
    base_model: str,
    wandb_run_id: str,
    wandb_project: str = "grouping-trainer",
    poll_interval_sec: int = 60 * 2,
    sample_val: int | None = None,  # may be fast enough to encode everything b/t saves. big enough to stay busy.
    truncate_dims: tuple[int, ...] = (64, 768),
    use_simple_precisions: bool = False,
    loss_type: Literal["sigmoid", "contrastive"] = "contrastive",
    contrastive_margin: float = 0.5,
    use_text_prefix: bool = False,
    use_confidence_score: bool = False,
    confidence_score_floor: float = 0.9,
):
    """
    Poll GCS for new training checkpoints and evaluate each one.

    Parameters
    ----------
    run_gcs_dir
        GCS path to the training run directory (e.g. gs://grouping-data/runs/...).
    base_model
        HuggingFace model ID for the base encoder. Used to load architecture before applying checkpoint weights.
        Example: "Qwen/Qwen3-Embedding-0.6B"
    wandb_run_id
        W&B run ID of the training run. Eval metrics are logged to the same run using shared mode.
    wandb_project
        W&B project name.
    poll_interval_sec
        Seconds to sleep between polling cycles when no new checkpoints are found.
    sample_val
        Number of validation examples to sample. None uses the full val set.
    truncate_dims
        Matryoshka dimensions to evaluate at.
    use_simple_precisions
        Use a simpler set of target precisions—[0.7, 0.8]—for faster eval.
    loss_type
        Which loss function was used for training. Must match so checkpoint weights load correctly.
    contrastive_margin
        Margin for contrastive loss. Only used when loss_type is "contrastive".
    use_text_prefix
        If True, add the model's designated prefix to the input text.
    use_confidence_score
        If True, use the confidence score as part of the evaluation loss computation.
    confidence_score_floor
        Minimum confidence score. Values below this are clipped up.
    """
    run_name = run_gcs_dir.rstrip("/").rsplit("/", 1)[-1]
    gt.logging.configure_logging(
        run_name=run_name,
        process_type="eval_poller",
    )

    wandb.login()
    wandb.init(
        project=wandb_project,
        id=wandb_run_id,
        settings=wandb.Settings(mode="shared", x_primary=False, x_label="eval", x_update_finish_state=False),
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

    try:
        evaluator = make_evaluator(
            sample_val,
            truncate_dims,
            use_simple_precisions=use_simple_precisions,
            use_confidence_score=use_confidence_score,
            confidence_score_floor=confidence_score_floor,
        )
        encoder = gt.utils.encoder_from_base(base_model, use_text_prefix=use_text_prefix)
        evaluate_baseline(run_gcs_dir, encoder, evaluator, loss_type=loss_type, contrastive_margin=contrastive_margin)
        poll(
            run_gcs_dir,
            poll_interval_sec,
            encoder,
            evaluator,
            loss_type=loss_type,
            contrastive_margin=contrastive_margin,
        )
    finally:
        wandb.finish()


if __name__ == "__main__":
    tapify(main, description=__doc__)
