"""
Trains a model, logs to wandb, and saves it to local and GCS.
Evaluation runs async on a separate machine. See eval/eval_poller.py
"""

import logging
import os
import subprocess
import tempfile
import warnings
from datetime import datetime
from typing import Literal

import torch
import wandb
from tap import tapify

import grouping_trainer as gt

_RUN_NAME_ENV_VAR = "GROUPING_TRAINER_RUN_NAME"

logger = logging.getLogger(__name__)


def upload_run_metadata(run_gcs_dir: str, training_config: gt.train.TrainingConfig) -> None:
    """
    Save training training_config and git commit to a local temp dir, then upload to GCS as metadata/.
    """

    with tempfile.TemporaryDirectory() as dir_metadata:
        with open(os.path.join(dir_metadata, "training_config.json"), "w") as f:
            f.write(training_config.model_dump_json(indent=2))
        git_commit = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--long"], capture_output=True, text=True
        ).stdout.strip()
        with open(os.path.join(dir_metadata, "git_commit.txt"), "w") as f:
            f.write(git_commit + "\n")
        subprocess.run(
            ["gcloud", "storage", "rsync", "-r", dir_metadata, f"{run_gcs_dir}/metadata"],
            check=True,
        )
    logger.info(f"Uploaded run metadata (git: {git_commit}) to {run_gcs_dir}/metadata")


base_model_to_per_device_token_budget_scale = {
    # Not including the v1 jinaai/jina-embeddings-v2-base-code model b/c it doesn't support SDPA.
    # Pls don't use models that don't support flash attention.
    "lightonai/modernbert-embed-large": 4,
    "Alibaba-NLP/gte-modernbert-base": 6,
    "Qwen/Qwen3-Embedding-0.6B": 3,
    "jinaai/jina-embeddings-v5-text-nano-text-matching": 4,
}


def run(
    base_model: str = "lightonai/modernbert-embed-large",
    run_shortname: str | None = None,
    use_text_prefix: bool = False,
    per_device_token_budget_scale: int | None = None,
    per_device_train_batch_size: int = 256,
    learning_rate: float = 1e-4,
    tiny_run: bool = False,
    gpu: Literal["h100", "h100-ddp", "a100", "a100-ddp"] | None = None,
    zone: str | None = None,
):
    """
    Train a grouping model. Writes checkpoints to GCS. Logs to wandb: https://wandb.ai/sentry-seer/grouping-trainer

    Parameters
    ----------
    base_model
        HuggingFace model ID or local path for the base encoder.
        Others we've tried: Alibaba-NLP/gte-modernbert-base, Qwen/Qwen3-Embedding-0.6B,
        jinaai/jina-embeddings-v5-text-nano-text-matching
    run_shortname
        Short name for the run. Doesn't need to be unique b/c it's appended to the timestamp.
    use_text_prefix
        If True, add the model's designated prefix to the input text. Didn't help lightonai/modernbert-embed-large
    per_device_token_budget_scale
        Sets per_device_token_budget = per_device_token_budget_scale * 8192. This is the memory and throughput knob.
        By default, if the base_model has a historically known good scale, we use that, o.w. uses 3.
    per_device_train_batch_size
        Training batch size per device. Be intentional about this when doing DDP. Only used for non-tiny runs.
        Technically this can be arbitrarily high b/c we accumulate the gradient based on per_device_token_budget_scale.
    learning_rate
        Should consider scaling this in proportion to per_device_train_batch_size.
    tiny_run
        Run a tiny training run to sanity check plumbing.
    gpu
        Flex-start a GPU instance and train on that machine.
    zone
        Override the default GCP zone when launching the GPU instance. Useful when
        flex-start capacity is dry in the default zone for the requested gpu type.
    """
    if not tiny_run:
        assert run_shortname is not None, "run_shortname is required for full training runs"

    # Generate run_name up front so we can log the artifact URL locally before
    # auto-launching. On the remote, re-use the local run_name via env var so
    # both sides log the same GCS path (rather than each generating its own
    # timestamp).
    run_name_env = os.environ.get(_RUN_NAME_ENV_VAR)
    if run_name_env:
        run_name = run_name_env
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_name = f"{timestamp}-{run_shortname or 'tiny-run'}"
    run_gcs_dir = f"gs://grouping-data/runs/{run_name}"
    run_console_url = run_gcs_dir.replace("gs://", "https://console.cloud.google.com/storage/browser/", 1)

    gt.logging.configure_logging(run_name=run_name, process_type="training")

    logger.info(f"Run artifacts: {run_console_url}")

    if gpu is not None:
        gt.launch.remote(
            gpu,
            ddp=gpu.endswith("-ddp"),
            zone=zone,
            extra_env={_RUN_NAME_ENV_VAR: run_name},
        )
        return

    is_cuda = torch.cuda.is_available()

    if not tiny_run:
        assert is_cuda, "CUDA is required for full training. Did you mean to pass --tiny_run ?"
        assert torch.cuda.is_bf16_supported(), "Get a GPU that supports bfloat16"

    model = gt.utils.encoder_from_base(base_model, use_text_prefix=use_text_prefix)

    if tiny_run:
        training_config = gt.train.TrainingConfig(
            run_shortname=run_shortname or "tiny-run",
            per_device_train_batch_size=2,
            per_device_token_budget=64,
            gradient_checkpointing=True,
            sample_size_train=30,
            num_logs=30,
            num_checkpoints=2,
            loss_type="contrastive",
            contrastive_margin=0.5,
        )
    else:
        per_device_token_budget_scale = (
            per_device_token_budget_scale or base_model_to_per_device_token_budget_scale.get(base_model, 3)
        )
        training_config = gt.train.TrainingConfig(
            run_shortname=run_shortname,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_token_budget=8192 * per_device_token_budget_scale,
            learning_rate=learning_rate,
            loss_type="contrastive",
            contrastive_margin=0.5,
            training_csvs=gt.data.DEFAULT_TRAIN_PATHS,
        )

    trainer = gt.train.make_trainer(model, training_config, run_name=run_name)

    is_main_process = trainer.accelerator.is_main_process
    eval_was_launched = False

    if is_main_process:
        upload_run_metadata(run_gcs_dir, training_config)

        wandb.login()
        wandb.init(
            project=training_config.wandb_project,
            name=trainer.args.run_name,
            group=trainer.args.run_name,
            settings=wandb.Settings(mode="shared", x_primary=True, x_label="train"),
        )
        logger.info(f"W&B logs: {wandb.run.url}/logs")
        eval_command = (
            f"python eval/eval_poller.py --run_gcs_dir {run_gcs_dir} --base_model {base_model} "
            f"--wandb_run_id {wandb.run.id} --wandb_project {training_config.wandb_project} "
            f"--loss_type {training_config.loss_type} --contrastive_margin {training_config.contrastive_margin}"
        )
        if use_text_prefix:
            eval_command += " --use_text_prefix"
        if tiny_run:
            eval_command += " --sample_val 200 --use_simple_precisions"
        logger.info(f"\nThis command will be run to evaluate the model:\n\n{eval_command}\n")
        if not tiny_run:
            gt.launch.l4_eval(eval_command)
            eval_was_launched = True
        else:
            logger.info("Skipping async eval on L4 for tiny_run")

        trainer.add_callback(gt.train.GCSCheckpointUploadCallback(run_gcs_dir=run_gcs_dir))

    warnings.filterwarnings(
        "ignore",
        message=".*torch.utils.checkpoint: the use_reentrant parameter.*",
        category=UserWarning,
    )
    try:
        logger.info("Training - start")
        trainer.train(resume_from_checkpoint=training_config.resume_from_checkpoint)
        logger.info("Training - complete")

        if is_main_process:
            dir_inference = os.path.join(trainer.args.output_dir, "inference")
            trainer.model.encoder.save_pretrained(dir_inference)
            subprocess.run(
                ["gcloud", "storage", "cp", "-r", "wandb", f"{run_gcs_dir}/wandb"],
                check=True,
            )
            subprocess.run(
                ["gcloud", "storage", "rsync", "-r", dir_inference, f"{run_gcs_dir}/inference"],
                check=True,
            )
            logger.info(f"Uploaded wandb artifacts and model to {run_gcs_dir}")
    finally:
        # So the eval poller always stops polling, exists, and then the instance shuts down
        if eval_was_launched:
            subprocess.run(
                ["gcloud", "storage", "cp", "-", f"{run_gcs_dir}/{gt.sentinels.TRAINING_DONE}"],
                input=b"",
                check=False,
            )


if __name__ == "__main__":
    tapify(run)
