"""
Trains a model, logs to wandb, and saves it to local and GCS.
Evaluation runs async on a separate machine. See eval/eval_poller.py
"""

import logging
import os
import subprocess
import warnings
import tempfile

import torch
import wandb
from tap import tapify

import grouping_trainer as gt

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
    "lightonai/modernbert-embed-large": 4,
    "Alibaba-NLP/gte-modernbert-base": 6,
    "Qwen/Qwen3-Embedding-0.6B": 3,
    "jinaai/jina-embeddings-v5-text-nano-text-matching": 4,
}


def run(
    base_model: str = "lightonai/modernbert-embed-large",
    run_shortname: str | None = None,
    use_prompt_prefix: bool = False,
    per_device_token_budget_scale: int | None = None,
    mini_cpu_test: bool = False,
):
    """
    Train a grouping model.

    Parameters
    ----------
    base_model
        HuggingFace model ID or local path for the base encoder.
        Others we've tried: Alibaba-NLP/gte-modernbert-base, Qwen/Qwen3-Embedding-0.6B,
        jinaai/jina-embeddings-v5-text-nano-text-matching
    run_shortname
        Short name for the run. Doesn't need to be unique b/c it's appended to the timestamp.
    use_prompt_prefix
        If True, add the prompt prefix to the input text. It does not seem to help lightonai/modernbert-embed-large
    per_device_token_budget_scale
        Sets per_device_token_budget = per_device_token_budget_scale * 8192—the max tokens for grouping in prod
    mini_cpu_test
        Run a mini training run on CPU to sanity check plumbing.
    """
    is_cuda = torch.cuda.is_available()

    if not mini_cpu_test:
        assert run_shortname is not None, "run_shortname is required for full training runs"
        assert is_cuda, "CUDA is required for full training. Did you mean to pass --mini_cpu_test ?"
        assert torch.cuda.is_bf16_supported(), "Get a GPU that supports bfloat16"

    model = gt.utils.encoder_from_base(base_model, use_prompt_prefix=use_prompt_prefix)

    if mini_cpu_test:
        training_config = gt.train.TrainingConfig(
            run_shortname=run_shortname or "cpu-sanity-check",
            per_device_train_batch_size=2,
            per_device_token_budget=64,
            gradient_checkpointing=True,
            sample_size_train=30,
            num_logs=30,
            num_checkpoints=2,
            loss_type="contrastive",
            contrastive_margin=0.25,
        )
    else:
        per_device_token_budget_scale = (
            per_device_token_budget_scale or base_model_to_per_device_token_budget_scale.get(base_model, 3)
        )
        training_config = gt.train.TrainingConfig(
            run_shortname=run_shortname,
            per_device_train_batch_size=128,
            per_device_token_budget=8192 * per_device_token_budget_scale,
            loss_type="contrastive",
            contrastive_margin=0.5,
            training_csvs=(
                gt.data.DEFAULT_TRAIN_PATHS
                + (
                    "final_csvs/synthetic-hard-negatives-llm.csv",
                    # "final_csvs/synthetic-hard-positives-llm.csv",
                )
            ),
        )

    trainer = gt.train.make_trainer(model, training_config)
    gt.logging.configure_logging(
        run_name=trainer.args.run_name,
        process_type="training",
    )

    is_main_process = trainer.accelerator.is_main_process
    run_gcs_dir = f"gs://grouping-data/runs/{trainer.args.run_name}"

    if is_main_process:
        upload_run_metadata(run_gcs_dir, training_config)

        wandb.login()
        wandb.init(
            project=training_config.wandb_project,
            name=trainer.args.run_name,
            group=trainer.args.run_name,
            settings=wandb.Settings(mode="shared", x_primary=True, x_label="train"),
        )
        eval_cmd = (
            f"python eval/eval_poller.py --run_gcs_dir {run_gcs_dir} --base_model {base_model} "
            f"--wandb_run_id {wandb.run.id} --wandb_project {training_config.wandb_project} "
            f"--loss_type {training_config.loss_type} --contrastive_margin {training_config.contrastive_margin}"
        )
        if use_prompt_prefix:
            eval_cmd += " --use_prompt_prefix"
        if mini_cpu_test:
            eval_cmd += " --sample_val 200 --use_simple_precisions"
        logger.info(f"\nThis command will be run to evaluate the model:\n\n{eval_cmd}\n")
        if not mini_cpu_test:
            # gt.train.launch_l4_eval(eval_cmd)
            pass
        else:
            logger.info("Skipping async eval on L4 for mini_cpu_test")

        trainer.add_callback(gt.train.GCSCheckpointUploadCallback(run_gcs_dir=run_gcs_dir))

    warnings.filterwarnings(
        "ignore",
        message=".*torch.utils.checkpoint: the use_reentrant parameter.*",
        category=UserWarning,
    )
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


if __name__ == "__main__":
    tapify(run)
