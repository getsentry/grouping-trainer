"""
Trains a model, logs to wandb, and saves it to local and GCS.
Evaluation runs async on a separate machine. See eval/eval_poller.py
"""

import logging
import os
import subprocess
import warnings

import torch
import wandb
from tap import tapify

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def run(mini_cpu_test: bool = False):
    """
    Train a grouping model.

    Parameters
    ----------
    mini_cpu_test
        Run a mini training run on CPU to sanity check plumbing.
    """
    is_cuda = torch.cuda.is_available()

    if not mini_cpu_test:
        assert is_cuda, "CUDA is required for full training. Did you mean to pass --mini_cpu_test ?"
        assert torch.cuda.is_bf16_supported(), "Get a GPU that supports bfloat16"

    model = gt.utils.SentenceTransformer(  # 150M params. small enough we don't need a tiny-random for CPU runs
        "Alibaba-NLP/gte-modernbert-base",
        model_kwargs=(
            dict(
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            if is_cuda
            else None
        ),
    )

    if mini_cpu_test:
        config = gt.train.TrainingConfig(
            run_shortname="cpu-sanity-check",
            per_device_train_batch_size=2,
            per_device_token_budget=64,
            gradient_checkpointing=True,
            sample_size_train=30,
            num_logs=30,
            num_checkpoints=2,
        )
    else:
        config = gt.train.TrainingConfig(
            run_shortname="gte",
            # Sample a large-enough batch to capture a good amount of the same query for cache hits in the forward pass.
            per_device_train_batch_size=64,
            # Accumulate over enough batches to get signal from more projects and reduce gradient variance.
            gradient_accumulation_steps=16,
            per_device_token_budget=8192 * 5,
        )

    trainer = gt.train.make_trainer(model, config)
    gt.logging.configure_logging(
        run_name=trainer.args.run_name,
        process_type="training",
    )

    is_main_process = trainer.accelerator.is_main_process
    run_gcs_dir = f"gs://grouping-data/runs/{trainer.args.run_name}"

    if is_main_process:
        wandb.login()
        wandb.init(project=config.wandb_project, name=trainer.args.run_name)

        base_model = trainer.model.encoder.model_card_data.base_model
        eval_cmd = (
            f"python eval/eval_poller.py"
            f" --run_gcs_dir {run_gcs_dir}"
            f" --wandb_run_id {wandb.run.id}"
            f" --base_model {base_model}"
        )
        if mini_cpu_test:
            eval_cmd += " --sample_val 200 --use_auto_detected_device --use_simple_precisions"
        logger.info(f"\nThis command will be run to evaluate the model:\n\n{eval_cmd}\n")
        if not mini_cpu_test:
            gt.train.launch_l4_eval(eval_cmd)
        else:
            logger.info("Skipping async eval on L4 for mini_cpu_test")

        trainer.add_callback(gt.train.GCSCheckpointUploadCallback(run_gcs_dir=run_gcs_dir))

    warnings.filterwarnings(
        "ignore",
        message=".*torch.utils.checkpoint: the use_reentrant parameter.*",
        category=UserWarning,
    )
    logger.info("Training - start")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
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
