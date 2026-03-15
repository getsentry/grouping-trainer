"""
Trains a model, logs to wandb, and saves it to local and GCS.
Evaluation runs async on a separate machine. See eval/eval_poller.py
"""

import logging
import math
import os
import subprocess
import warnings
import tempfile

import torch
import wandb
from tap import tapify

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def upload_run_metadata(run_gcs_dir: str, config: gt.train.TrainingConfig) -> None:
    """
    Save training config and git commit to a local temp dir, then upload to GCS as metadata/.
    """

    with tempfile.TemporaryDirectory() as dir_metadata:
        with open(os.path.join(dir_metadata, "training_config.json"), "w") as f:
            f.write(config.model_dump_json(indent=2))
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
            training_csvs=(
                "final_csvs/train.csv",
                "final_csvs/synthetic-semi-easy-negatives.csv",
                "final_csvs/train_more.csv",
                "final_csvs/synthetic-hard-negatives-llm.csv",
            ),
            gradient_checkpointing=True,
            sample_size_train=30,
            num_logs=30,
            num_checkpoints=2,
        )
    else:
        config = gt.train.TrainingConfig(
            run_shortname="gte-mix-more",
            per_device_train_batch_size=32,
            gradient_accumulation_steps=8,
            shuffle_within_dataset=True,
            per_device_token_budget=8192 * 6,
            log_of_scale_init=math.log(10),  # TODO: wandb this param and bias
            training_csvs=(
                "final_csvs/train.csv",
                "final_csvs/synthetic-semi-easy-negatives.csv",
                "final_csvs/train_more.csv",
                # "final_csvs/synthetic-hard-negatives-llm.csv",
            ),
        )

    trainer = gt.train.make_trainer(model, config)
    gt.logging.configure_logging(
        run_name=trainer.args.run_name,
        process_type="training",
    )

    is_main_process = trainer.accelerator.is_main_process
    run_gcs_dir = f"gs://grouping-data/runs/{trainer.args.run_name}"

    if is_main_process:
        upload_run_metadata(run_gcs_dir, config)

        wandb.login()
        wandb.init(project=config.wandb_project, name=trainer.args.run_name, group=trainer.args.run_name)

        base_model = trainer.model.encoder.model_card_data.base_model
        eval_cmd = f"python eval/eval_poller.py --run_gcs_dir {run_gcs_dir} --base_model {base_model}"
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
