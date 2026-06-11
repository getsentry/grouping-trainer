"""
Trains a model, logs to wandb, and saves it to local and GCS.
Evaluation runs async on a separate machine. See eval/eval_poller.py
"""

import logging
import math
import os
import subprocess
import warnings

import torch
from tap import tapify

import grouping_trainer as gt

logger = logging.getLogger(__name__)


base_model_to_per_device_token_budget_scale = {
    # jinaai/jina-embeddings-v2-base-code seemed to not support SDPA
    "lightonai/modernbert-embed-large": 4.0,
    "Alibaba-NLP/gte-modernbert-base": 6.0,
    "Qwen/Qwen3-Embedding-0.6B": 3.0,
    "jinaai/jina-embeddings-v5-text-nano-text-matching": 4.0,
    "microsoft/harrier-oss-v1-0.6b": 3.0,
    "BidirLM/BidirLM-1B-Embedding": 0.7,  # NOTE: doesn't support gradient_checkpointing
}


def run(
    base_model: str = "lightonai/modernbert-embed-large",
    run_shortname: str | None = None,
    resume_from: str | None = None,
    per_device_token_budget_scale: float | None = None,
    global_train_batch_size: int = 256,
    learning_rate: float = 1e-4,
    tiny_run: bool = False,
    use_text_prefix: bool = False,
    dont_spin_up_eval_poller: bool = False,
    *,
    gpu: gt.launch.TrainingGpuType | None = None,  # type: ignore[valid-type]
    zone: str | None = None,
    sync_start: bool = False,
    multi_flex_start: bool = False,
):
    """
    Train a grouping model. Writes checkpoints to GCS. Logs to wandb.

    Parameters
    ----------
    base_model
        HuggingFace model ID or local path for the base encoder, or a `gs://...` path to a custom model directory in our
        bucket, e.g., the checkpoint to a model pretrained using pretrain.py. Others we've tried:
        Alibaba-NLP/gte-modernbert-base, Qwen/Qwen3-Embedding-0.6B, jinaai/jina-embeddings-v5-text-nano-text-matching,
        microsoft/harrier-oss-v1-0.6b, BidirLM/BidirLM-1B-Embedding
    run_shortname
        Short name for the run. Doesn't need to be unique b/c it's appended to the timestamp. Not required when
        `--resume_from` is given (derived from the resumed run's name).
    resume_from
        Resume from a previously-uploaded run. If it's a run dir (`gs://$GROUPING_TRAINER_BUCKET/runs/<run_name>`) then
        the latest checkpoint is picked. Otherwise you specify the checkpoint
        (`gs://$GROUPING_TRAINER_BUCKET/runs/<run_name>/checkpoint-<step>`). Reuses the original run_name, GCS run dir,
        and W&B run id.
    per_device_token_budget_scale
        The scale in per_device_token_budget = scale * 8192. This is the memory and throughput knob. By default, if the
        base_model has a historically known good scale, we use that, o.w. uses 3.
    global_train_batch_size
        Total training batch size across all devices. Only used for non-tiny runs. Technically this can be arbitrarily
        high b/c we accumulate the gradient based on per_device_token_budget_scale.
    tiny_run
        Run a tiny training run to sanity check plumbing.
    use_text_prefix
        If True, add the model's designated prefix to the input text. Didn't help lightonai/modernbert-embed-large
    dont_spin_up_eval_poller
        Set this flag while debugging to skip spinning up the eval poller.
    gpu
        The type of GPU to flex-start and run on.
    zone
        Override the default GCP zone for the gpu type when launching the GPU instance. Useful when flex-start capacity
        is dry in the default zone for the requested gpu type.
    sync_start
        If False (default), flex-starts the instance—GCP waits up to 2h to find one. `--sync_start` uses on-demand
        pricing and finds an instance in any zone, as flex-starting often can't find instances in time.
    multi_flex_start
        Fan out async flex-start submits across many zones simultaneously; first VM to boot claims a GCS lock and the
        rest self-delete. Better odds than a single-zone flex-start when capacity is dry. Mutually exclusive with
        --sync_start
    """
    if resume_from is None and gt.utils.is_gcs_uri(base_model):
        gt.utils.assert_gcs_path_exists(base_model)

    if resume_from is None:
        run_shortname = run_shortname or (f"tiny-{gt.launch.JobType.TRAIN}" if tiny_run else None)

    run = gt.launch.bootstrap_run(run_shortname=run_shortname, process_type="training", resume_from=resume_from)

    if resume_from is not None:
        base_model = gt.resume.read_uploaded_config(run.gcs_dir, "training_config.json")["base_model"]

    if gpu is not None:
        gt.launch.run_argv_remotely(
            gpu=gpu,
            job_type=gt.launch.JobType.TRAIN,
            run_name=run.name,
            sync_start=sync_start,
            multi_flex_start=multi_flex_start,
            zone=zone,
        )
        return

    is_cuda = torch.cuda.is_available()

    if not tiny_run:
        assert is_cuda, "CUDA is required for full training. Did you mean to pass --tiny_run ?"
        assert torch.cuda.is_bf16_supported(), "Get a GPU that supports bfloat16"

    # TODO: when resuming, load from the checkpoint instead so we skip a redundant gs:// base-model rsync.
    model = gt.utils.encoder_from_base(base_model, use_text_prefix=use_text_prefix)

    if tiny_run:
        training_config = gt.train.TrainingConfig(
            run_shortname=run.shortname,
            base_model=base_model,
            global_train_batch_size=2,
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
            run_shortname=run.shortname,
            base_model=base_model,
            global_train_batch_size=global_train_batch_size,
            per_device_token_budget=math.floor(8192 * per_device_token_budget_scale),
            warmup_ratio=0.25,
            learning_rate=learning_rate,
            loss_type="contrastive",
            contrastive_margin=0.6,
            training_csvs=gt.data.DEFAULT_TRAIN_PATHS_NO_SYNTHETIC,
        )

    gt.data.ensure_local(training_config.training_csvs)

    trainer = gt.train.make_trainer(model, training_config, run_name=run.name)
    eval_was_launched = False

    checkpoint_path_local: str | None = None
    if resume_from is not None:
        checkpoint_path_local = gt.resume.download_for_resume(trainer, run.gcs_dir, run.checkpoint_gcs_uri)
        if trainer.accelerator.is_main_process:
            subprocess.run(
                ["gcloud", "storage", "rm", f"{run.gcs_dir}/{gt.sentinels.TRAINING_DONE}"],
                check=False,
            )

    if trainer.accelerator.is_main_process:
        gt.launch.upload_run_metadata(run.gcs_dir, training_config, config_filename="training_config.json")
        gt.launch.init_wandb(run_name=run.name, display_name=run.shortname, x_label="train", resume_from=resume_from)

        # Start eval poller on a separate machine
        eval_command = (
            f"python eval/eval_poller.py --run_gcs_dir {run.gcs_dir} --base_model {base_model} "
            f"--wandb_run_id {run.name} --loss_type {training_config.loss_type} "
            f"--contrastive_margin {training_config.contrastive_margin}"
        )
        if use_text_prefix:
            eval_command += " --use_text_prefix"
        if tiny_run:
            eval_command += " --sample_val 200 --use_simple_precisions"
        logger.info(f"This command will be run to evaluate the model:\n\n{eval_command}\n")
        if not (tiny_run or dont_spin_up_eval_poller):
            gt.launch.gce_vm(
                gpu="l4",
                job_type=gt.launch.JobType.EVAL,
                name_suffix=run.shortname,
                command=eval_command,
                delete_if_exists=resume_from is not None,
            )
            logger.info("Created l4-eval instance with eval poller in startup script")
            eval_was_launched = True
        else:
            logger.info("Skipping async eval on L4 for tiny_run")

        trainer.add_callback(gt.train.GCSCheckpointUploadCallback(run_gcs_dir=run.gcs_dir))

    warnings.filterwarnings(
        "ignore",
        message=".*torch.utils.checkpoint: the use_reentrant parameter.*",
        category=UserWarning,
    )

    try:
        logger.info("Training started")
        trainer.train(resume_from_checkpoint=checkpoint_path_local)
        logger.info("Training completed")

        if trainer.accelerator.is_main_process:
            assert trainer.args.output_dir is not None
            dir_inference = os.path.join(trainer.args.output_dir, "inference")
            trainer.model.encoder.save_pretrained(dir_inference)
            subprocess.run(
                ["gcloud", "storage", "cp", "-r", "wandb", f"{run.gcs_dir}/wandb"],
                check=True,
            )
            subprocess.run(
                ["gcloud", "storage", "rsync", "-r", dir_inference, f"{run.gcs_dir}/inference"],
                check=True,
            )
            logger.info(f"Uploaded wandb artifacts and model to {run.gcs_dir}")
    finally:
        # So the eval poller always stops polling, exists, and then the instance shuts down
        if eval_was_launched:
            subprocess.run(
                ["gcloud", "storage", "cp", "-", f"{run.gcs_dir}/{gt.sentinels.TRAINING_DONE}"],
                input=b"",
                check=False,
            )


if __name__ == "__main__":
    tapify(run)
