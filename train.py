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

_RUN_NAME_ENV_VAR = "GROUPING_TRAINER_RUN_NAME"

logger = logging.getLogger(__name__)


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
    per_device_token_budget_scale: int | None = None,
    global_train_batch_size: int = 256,
    learning_rate: float = 1e-4,
    tiny_run: bool = False,
    use_text_prefix: bool = False,
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
        bucket (downloaded into `_base_models/` on the instance). Others we've tried:
        Alibaba-NLP/gte-modernbert-base, Qwen/Qwen3-Embedding-0.6B, jinaai/jina-embeddings-v5-text-nano-text-matching
    run_shortname
        Short name for the run. Doesn't need to be unique b/c it's appended to the timestamp.
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
    gpu
        The type of GPU to flex-start and run on.
    zone
        Override the default GCP zone for the gpu type when launching the GPU instance. Useful when flex-start capacity
        is dry in the default zone for the requested gpu type.
    sync_start
        If False (default), flex-starts the instance—GCP waits up to 1h to find one. `--sync_start` uses on-demand
        pricing and finds an instance in any zone, as flex-starting often can't find instances in time.
    multi_flex_start
        Fan out async flex-start submits across 10 zones simultaneously; first VM to boot claims a GCS lock and the rest
        self-delete. Better odds than a single-zone flex-start when capacity is dry. Mutually exclusive with
        --sync_start
    """
    if not tiny_run:
        assert run_shortname is not None, "run_shortname is required for full training runs"

    # Fail fast on a typo'd gs:// model URI before wasting time launching training.
    if gt.utils.is_gcs_uri(base_model):
        gt.utils.assert_gcs_path_exists(base_model)

    # Generate run_name up front so we can log the artifact URL locally before auto-launching. On the remote, re-use the
    # local run_name via env var so both sides log the same GCS path (rather than each generating its own timestamp).
    run_name = os.environ.get(_RUN_NAME_ENV_VAR) or gt.launch.run_name_from_shortname(run_shortname or "tiny-run")
    run_gcs_dir = f"gs://{os.environ['GROUPING_TRAINER_BUCKET']}/runs/{run_name}"
    run_console_url = run_gcs_dir.replace("gs://", "https://console.cloud.google.com/storage/browser/", 1)

    gt.logging.configure_logging(run_name=run_name, process_type="training")

    logger.info(f"Run artifacts: {run_console_url}")
    if (wandb_entity := os.environ.get("WANDB_ENTITY")) and (wandb_project := os.environ.get("WANDB_PROJECT")):
        logger.info(f"W&B logs: https://wandb.ai/{wandb_entity}/{wandb_project}/runs/{run_name}/logs")

    if gpu is not None:
        gt.launch.run_argv_remotely(
            gpu=gpu,
            job_type=gt.launch.JobType.TRAIN,
            name_suffix=run_shortname or "tiny-run",
            sync_start=sync_start,
            multi_flex_start=multi_flex_start,
            zone=zone,
            env_var_to_value={_RUN_NAME_ENV_VAR: run_name},
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
        assert run_shortname is not None
        per_device_token_budget_scale = (
            per_device_token_budget_scale or base_model_to_per_device_token_budget_scale.get(base_model, 3)
        )
        training_config = gt.train.TrainingConfig(
            run_shortname=run_shortname,
            global_train_batch_size=global_train_batch_size,
            per_device_token_budget=8192 * per_device_token_budget_scale,
            learning_rate=learning_rate,
            loss_type="contrastive",
            contrastive_margin=0.5,
            training_csvs=gt.data.DEFAULT_TRAIN_PATHS,
        )

    gt.data.ensure_local(training_config.training_csvs)

    trainer = gt.train.make_trainer(model, training_config, run_name=run_name)
    eval_was_launched = False

    if trainer.accelerator.is_main_process:
        gt.launch.upload_run_metadata(run_gcs_dir, training_config, config_filename="training_config.json")

        wandb.login()
        wandb.init(
            id=run_name,
            name=trainer.args.run_name,
            group=trainer.args.run_name,
            settings=wandb.Settings(mode="shared", x_primary=True, x_label="train"),
        )
        assert wandb.run is not None

        # Start eval poller on a separate machine. WANDB_ENTITY/WANDB_PROJECT are forwarded by gce_vm.
        eval_command = (
            f"python eval/eval_poller.py --run_gcs_dir {run_gcs_dir} --base_model {base_model} "
            f"--wandb_run_id {wandb.run.id} "
            f"--loss_type {training_config.loss_type} --contrastive_margin {training_config.contrastive_margin}"
        )
        if use_text_prefix:
            eval_command += " --use_text_prefix"
        if tiny_run:
            eval_command += " --sample_val 200 --use_simple_precisions"
        logger.info(f"\nThis command will be run to evaluate the model:\n\n{eval_command}\n")
        if not tiny_run:
            assert run_shortname is not None
            gt.launch.gce_vm(
                gpu="l4",
                job_type=gt.launch.JobType.EVAL,
                name_suffix=run_shortname,
                command=eval_command,
            )
            logger.info("Created l4-eval instance with eval poller in startup script")
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

        if trainer.accelerator.is_main_process:
            assert trainer.args.output_dir is not None
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
