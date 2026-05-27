"""
Continues MLM pretraining of a base encoder on Sentry-grouping LLM prompts and completions:
`prompt[SEP]thinking_output[SEP]response_output`

Logs to wandb. Writes checkpoints + the final model to GCS. Unlike `train.py`, there's no async eval. Just MLM loss on a
subsample of val data run sync.
"""

import logging
import os
import subprocess
import warnings

import torch
from tap import tapify

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def run(
    base_model: str = "answerdotai/ModernBERT-large",
    run_shortname: str | None = None,
    resume_from: str | None = None,
    global_train_batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 1.0,
    max_seq_length: int = 8192,
    sample_size: int | None = None,
    gradient_checkpointing: bool = True,
    sort_by_seq_length_desc: bool = False,
    tiny_run: bool = False,
    *,
    gpu: gt.launch.TrainingGpuType | None = None,  # type: ignore[valid-type]
    zone: str | None = None,
    sync_start: bool = False,
    multi_flex_start: bool = False,
):
    """
    Continue MLM pretraining of `base_model` on prompts w/ LLM responses. Writes checkpoints to GCS. Logs to wandb.

    Parameters
    ----------
    base_model
        HuggingFace model ID, local path, or `gs://...` path to a custom model directory in our bucket (downloaded into
        `_base_models/` on the instance).
    run_shortname
        Short name for the run. Doesn't need to be unique b/c it's appended to the timestamp. Not required when
        `--resume_from` is given (derived from the resumed run's name).
    resume_from
        Resume from a previously-uploaded run. If it's a run dir (`gs://$GROUPING_TRAINER_BUCKET/runs/<run_name>`) then
        the latest checkpoint is picked. Otherwise you specify the checkpoint
        (`gs://$GROUPING_TRAINER_BUCKET/runs/<run_name>/checkpoint-<step>`). Reuses the original run_name, GCS run dir,
        and W&B run id.
    global_train_batch_size
        Total batch size across all devices. Only used for non-tiny runs.
    max_seq_length
        Truncate inputs to this many tokens. ModernBERT supports up to 8192.
    sample_size
        If set, downsample the unique-texts corpus to this many. Useful for quick iteration.
    gradient_checkpointing
        Trade compute for memory. Useful for fitting large batches / long contexts.
    sort_by_seq_length_desc
        Stress-probe mode: train iterates longest sequences first so OOMs surface in the first few steps. Ctrl+C after a
        few successful steps. This flag isn't meant for real training.
    tiny_run
        Tiny CPU/GPU sanity check.
    gpu
        The type of GPU to flex-start and run on.
    zone
        Override the default GCP zone for the gpu type when launching the GPU instance.
    sync_start
        If False (default), flex-starts the instance. `--sync_start` uses on-demand pricing.
    multi_flex_start
        Fan out async flex-start submits across many zones simultaneously; first VM to boot claims a GCS lock and the
        rest self-delete. Better odds than a single-zone flex-start when capacity is dry. Mutually exclusive with
        --sync_start
    """
    if resume_from is None and gt.utils.is_gcs_uri(base_model):
        gt.utils.assert_gcs_path_exists(base_model)

    if resume_from is None:
        run_shortname = run_shortname or (f"tiny-{gt.launch.JobType.PRETRAIN}" if tiny_run else None)

    run = gt.launch.bootstrap_run(run_shortname=run_shortname, process_type="pretrain", resume_from=resume_from)

    if resume_from is not None:
        base_model = gt.resume.read_uploaded_config(run.gcs_dir, "pretraining_config.json")["base_model"]

    if gpu is not None:
        gt.launch.run_argv_remotely(
            gpu=gpu,
            job_type=gt.launch.JobType.PRETRAIN,
            run_name=run.name,
            sync_start=sync_start,
            multi_flex_start=multi_flex_start,
            zone=zone,
        )
        return

    is_cuda = torch.cuda.is_available()
    if not tiny_run:
        assert is_cuda, "CUDA is required for full pretraining. Did you mean to pass --tiny_run ?"
        assert torch.cuda.is_bf16_supported(), "Get a GPU that supports bfloat16"

    if tiny_run:
        pretraining_config = gt.pretrain.PretrainingConfig(
            run_shortname=run.shortname,
            base_model=base_model,
            global_train_batch_size=2,
            learning_rate=learning_rate,
            max_seq_length=128,
            sample_size=30,
            n_rows_per_csv=100,
            num_logs=5,
            num_checkpoints=2,
            gradient_checkpointing=True,
            eval_sample_size=10,
        )
    else:
        pretraining_config = gt.pretrain.PretrainingConfig(
            run_shortname=run.shortname,
            base_model=base_model,
            global_train_batch_size=global_train_batch_size,
            mlm_probability=0.3,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            max_seq_length=max_seq_length,
            sample_size=sample_size,
            gradient_checkpointing=gradient_checkpointing,
            sort_by_seq_length_desc=sort_by_seq_length_desc,
        )

    gt.data.ensure_local(pretraining_config.training_csvs)
    if pretraining_config.eval_sample_size is not None:
        gt.data.ensure_local(gt.data.DEFAULT_VAL_PATHS)

    pretrainer = gt.pretrain.make_pretrainer(pretraining_config, run_name=run.name, is_resumed=resume_from is not None)

    checkpoint_path_local = (
        gt.resume.download_for_resume(pretrainer, run.gcs_dir, run.checkpoint_gcs_uri)
        if resume_from is not None
        else None
    )

    if pretrainer.accelerator.is_main_process:
        gt.launch.upload_run_metadata(run.gcs_dir, pretraining_config, config_filename="pretraining_config.json")
        gt.launch.init_wandb(run_name=run.name, x_label="pretrain", resume_from=resume_from)
        pretrainer.add_callback(gt.train.GCSCheckpointUploadCallback(run_gcs_dir=run.gcs_dir))

    warnings.filterwarnings(
        "ignore",
        message=".*torch.utils.checkpoint: the use_reentrant parameter.*",
        category=UserWarning,
    )
    logger.info("Pretraining started")
    pretrainer.train(resume_from_checkpoint=checkpoint_path_local)
    logger.info("Pretraining completed")

    if pretrainer.accelerator.is_main_process:
        assert pretrainer.args.output_dir is not None
        dir_final = os.path.join(pretrainer.args.output_dir, "final")
        pretrainer.save_model(dir_final)
        subprocess.run(
            ["gcloud", "storage", "cp", "-r", "wandb", f"{run.gcs_dir}/wandb"],
            check=True,
        )
        subprocess.run(
            ["gcloud", "storage", "rsync", "-r", dir_final, f"{run.gcs_dir}/final"],
            check=True,
        )
        logger.info(f"Uploaded wandb artifacts and final model to {run.gcs_dir}")


if __name__ == "__main__":
    tapify(run)
