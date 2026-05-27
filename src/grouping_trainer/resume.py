"""
Resume-from-checkpoint helpers. Haven't needed this yet so it's not battle-tested. TODO: add tests for train.py when
doing interleaved per-project batching.
"""

import json
import logging
import os
import re
import subprocess
from typing import Any

import torch.distributed as dist
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import grouping_trainer as gt

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR_RE = re.compile(rf"^{re.escape(PREFIX_CHECKPOINT_DIR)}-\d+$")


def parse_resume_uri(resume_from: str) -> tuple[str, str | None]:
    """
    Parse a `resume_from` URI into `(run_gcs_dir, checkpoint_gcs_uri_or_None)`.

    If it's a run dir (`gs://$GROUPING_TRAINER_BUCKET/runs/<run_name>`) then `checkpoint_gcs_uri_or_None` is `None`.

    If it's a specific checkpoint (`gs://$GROUPING_TRAINER_BUCKET/runs/<run_name>/checkpoint-<step>`), then
    `checkpoint_gcs_uri_or_None` is that URI.
    """
    uri = resume_from.rstrip("/")
    if not uri.startswith("gs://"):
        raise ValueError(f"resume_from must be a gs:// URI, got: {resume_from}")
    parent, last = uri.rsplit("/", 1)
    if _CHECKPOINT_DIR_RE.match(last):
        return parent, uri
    return uri, None


def find_latest_checkpoint(run_gcs_dir: str) -> str:
    """
    Return the GCS URI of the highest-step `checkpoint-<N>` directory under `run_gcs_dir`.
    """
    args = ["gcloud", "storage", "ls", f"{run_gcs_dir.rstrip('/')}/{PREFIX_CHECKPOINT_DIR}-*"]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        gt.utils.log_and_raise_subprocess(result, args, log_prefix=f"find_latest_checkpoint({run_gcs_dir}) ")
    uris = [line.rstrip("/") for line in result.stdout.splitlines() if line.strip()]
    checkpoints = [uri for uri in uris if _CHECKPOINT_DIR_RE.match(uri.rsplit("/", 1)[-1])]
    if not checkpoints:
        raise ValueError(f"No {PREFIX_CHECKPOINT_DIR}-* directories under {run_gcs_dir}")
    return max(checkpoints, key=lambda uri: int(uri.rsplit("-", 1)[-1]))


def download_checkpoint(checkpoint_gcs_uri: str, output_dir_local: str) -> str:
    """
    Rsync `checkpoint_gcs_uri` into `output_dir_local/checkpoint-<step>/` and return the local path.
    """
    basename = checkpoint_gcs_uri.rstrip("/").rsplit("/", 1)[-1]
    path_local = os.path.join(output_dir_local, basename)
    logger.info(f"Downloading checkpoint: {checkpoint_gcs_uri} -> {path_local}")
    os.makedirs(output_dir_local, exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "rsync", "-r", checkpoint_gcs_uri.rstrip("/"), path_local],
        check=True,
    )
    return path_local


def read_uploaded_config(run_gcs_dir: str, config_filename: str) -> dict[str, Any]:
    """
    Fetch and parse `{run_gcs_dir}/metadata/{config_filename}`, uploaded by `gt.launch.upload_run_metadata`.
    """
    result = subprocess.run(
        ["gcloud", "storage", "cat", f"{run_gcs_dir.rstrip('/')}/metadata/{config_filename}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def download_for_resume(trainer: Trainer, run_gcs_dir: str, checkpoint_gcs_uri: str | None) -> str:
    """
    Pull the resume checkpoint locally on the main process, block the other ranks, and return the local path to hand to
    `trainer.train(resume_from_checkpoint=...)`.

    `checkpoint_gcs_uri=None` means "pick the latest checkpoint in `run_gcs_dir`".
    """
    assert trainer.args.output_dir is not None

    # Resolve the URI on rank 0 only and broadcast, so ranks can't disagree on which checkpoint to load.
    checkpoint_gcs_holder: list[str | None] = [None]
    if trainer.accelerator.is_main_process:
        checkpoint_gcs_holder[0] = checkpoint_gcs_uri or find_latest_checkpoint(run_gcs_dir)
    if trainer.accelerator.num_processes > 1:
        dist.broadcast_object_list(checkpoint_gcs_holder, src=0)
    checkpoint_gcs = checkpoint_gcs_holder[0]

    assert checkpoint_gcs is not None
    if trainer.accelerator.is_main_process:
        checkpoint_path_local = download_checkpoint(checkpoint_gcs, trainer.args.output_dir)
    else:
        basename = checkpoint_gcs.rstrip("/").rsplit("/", 1)[-1]
        checkpoint_path_local = os.path.join(trainer.args.output_dir, basename)
    trainer.accelerator.wait_for_everyone()
    return checkpoint_path_local
