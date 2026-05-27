"""
Launch GCE instances. The instance's startup script does bin/_startup.sh to set up the Python env and then `eval`s
whatever `python` command was locally run.

Training jobs don't need to start in time, so by default they're launched async via flex-start w/ a max wait time of 2
hours. Also saves some money. https://docs.cloud.google.com/compute/docs/instances/about-flex-start-vms. Specifically,
an instance is flex-started in every zone and races to write a lock in GCS identifying this launch. The first to write
it wins, the rest self-delete.

Eval jobs which use cheap L4 GPUs are launched by sync-looping through zones ourselves b/c eval ideally starts in time,
e.g., training shouldn't start w/o an eval poller. L4s are cheap-enough that the flex-start discount isn't worth the
flex-start headache. In current year, GCE doesn't have a simple, built-in cross-region fallback mechanism. Our jobs have
no networking or region dependence. They communicate via GCS if at all.
"""

import contextlib
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

import wandb
from pydantic import AfterValidator, BaseModel, ConfigDict
from tap import tapify

import grouping_trainer as gt

logger = logging.getLogger(__name__)

_REMOTE_ENV_VAR = "GROUPING_TRAINER_REMOTE"
"""
Env var set by `run_argv_remotely()` in the cmd that runs on the remote instance, so the remote knows it shouldn't try
to re-launch. Callers can check `is_on_remote()` if it wants to avoid launch from remote. The main training job doesn't
check `is_on_remote()` so that it can launch the eval poller.
"""

_RUN_NAME_ENV_VAR = "GROUPING_TRAINER_RUN_NAME"
"""
Env var forwarded by `run_argv_remotely()` so the remote `bootstrap_run` reuses the local run name.
"""

_IMAGE = "projects/ml-images/global/images/pytorch-2-7-cu128-ubuntu-2404-nvidia-570-v20260323"

_INSTANCE_NAME_PREFIX = "gt"

_FLEX_START_REQUEST_VALID_FOR_DURATION = "2h"

_TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S"


class JobType(StrEnum):
    TRAIN = "train"
    PRETRAIN = "pretrain"
    EVAL = "eval"
    SAVE = "save"
    SYNTH = "synth"
    BENCHMARK = "bench"
    SSH = "ssh"


def is_on_remote() -> bool:
    return bool(os.environ.get(_REMOTE_ENV_VAR))


def run_name_from_shortname(shortname: str) -> str:
    timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
    return f"{timestamp}-{shortname}"


def shortname_from_run_name(run_name: str) -> str:
    """
    Extract the shortname from a run_name formatted as `YYYY-MM-DD-HH-MM-SS-<shortname>`.
    Returns `run_name` unchanged if it isn't in that format.
    """
    maxsplit = len(_TIMESTAMP_FORMAT.split("-"))
    parts = run_name.split("-", maxsplit=maxsplit)
    return parts[-1] if len(parts) == (maxsplit + 1) else run_name


def check_run_has_model_for_inference(run_gcs_dir: str) -> None:
    path_inference = f"{run_gcs_dir.rstrip('/')}/inference/"
    subprocess.run(["gcloud", "storage", "ls", path_inference], check=True, stdout=subprocess.DEVNULL)


def check_run_shortname(run_shortname: str) -> str:
    """
    Validate that `run_shortname` is usable as the suffix of a GCE instance name (see `gce_vm`). See
    https://docs.cloud.google.com/compute/docs/naming-resources.
    """
    # Worst-case prefix is `gt-{gpu}-{job_type}-`. A limit of 35 should be fine
    gce_instance_name_max_length = 63
    shortname_max_length = 35
    pattern = r"[a-z]([-a-z0-9]*[a-z0-9])?"
    if not re.fullmatch(pattern, run_shortname):
        raise ValueError(
            f"run_shortname {run_shortname!r} is not a valid GCE instance name suffix. Must match {pattern!r}: "
            "lowercase letters, digits, and hyphens; must start with a letter and not end with a hyphen."
        )
    if len(run_shortname) > shortname_max_length:
        raise ValueError(
            f"run_shortname {run_shortname!r} is {len(run_shortname)} chars; must be <= {shortname_max_length} so the "
            f"full GCE instance name comfortably fits in {gce_instance_name_max_length} chars."
        )
    return run_shortname


class RunInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    gcs_dir: str
    checkpoint_gcs_uri: str | None
    "Set when `resume_from` pointed at a specific `checkpoint-<step_number>` (rather than the run dir)."
    shortname: Annotated[str, AfterValidator(check_run_shortname)]


def bootstrap_run(
    *,
    run_shortname: str | None,
    process_type: str,
    resume_from: str | None = None,
) -> RunInfo:
    """
    Returns, e.g., for `run_shortname="my-run"`:

        ```
        RunInfo(
            name="2026-05-25-12-34-56-my-run",
            gcs_dir="gs://$GROUPING_TRAINER_BUCKET/runs/2026-05-25-12-34-56-my-run",
            checkpoint_gcs_uri=None,
            shortname="my-run",
        )
        ```
    """
    # Check the run name can be unambiguously inferred
    if (resume_from is not None) and (run_shortname is not None):
        raise ValueError("Pass either run_shortname or resume_from, not both")
    if (
        ((run_name_from_env := os.environ.get(_RUN_NAME_ENV_VAR)) is None)
        and (resume_from is None)
        and (run_shortname is None)
    ):
        raise ValueError(
            f"Pass run_shortname or resume_from (or set ${_RUN_NAME_ENV_VAR}—done automatically by --gpu launches)"
        )

    # Resolve run_name, run_gcs_dir, run_shortname, and (if resume_from is given) checkpoint_gcs_uri
    if resume_from is not None:
        run_gcs_dir, checkpoint_gcs_uri = gt.resume.parse_resume_uri(resume_from)
        run_name = run_gcs_dir.rstrip("/").rsplit("/", 1)[-1]
        run_shortname = shortname_from_run_name(run_name)
        gt.utils.assert_gcs_path_exists(checkpoint_gcs_uri or run_gcs_dir)
    else:
        if run_name_from_env is None:
            assert run_shortname is not None
            run_name = run_name_from_shortname(run_shortname)
        else:
            run_name = run_name_from_env
            if run_shortname is None:
                run_shortname = shortname_from_run_name(run_name)
        run_gcs_dir = f"gs://{os.environ['GROUPING_TRAINER_BUCKET']}/runs/{run_name}"
        checkpoint_gcs_uri = None

    # Log run info locally and remotely
    gt.logging.configure_logging(run_name=run_name, process_type=process_type)
    run_console_url = run_gcs_dir.replace("gs://", "https://console.cloud.google.com/storage/browser/", 1)
    logger.info(f"Run artifacts: {run_console_url}")
    if (wandb_entity := os.environ.get("WANDB_ENTITY")) and (wandb_project := os.environ.get("WANDB_PROJECT")):
        logger.info(f"W&B logs: https://wandb.ai/{wandb_entity}/{wandb_project}/runs/{run_name}/logs")

    return RunInfo(
        name=run_name,
        gcs_dir=run_gcs_dir,
        checkpoint_gcs_uri=checkpoint_gcs_uri,
        shortname=run_shortname,
    )


def init_wandb(*, run_name: str, x_label: str, resume_from: str | None = None) -> None:
    wandb.login()
    wandb.init(
        id=run_name,
        name=run_name,
        group=run_name,
        settings=wandb.Settings(mode="shared", x_primary=True, x_label=x_label),
        resume="must" if resume_from is not None else None,
    )
    assert wandb.run is not None


class GpuConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default_zone_for_flex_start: str | None
    "`None` disables FLEX_START launches (currently just for L4)."
    zones: tuple[str, ...]
    machine_type: str
    accelerator: str | None  # None for multi-GPU variants b/c accelerators are built into the machine type
    max_run_duration: str
    install_nvidia_driver: bool
    reservation_affinity: Literal["none", "any"]
    is_for_training: bool
    n_gpu: int
    boot_disk_type: str = "pd-balanced"


# gcloud compute accelerator-types list --filter="name=nvidia-l4" --format='value(zone)'
L4_ZONES = (
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east1-b",
    "us-east1-c",
    "us-east1-d",
    "us-east4-a",
    "us-east4-c",
    "us-west1-a",
    "us-west1-b",
    "us-west1-c",
    "us-west4-a",
    "us-west4-c",
    "northamerica-northeast1-b",
    "northamerica-northeast1-c",
    "northamerica-northeast2-a",
    "northamerica-northeast2-b",
    "asia-east1-a",
    "asia-east1-b",
    "asia-east1-c",
    "asia-northeast1-a",
    "asia-northeast1-b",
    "asia-northeast1-c",
    "asia-northeast3-a",
    "asia-northeast3-b",
    "asia-south1-a",
    "asia-south1-b",
    "asia-south1-c",
    "asia-southeast1-a",
    "asia-southeast1-b",
    "asia-southeast1-c",
    "europe-west1-b",
    "europe-west1-c",
    "europe-west2-a",
    "europe-west2-b",
    "europe-west3-a",
    "europe-west3-b",
    "europe-west4-a",
    "europe-west4-b",
    "europe-west4-c",
    "europe-west6-b",
    "europe-west6-c",
    "me-central2-a",
    "me-central2-c",
)
H100_ZONES = (
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-east4-a",
    "us-east4-b",
    "us-east4-c",
    "us-east5-a",
    "us-west1-a",
    "us-west1-b",
    "us-west4-a",
    "asia-east1-c",
    "asia-northeast1-b",
    "asia-south1-c",
    "asia-south2-b",
    "asia-southeast1-b",
    "asia-southeast1-c",
    "australia-southeast1-c",
    "europe-north1-c",
    "europe-west1-b",
    "europe-west1-c",
    "europe-west3-c",
    "europe-west4-b",
    "europe-west4-c",
    "europe-west9-c",
)
# Our quota allows 4 A100 80 GB GPUs in us-central1-a, us-east4-c, europe-west4-a
A100_ZONES = (
    "us-central1-a",
    "us-east4-c",
    "europe-west4-a",
)
# gcloud compute accelerator-types list --filter="name=nvidia-h200-141gb" --format='value(zone)'
H200_ZONES = (
    "us-central1-b",
    "us-east4-b",
    "us-east5-a",
    "us-south1-b",
    "us-west1-c",
    "asia-south1-b",
    "asia-south2-c",
    "europe-west1-b",
    "europe-west4-a",
)

GpuType = Literal[
    "l4",
    "h100",
    "h100-2",
    "h100-4",
    "a100",
    "a100-2",
    "a100-4",
    "h200-8",
]

gpu_type_to_config: dict[GpuType, GpuConfig] = {
    "l4": GpuConfig(
        default_zone_for_flex_start=None,
        zones=L4_ZONES,
        machine_type="g2-standard-4",
        accelerator="count=1,type=nvidia-l4",
        max_run_duration="172800s",
        install_nvidia_driver=False,
        reservation_affinity="any",
        is_for_training=False,
        n_gpu=1,
    ),
    "h100": GpuConfig(
        default_zone_for_flex_start="us-central1-a",
        zones=H100_ZONES,
        machine_type="a3-highgpu-1g",
        accelerator="count=1,type=nvidia-h100-80gb",
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=1,
    ),
    "h100-2": GpuConfig(
        default_zone_for_flex_start="us-central1-a",
        zones=H100_ZONES,
        machine_type="a3-highgpu-2g",
        accelerator=None,
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=2,
    ),
    "h100-4": GpuConfig(
        default_zone_for_flex_start="us-central1-a",
        zones=H100_ZONES,
        machine_type="a3-highgpu-4g",
        accelerator=None,
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=4,
    ),
    "a100": GpuConfig(
        default_zone_for_flex_start="us-central1-a",
        zones=A100_ZONES,
        machine_type="a2-ultragpu-1g",
        accelerator="count=1,type=nvidia-a100-80gb",
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=1,
    ),
    "a100-2": GpuConfig(
        default_zone_for_flex_start="us-central1-a",
        zones=A100_ZONES,
        machine_type="a2-ultragpu-2g",
        accelerator=None,
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=2,
    ),
    "a100-4": GpuConfig(
        default_zone_for_flex_start="us-central1-a",
        zones=A100_ZONES,
        machine_type="a2-ultragpu-4g",
        accelerator=None,
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=4,
    ),
    "h200-8": GpuConfig(
        default_zone_for_flex_start="us-central1-a",
        zones=H200_ZONES,
        machine_type="a3-ultragpu-8g",
        accelerator=None,
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=8,
        boot_disk_type="hyperdisk-balanced",  # doesn't support pd-balanced
    ),
}


TrainingGpuType = Literal[
    tuple(gpu_type for gpu_type, config in gpu_type_to_config.items() if config.is_for_training)  # type: ignore[invalid-literal]
]


def upload_run_metadata(run_gcs_dir: str, config: BaseModel, config_filename: str = "config.json") -> None:
    """
    Upload `config` and the git commit to `<run_gcs_dir>/metadata/`.
    """
    with tempfile.TemporaryDirectory() as dir_metadata:
        with open(os.path.join(dir_metadata, config_filename), "w") as f:
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


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _local_head_sha() -> str:
    repo_root = _repo_root()

    def git(*args: str) -> str:
        cmd = ["git", "-C", repo_root, *args]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            gt.utils.log_and_raise_subprocess(result, cmd, log_prefix="git failed ")
        return result.stdout.strip()

    sha = git("rev-parse", "HEAD")

    # On the remote, fetch-by-SHA only updates FETCH_HEAD (no remote-tracking ref), so `branch -r --contains` would
    # spuriously fail.
    if is_on_remote():
        return sha

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    # Refresh remote-tracking refs so a push from another machine (or via `gh`) is reflected before we check.
    git("fetch", "--quiet")
    if not git("branch", "-r", "--contains", sha):
        raise RuntimeError(f"HEAD {sha} (branch: {branch}) isn't on any remote branch. Run `git push` first.")

    if git("status", "--porcelain"):
        logger.warning(
            f"Local working tree is dirty; remote will run committed SHA {sha}, not your uncommitted changes."
        )

    logger.info(f"Using git ref {sha} (branch: {branch})")
    return sha


def _is_stockout(stderr: str) -> bool:
    return "ZONE_RESOURCE_POOL_EXHAUSTED" in stderr


def _is_phantom_already_exists(stderr: str) -> bool:
    # It seems like a 409 occurs when a prev VM from a diff zone in the loop is getting staged. I don't understand why
    # it's an error—the instances are in diff zones. Choosing to treat this as benign.
    return "HTTPError 409" in stderr and "already exists" in stderr


def _gce_create_cmd(
    config: GpuConfig,
    instance_name: str,
    zone: str,
    *,
    provision_type: Literal["FLEX_START", "STANDARD"],
    wait_for_instance_creation: bool,
    path_to_metadata_script: dict[str, str],
    launch_id: str | None = None,
    git_ref: str | None = None,
) -> list[str]:
    metadata = {
        "gcs-bucket": os.environ["GROUPING_TRAINER_BUCKET"],
        # Forward serial port output to Cloud Logging so we can post-mortem deleted VMs.
        "serial-port-logging-enable": "TRUE",
    }
    if launch_id is not None:
        metadata["launch-id"] = launch_id
    if git_ref is not None:
        metadata["git-ref"] = git_ref
    if config.install_nvidia_driver:
        metadata["enable-osconfig"] = "TRUE"
        metadata["install-nvidia-driver"] = "True"

    args = [
        "gcloud",
        "compute",
        "instances",
        "create",
        instance_name,
        f"--zone={zone}",
        f"--provisioning-model={provision_type}",
        f"--machine-type={config.machine_type}",
        "--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default",
        f"--metadata-from-file={','.join(f'{k}={v}' for k, v in path_to_metadata_script.items())}",
        f"--metadata={','.join(f'{k}={v}' for k, v in metadata.items())}",
        "--maintenance-policy=TERMINATE",
        "--instance-termination-action=DELETE",
        f"--max-run-duration={config.max_run_duration}",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        (
            f"--create-disk=auto-delete=yes,boot=yes,device-name={instance_name},"
            f"image={_IMAGE},mode=rw,size=200,type={config.boot_disk_type}"
        ),
        "--no-shielded-secure-boot",
        "--shielded-vtpm",
        "--shielded-integrity-monitoring",
        f"--reservation-affinity={config.reservation_affinity}",
    ]
    if provision_type == "FLEX_START":
        args.append(f"--request-valid-for-duration={_FLEX_START_REQUEST_VALID_FOR_DURATION}")
    if config.accelerator:
        args.append(f"--accelerator={config.accelerator}")
    if not wait_for_instance_creation:
        args.append("--async")
    return args


def _gce_multi_flex_start(
    *,
    config: GpuConfig,
    instance_name: str,
    num_zones: int,
    path_to_metadata_script: dict[str, str],
    git_ref: str | None = None,
) -> None:
    """
    Fan out async `FLEX_START` submits across the first `num_zones` of `config.zones`, all sharing the same `launch-id`
    metadata. First VM to boot claims a GCS atomic-create lock (see `bin/_startup.sh`). Losers self-delete.
    """
    launch_id = uuid.uuid4().hex[:12]
    zones = config.zones[:num_zones]
    logger.info(f"Multi-flex-start launch-id={launch_id} is fanning out to {len(zones)} zones: {zones}")
    n_submitted = 0
    last_stockout_stderr = ""

    for zone in zones:
        gce_create_args = _gce_create_cmd(
            config,
            instance_name,  # NOTE: instance names only need to be unique w/in (project, zone).
            zone,
            provision_type="FLEX_START",
            wait_for_instance_creation=False,
            path_to_metadata_script=path_to_metadata_script,
            launch_id=launch_id,
            git_ref=git_ref,
        )
        result = subprocess.run(gce_create_args, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Submitted flex-start for {instance_name} in zone {zone}")
            n_submitted += 1
            continue

        if _is_stockout(result.stderr):  # I've gotten stockouts on flex-starts before
            logger.warning(f"Stockout in {zone}. Continuing with remaining zones")
            last_stockout_stderr = result.stderr
            continue

        if _is_phantom_already_exists(result.stderr):
            logger.warning(f"409: already-exists on {zone}. Continuing with remaining zones")
            continue

        if n_submitted > 0:
            logger.warning(
                f"{n_submitted} instances already submitted will still race for the lock. "
                "You may still end up with a VM despite this error."
            )
        gt.utils.log_and_raise_subprocess(result, gce_create_args, log_prefix="gcloud failed ")

    if n_submitted == 0:
        suffix = f" Last stderr:\n{last_stockout_stderr}" if last_stockout_stderr else ""
        raise RuntimeError(f"Multi-flex-start: all {len(zones)} zones stocked out.{suffix}")
    logger.info(
        f"Multi-flex-start launch-id={launch_id} submitted {n_submitted}/{len(zones)} VMs. "
        f"First to boot w/in {_FLEX_START_REQUEST_VALID_FOR_DURATION} will stay running, others self-delete."
    )
    logger.info("Run bin/gtlist to list statuses, and bin/gtssh <name> <zone> to SSH into the one that's running.")


def _wandb_env_prefix() -> str:
    """
    Returns `WANDB_ENTITY=... WANDB_PROJECT=...` prefix to splice into a remote shell command, if those vars are set
    locally.
    """
    parts: list[str] = []
    for env_var in ("WANDB_ENTITY", "WANDB_PROJECT"):
        if value := os.environ.get(env_var):
            parts.append(f"{env_var}={shlex.quote(value)}")
    return " ".join(parts)


def _gce_instance_name(*, gpu: GpuType, job_type: JobType, name_suffix: str = "") -> str:
    # GCE instance names must match `[a-z]([-a-z0-9]*[a-z0-9])?`, so swap underscores for hyphens.
    name = f"{_INSTANCE_NAME_PREFIX}-{gpu}-{job_type}"
    if name_suffix:
        name += f"-{name_suffix.replace('_', '-')}"
    return name


def _delete_gce_instance_if_exists(instance_name: str) -> None:
    """Delete the instance(s) with this name, across all zones. No-op if none exist."""
    result = subprocess.run(
        ["gcloud", "compute", "instances", "list", f"--filter=name={instance_name}", "--format=value(zone)"],
        check=True,
        capture_output=True,
        text=True,
    )
    for zone_url in result.stdout.split():
        zone = zone_url.rsplit("/", 1)[-1]
        logger.info(f"Pre-deleting existing instance {instance_name} in zone {zone}")
        subprocess.run(
            ["gcloud", "compute", "instances", "delete", instance_name, f"--zone={zone}", "--quiet"],
            check=False,
        )


def gce_vm(
    *,
    gpu: GpuType,
    # TODO: support StrEnum-typed params in tap
    job_type: Literal[tuple(member.value for member in JobType)] = JobType.SSH,  # type: ignore[valid-type]
    name_suffix: str = "",
    sync_start: bool = False,
    multi_flex_start: bool = False,
    multi_flex_start_num_zones: int = 10,
    command: str | None = None,
    zone: str | None = None,
    num_cycles_through_zones: int = 5,
    seconds_between_gce_create_attempts: int = 1,
    delete_if_exists: bool = False,
    git_ref: str | None = None,
) -> None:
    """
    Launch a GCE instance for the given GPU type. If `command` is given, it's passed via instance metadata and
    `_startup.sh` `eval`s it after env setup. If not, the instance just sets up the env and stops for when you want to
    SSH in and iterate.

    The instance's name is `gt-{gpu}-{job_type}[-{name_suffix}]`. In `--multi_flex_start` mode, the name is the same
    across zones.

    Parameters
    ----------
    gpu
        The type of GPU to flex-start and run on.
    job_type
        Discriminator for what this VM is for (train/eval/save/synth/bench/ssh).
    name_suffix
        Optional suffix appended to the instance name for collision avoidance between concurrent launches. Empty
        (default) for the SSH-in-and-iterate CLI use case; programmatic callers should pass `run_shortname` or similar.
    sync_start
        If False (default), flex-starts the instance—GCP waits up to 2h to find one. `--sync_start` uses on-demand
        pricing and finds an instance in any zone, as flex-starting often can't find instances in time.
    multi_flex_start
        Fan out async FLEX_START submits across the first `multi_flex_start_num_zones` zones of `zones`. First VM to
        boot wins via a GCS atomic-create lock. Losers self-delete. Mutually exclusive with `sync_start`.
    multi_flex_start_num_zones
        How many zones to fan out to in multi-flex mode.
    command
        The command to run on the remote instance. If not given, the instance just sets up the env and stops for when
        you want to SSH in and iterate.
    zone
        Override the GCP zone. In FLEX_START mode pins the single submit zone; in STANDARD mode pins the loop to that
        zone (still retried `num_cycles_through_zones` times). Ignored in `--multi_flex_start` mode.
    num_cycles_through_zones
        No-op for sync_start. Otherwise, loop through zones this many times before giving up.
    seconds_between_gce_create_attempts
        No-op for sync_start. Otherwise, sleep this many seconds b/t consecutive zone attempts.
    delete_if_exists
        If True, delete any existing instance with this name (across all zones) before launching.
    git_ref
        Git ref (SHA or branch name) the remote should check out after cloning. When None (default), resolves to the
        local HEAD SHA, and requires that SHA to have been pushed to a remote branch. Pass `--git_ref main` (or any
        explicit ref) to skip the auto-resolve when you don't care which SHA the VM starts from, e.g. for a debug VM.
    """
    config = gpu_type_to_config[gpu]
    if multi_flex_start and sync_start:
        raise ValueError("--multi_flex_start and --sync_start are mutually exclusive")
    if multi_flex_start and config.default_zone_for_flex_start is None:
        raise ValueError(f"GPU type {gpu!r} does not support FLEX_START, so --multi_flex_start is not applicable")

    if git_ref is None:
        git_ref = _local_head_sha()
    else:
        ls_remote_args = ["git", "-C", _repo_root(), "ls-remote", "--exit-code", "origin", git_ref]
        ls_remote_result = subprocess.run(ls_remote_args, capture_output=True, text=True)
        if ls_remote_result.returncode != 0:
            gt.utils.log_and_raise_subprocess(
                ls_remote_result, ls_remote_args, log_prefix=f"git ref {git_ref!r} not found on origin "
            )

    instance_name = _gce_instance_name(gpu=gpu, job_type=job_type, name_suffix=name_suffix)
    if delete_if_exists:
        _delete_gce_instance_if_exists(instance_name)

    if (not sync_start) and (not multi_flex_start) and (config.default_zone_for_flex_start is not None):
        provision_type: Literal["FLEX_START", "STANDARD"] = "FLEX_START"
        wait_for_instance_creation = False
        zones_to_try: tuple[str, ...] = (zone or config.default_zone_for_flex_start,)
    else:
        provision_type = "STANDARD"
        wait_for_instance_creation = True
        zones_unique = (zone,) if zone else config.zones
        zones_to_try = zones_unique * num_cycles_through_zones

    # Use --metadata-from-file=command=<tempfile> instead of --metadata=command=<cmd> because gcloud splits --metadata
    # values on commas to find KEY=VAL pairs, so any comma in the cmd would break it. Accumulate these files in a dict:
    path_to_metadata_script = {"startup-script": f"{_repo_root()}/bin/_startup.sh"}

    with contextlib.ExitStack() as stack:
        if command:
            # Forward W&B config so the remote logs the same wandb URL and `wandb.init` pins to the same entity/project.
            # bin/_startup.sh only sets WANDB_API_KEY; entity/project come from the launcher's local env.
            if env_prefix := _wandb_env_prefix():
                command = f"{env_prefix} {command}"
            cmd_file = stack.enter_context(tempfile.NamedTemporaryFile(mode="w", suffix=".cmd", encoding="utf-8"))
            cmd_file.write(command)
            cmd_file.flush()
            path_to_metadata_script["command"] = cmd_file.name

        if multi_flex_start:
            _gce_multi_flex_start(
                config=config,
                instance_name=instance_name,
                num_zones=multi_flex_start_num_zones,
                path_to_metadata_script=path_to_metadata_script,
                git_ref=git_ref,
            )
            return

        for zone_idx, zone in enumerate(zones_to_try):
            # Attempt creation in this zone
            gce_create_args = _gce_create_cmd(
                config,
                instance_name,
                zone,
                provision_type=provision_type,
                wait_for_instance_creation=wait_for_instance_creation,
                path_to_metadata_script=path_to_metadata_script,
                git_ref=git_ref,
            )
            logger.info(f"Attempting to create {instance_name} in zone {zone}")
            gce_create_cmd_result = subprocess.run(gce_create_args, capture_output=True, text=True)
            if gce_create_cmd_result.returncode == 0:
                creation_type = "Flex-started" if provision_type == "FLEX_START" else "Created"
                logger.info(f"{creation_type} {instance_name} in zone {zone}")
                return

            # Retry next zone if stockout
            is_last_attempt = zone_idx == len(zones_to_try) - 1
            if _is_stockout(gce_create_cmd_result.stderr) and not is_last_attempt:
                logger.warning(f"Stockout in {zone}. Trying next zone in {seconds_between_gce_create_attempts}s")
                time.sleep(seconds_between_gce_create_attempts)
                continue

            # Fail
            if not sync_start:
                logger.warning("You may have success with --multi_flex_start if you're fine waiting, else --sync_start")
            gt.utils.log_and_raise_subprocess(gce_create_cmd_result, gce_create_args, log_prefix="gcloud failed ")


def _strip_flags_and_their_values(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in flags:
            skip = True
            continue
        if any(arg.startswith(f"{flag}=") for flag in flags):
            continue
        out.append(arg)
    return out


def _strip_bool_flags(argv: list[str], flags: tuple[str, ...]) -> list[str]:
    """
    Also strips negation forms like `--no_flag`.
    """
    flags_to_strip = set(flags) | {flag.replace("--", "--no_", 1) for flag in flags}
    return [arg for arg in argv if arg not in flags_to_strip]


def run_argv_remotely(
    *,
    gpu: GpuType,
    job_type: JobType,
    run_name: str,
    sync_start: bool = False,
    multi_flex_start: bool = False,
    zone: str | None = None,
) -> None:
    """
    Launch a remote GPU instance and re-run the current Python invocation on it::

        python sys.argv[0] <args>

    w/ remote-only flags stripped:
      - `--gpu ...`
      - `--zone ...`
      - `--sync_start`
      - `--multi_flex_start`.

    Invokes `accelerate launch` instead of `python` if the instance has multiple GPUs.

    Callers should `return` immediately after invoking this, as the remote instance runs the actual workload. Callers
    should also check `is_on_remote()` before deciding to launch, so a CUDA-detection misfire on the remote can't
    trigger a recursive launch. I can think about a better pattern later.
    """
    if is_on_remote():
        raise RuntimeError(
            f"run_argv_remotely() called while {_REMOTE_ENV_VAR} is set — likely a recursion bug. "
            "Callers should guard the launch with `not gt.launch.is_on_remote()`."
        )

    command_parts: list[str] = [
        f"{_REMOTE_ENV_VAR}=1",
        f"{_RUN_NAME_ENV_VAR}={shlex.quote(run_name)}",
    ]

    # Command
    if gpu_type_to_config[gpu].n_gpu > 1:
        command_parts.append("NCCL_NET=Socket LD_LIBRARY_PATH= accelerate launch --multi_gpu")
    else:
        command_parts.append("python")

    # Script path and args
    script_path = os.path.relpath(os.path.abspath(sys.argv[0]), _repo_root())
    argv_remote = _strip_flags_and_their_values(argv=sys.argv[1:], flags=("--gpu", "--zone"))
    argv_remote = _strip_bool_flags(argv=argv_remote, flags=("--sync_start", "--multi_flex_start"))
    args_for_remote = shlex.join(argv_remote)

    command_parts.append(script_path)
    command_parts.append(args_for_remote)

    # Pass as metadata to a flex-start instance
    command = " ".join(command_parts)
    logger.info(f"Launching {gpu} with remote command:\n{command}")
    gce_vm(
        gpu=gpu,
        job_type=job_type,
        name_suffix=shortname_from_run_name(run_name),
        sync_start=sync_start,
        multi_flex_start=multi_flex_start,
        command=command,
        zone=zone,
    )


if __name__ == "__main__":
    gt.logging.configure_logging(process_type="launch")
    tapify(gce_vm)
