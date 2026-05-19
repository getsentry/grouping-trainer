"""
Helpers for launching remote GCE instances that run a Python entry-point.

The instance's startup script does bin/_startup.sh to set up the python env and then `eval`s an inputted command.

Training jobs don't need to start in time, so by default they're launched async via flex-start w/ a max wait time of 1
hour. Also saves some money.

Eval jobs (on cheap L4 GPUs) are launched by sync-looping through zones ourselves b/c eval ideally does start in time,
e.g., training shouldn't start w/o an eval poller. L4s are cheap-enough that the flex-start discount isn't worth the
flex-start headache. In current year, GCE doesn't have a simple, built-in cross-region fallback mechanism. Our jobs have
no networking or region dependence. They communicate via GCS if at all.
"""

import contextlib
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict
from tap import tapify

logger = logging.getLogger(__name__)

_REMOTE_ENV_VAR = "GROUPING_TRAINER_REMOTE"
"""
Env var set by `run_argv_remotely()` in the cmd that runs on the remote instance, so the remote knows it shouldn't try
to re-launch. Callers can check `is_on_remote()` before deciding to launch.
"""

_IMAGE = "projects/ml-images/global/images/pytorch-2-7-cu128-ubuntu-2404-nvidia-570-v20260323"

_INSTANCE_NAME_PREFIX = "grouping-trainer"


class JobType(StrEnum):
    TRAIN = "train"
    EVAL = "eval"
    SAVE = "save"
    SYNTH = "synth"
    BENCHMARK = "bench"
    SSH = "ssh"


def is_on_remote() -> bool:
    return bool(os.environ.get(_REMOTE_ENV_VAR))


def run_name_from_shortname(shortname: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{timestamp}-{shortname}"


def shortname_from_run_name(run_name: str) -> str:
    """
    Extract the shortname from a run_name formatted as 'YYYY-MM-DD-HH-MM-SS-<shortname>'.
    Returns `run_name` unchanged if it isn't in that format.
    """
    parts = run_name.split("-", maxsplit=6)
    return parts[-1] if len(parts) == 7 else run_name


def validate_run_gcs_dir(run_gcs_dir: str) -> None:
    path_inference = f"{run_gcs_dir.rstrip('/')}/inference/"
    subprocess.run(["gcloud", "storage", "ls", path_inference], check=True, stdout=subprocess.DEVNULL)


class GpuConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    flex_start_zone: str | None
    """Single zone for FLEX_START launches (DWS queues the request for up to 1h). None disables FLEX_START (L4)."""
    standard_zones: tuple[str, ...]
    """Zones to try in order for STANDARD launches (multi-zone fail-fast fallback on stockout)."""
    machine_type: str
    accelerator: str | None  # None for *-ddp variants b/c accelerators are built into the machine type
    max_run_duration: str
    install_nvidia_driver: bool
    reservation_affinity: Literal["none", "any"]
    is_for_training: bool
    n_gpu: int


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
A100_ZONES = (
    "us-central1-a",
    "us-east4-c",
    "europe-west4-a",
)

GpuType = Literal[
    "l4",
    "h100",
    "h100-ddp-2",
    "h100-ddp-4",
    "a100",
    "a100-ddp-2",
    "a100-ddp-4",
]

gpu_type_to_config: dict[GpuType, GpuConfig] = {
    "l4": GpuConfig(
        flex_start_zone=None,
        # gcloud compute accelerator-types list --filter="name=nvidia-l4" --format='value(zone)'
        standard_zones=(
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
        ),
        machine_type="g2-standard-4",
        accelerator="count=1,type=nvidia-l4",
        max_run_duration="86400s",
        install_nvidia_driver=False,
        reservation_affinity="any",
        is_for_training=False,
        n_gpu=1,
    ),
    "h100": GpuConfig(
        flex_start_zone="us-central1-a",
        standard_zones=H100_ZONES,
        machine_type="a3-highgpu-1g",
        accelerator="count=1,type=nvidia-h100-80gb",
        max_run_duration="86400s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=1,
    ),
    "h100-ddp-2": GpuConfig(
        flex_start_zone="us-central1-a",
        standard_zones=H100_ZONES,
        machine_type="a3-highgpu-2g",
        accelerator=None,
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=2,
    ),
    "h100-ddp-4": GpuConfig(
        flex_start_zone="us-central1-a",
        standard_zones=H100_ZONES,
        machine_type="a3-highgpu-4g",
        accelerator=None,
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=4,
    ),
    #
    # Our quota allows 4 A100 80 GB GPUs for DDP in us-central1-a, us-east4-c, europe-west4-a
    "a100": GpuConfig(
        flex_start_zone="us-central1-a",
        standard_zones=A100_ZONES,
        machine_type="a2-ultragpu-1g",
        accelerator="count=1,type=nvidia-a100-80gb",
        max_run_duration="86400s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=1,
    ),
    "a100-ddp-2": GpuConfig(
        flex_start_zone="us-central1-a",
        standard_zones=A100_ZONES,
        machine_type="a2-ultragpu-2g",
        accelerator=None,
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=2,
    ),
    "a100-ddp-4": GpuConfig(
        flex_start_zone="us-central1-a",
        standard_zones=A100_ZONES,
        machine_type="a2-ultragpu-4g",
        accelerator=None,
        max_run_duration="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        is_for_training=True,
        n_gpu=4,
    ),
}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


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
    """Strip boolean flags (which take no following value) from argv. Eats both `--flag` and `--no_flag`."""
    flags_to_strip = set(flags) | {flag.replace("--", "--no_", 1) for flag in flags}
    return [arg for arg in argv if arg not in flags_to_strip]


def _is_stockout(stderr: str) -> bool:
    return "ZONE_RESOURCE_POOL_EXHAUSTED" in stderr


def _gce_create_cmd(
    config: GpuConfig,
    instance_name: str,
    zone: str,
    *,
    provisioning_model: Literal["FLEX_START", "STANDARD"],
    wait_for_instance_creation: bool,
    path_to_metadata_script: dict[str, str],
) -> list[str]:
    metadata = {"gcs-bucket": os.environ["GROUPING_TRAINER_BUCKET"]}
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
        f"--provisioning-model={provisioning_model}",
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
            f"image={_IMAGE},mode=rw,size=200,type=pd-balanced"
        ),
        "--no-shielded-secure-boot",
        "--shielded-vtpm",
        "--shielded-integrity-monitoring",
        f"--reservation-affinity={config.reservation_affinity}",
    ]
    if provisioning_model == "FLEX_START":
        args.append("--request-valid-for-duration=1h")
    if config.accelerator:
        args.append(f"--accelerator={config.accelerator}")
    if not wait_for_instance_creation:
        args.append("--async")
    return args


def gce_vm(
    *,
    gpu: GpuType,
    # TODO: support StrEnum-typed params in tap
    job_type: Literal[tuple(member.value for member in JobType)] = JobType.SSH,  # type: ignore[valid-type]
    name_suffix: str = "",
    sync_start: bool = False,
    command: str | None = None,
    zone: str | None = None,
    num_cycles_through_zones: int = 5,
    seconds_between_gce_create_attempts: int = 1,
) -> None:
    """
    Launch a GCE instance for the given GPU type. If `command` is given, it's passed via instance metadata and
    `_startup.sh` `eval`s it after env setup. If not, the instance just sets up the env and stops for when you want to
    SSH in and iterate.

    The instance's name is `grouping-trainer-{gpu}-{job_type}[-{name_suffix}]` (the suffix is dropped when empty).

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
        If False (default), flex-starts the instance—GCP waits up to 1h to find one. `--sync_start` uses on-demand
        pricing and finds an instance in any zone, as flex-starting often can't find instances in time.
    command
        The command to run on the remote instance. If not given, the instance just sets up the env and stops for when
        you want to SSH in and iterate.
    zone
        Override the GCP zone. In FLEX_START mode pins the single submit zone; in STANDARD mode pins the loop to that
        zone (still retried `num_cycles_through_zones` times).
    num_cycles_through_zones
        No-op for sync_start. Otherwise, loop through zones this many times before giving up.
    seconds_between_gce_create_attempts
        No-op for sync_start. Otherwise, sleep this many seconds b/t consecutive zone attempts.
    """
    config = gpu_type_to_config[gpu]
    instance_name = f"{_INSTANCE_NAME_PREFIX}-{gpu}-{job_type}"
    if name_suffix:
        # GCE instance names must match `[a-z]([-a-z0-9]*[a-z0-9])?`, so swap underscores for hyphens.
        instance_name += f"-{name_suffix.replace('_', '-')}"

    if (not sync_start) and (config.flex_start_zone is not None):
        provisioning_model: Literal["FLEX_START", "STANDARD"] = "FLEX_START"
        wait_for_instance_creation = False
        zones_to_try: tuple[str, ...] = (zone or config.flex_start_zone,)
    else:
        provisioning_model = "STANDARD"
        wait_for_instance_creation = True
        zones_unique = (zone,) if zone else config.standard_zones
        zones_to_try = zones_unique * num_cycles_through_zones

    # Use --metadata-from-file=command=<tempfile> instead of --metadata=command=<cmd> because gcloud splits --metadata
    # values on commas to find KEY=VAL pairs, so any comma in the cmd would break it. Accumulate these files in a dict:
    path_to_metadata_script = {"startup-script": f"{_repo_root()}/bin/_startup.sh"}

    with contextlib.ExitStack() as stack:
        if command:
            cmd_file = stack.enter_context(tempfile.NamedTemporaryFile(mode="w", suffix=".cmd", encoding="utf-8"))
            cmd_file.write(command)
            cmd_file.flush()
            path_to_metadata_script["command"] = cmd_file.name

        for zone_idx, zone in enumerate(zones_to_try):
            # Attempt creation in this zone
            gce_create_args = _gce_create_cmd(
                config,
                instance_name,
                zone,
                provisioning_model=provisioning_model,
                wait_for_instance_creation=wait_for_instance_creation,
                path_to_metadata_script=path_to_metadata_script,
            )
            logger.info(f"Attempting to create {instance_name} in zone {zone}")
            gce_create_cmd_result = subprocess.run(gce_create_args, capture_output=True, text=True)
            if gce_create_cmd_result.returncode == 0:
                logger.info(f"Created {instance_name} in zone {zone}")
                return

            # Retry next zone if stockout
            is_last_attempt = zone_idx == len(zones_to_try) - 1
            if _is_stockout(gce_create_cmd_result.stderr) and not is_last_attempt:
                logger.warning(f"Stockout in {zone}. Trying next zone in {seconds_between_gce_create_attempts}s")
                time.sleep(seconds_between_gce_create_attempts)
                continue

            # Fail
            logger.error(f"gcloud failed (exit {gce_create_cmd_result.returncode}):\n{gce_create_cmd_result.stderr}")
            # CalledProcessError's repr drops stderr, so log it
            raise subprocess.CalledProcessError(
                gce_create_cmd_result.returncode,
                gce_create_args,
                output=gce_create_cmd_result.stdout,
                stderr=gce_create_cmd_result.stderr,
            )


def run_argv_remotely(
    *,
    gpu: GpuType,
    job_type: JobType,
    name_suffix: str,
    sync_start: bool = False,
    zone: str | None = None,
    env_var_to_value: dict[str, str] | None = None,
) -> None:
    """
    Launch a remote GPU instance and re-run the current Python invocation on it::

        python sys.argv[0] <args>

    w/ `--gpu` and `--zone` stripped. Invokes via `accelerate launch` if multi-GPU, otherwise `python`.

    Caller should `return` immediately after invoking this — the remote instance runs the actual workload. Callers
    should also check `is_on_remote()` before deciding to launch, so a CUDA-detection misfire on the remote can't
    trigger a recursive launch.

    `job_type` and `name_suffix` are forwarded to `gce_vm` to disambiguate the instance name.

    `env_var_to_value` are additional env vars to set on the remote command.
    """
    if is_on_remote():
        raise RuntimeError(
            f"run_argv_remotely() called while {_REMOTE_ENV_VAR} is set — likely a recursion bug. "
            "Callers should guard the launch with `not gt.launch.is_on_remote()`."
        )

    command_parts: list[str] = []

    # Env vars
    command_parts.append(f"{_REMOTE_ENV_VAR}=1")
    for env_var, value in (env_var_to_value or {}).items():
        command_parts.append(f"{env_var}={shlex.quote(value)}")

    # Command
    if gpu_type_to_config[gpu].n_gpu > 1:
        command_parts.append("NCCL_NET=Socket LD_LIBRARY_PATH= accelerate launch --multi_gpu")
    else:
        command_parts.append("python")

    # Script path and args
    script_path = os.path.relpath(os.path.abspath(sys.argv[0]), _repo_root())
    argv_remote = _strip_flags_and_their_values(argv=sys.argv[1:], flags=("--gpu", "--zone"))
    argv_remote = _strip_bool_flags(argv=argv_remote, flags=("--sync_start",))
    args_for_remote = shlex.join(argv_remote)

    command_parts.append(script_path)
    command_parts.append(args_for_remote)

    # Pass as metadata to a flex-start instance
    command = " ".join(command_parts)
    logger.info(f"Launching {gpu} with remote command:\n  {command}")
    gce_vm(
        gpu=gpu,
        job_type=job_type,
        name_suffix=name_suffix,
        sync_start=sync_start,
        command=command,
        zone=zone,
    )


if __name__ == "__main__":
    tapify(gce_vm)
