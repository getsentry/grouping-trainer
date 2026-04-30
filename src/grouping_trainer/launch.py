"""
Helpers for launching remote GCE instances that run a Python entry-point.

The instance's startup script (`bin/_startup.sh`) reads the command from instance metadata
and `eval`s it after the env is set up.

CLI:
    python -m grouping_trainer.launch --gpu <gpu> [--zone ZONE] [--command COMMAND]
"""

import logging
import os
import shlex
import subprocess
import sys
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict
from tap import tapify

logger = logging.getLogger(__name__)

GpuType = Literal["l4", "h100", "h100-ddp", "a100", "a100-ddp"]

_REMOTE_ENV_VAR = "GROUPING_TRAINER_REMOTE"
"""
Env var set by `remote()` in the cmd that runs on the remote instance, so the remote knows it shouldn't try to
re-launch. Callers can check `is_on_remote()` before deciding to launch.
"""

_IMAGE = "projects/ml-images/global/images/pytorch-2-7-cu128-ubuntu-2404-nvidia-570-v20260323"


def is_on_remote() -> bool:
    """True when running inside an instance that was launched via `remote()`."""
    return bool(os.environ.get(_REMOTE_ENV_VAR))


class GpuConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    zone: str  # default zone; flex-start capacity varies across regions, so this
    # gets overridden via the --zone flag when the default is dry.
    machine_type: str
    accelerator: str | None  # None for *-ddp variants — accelerators are built
    # into the machine type, so passing --accelerator is redundant/erroneous.
    max_run: str
    install_nvidia_driver: bool
    reservation_affinity: Literal["none", "any"]
    wait: bool  # whether to block locally on instance creation. False adds --async.


gpu_type_to_config: dict[GpuType, GpuConfig] = {
    "l4": GpuConfig(
        name="grouping-trainer-l4-eval",
        zone="us-central1-a",
        machine_type="g2-standard-4",
        accelerator="count=1,type=nvidia-l4",
        max_run="86400s",
        install_nvidia_driver=False,
        reservation_affinity="any",
        wait=True,  # L4s come up fast; block so errors surface promptly
    ),
    "h100": GpuConfig(
        name="grouping-trainer-h100",
        zone="us-east4-c",
        machine_type="a3-highgpu-1g",
        accelerator="count=1,type=nvidia-h100-80gb",
        max_run="86400s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        wait=False,  # flex-start can queue for up to 1h; don't block the shell
    ),
    "h100-ddp": GpuConfig(
        name="grouping-trainer-h100-ddp",
        zone="us-central1-a",
        machine_type="a3-highgpu-2g",
        accelerator=None,
        max_run="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        wait=False,
    ),
    "a100": GpuConfig(
        name="grouping-trainer-a100",
        zone="us-central1-a",
        machine_type="a2-ultragpu-1g",
        accelerator="count=1,type=nvidia-a100-80gb",
        max_run="86400s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        wait=False,
    ),
    "a100-ddp": GpuConfig(
        name="grouping-trainer-a100-ddp",
        zone="us-east4-c",
        machine_type="a2-ultragpu-2g",
        accelerator=None,
        max_run="172800s",
        install_nvidia_driver=True,
        reservation_affinity="none",
        wait=False,
    ),
}


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _strip_flags(argv: list[str], flags: tuple[str, ...]) -> list[str]:
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


def flex(gpu: GpuType, command: str | None = None, zone: str | None = None) -> None:
    """
    Flex-start a GCE instance. If `command` is given, it's passed via instance metadata
    and `_startup.sh` `eval`s it after env setup. If not, the instance just sets up
    the env and stops — useful for ssh-ing in and iterating.

    Parameters
    ----------
    gpu
        One of l4, h100, h100-ddp, a100, a100-ddp.
    command
        Shell command to run on the instance after env setup.
    zone
        Override the default zone (use this when flex-start capacity is dry).
    """
    config = gpu_type_to_config[gpu]

    # Both startup-script and command must go in the SAME --metadata-from-file
    # flag (comma-separated): passing the flag twice makes the second
    # invocation overwrite the first, silently dropping the startup script.
    # And we use --metadata-from-file=command=<tempfile> rather than
    # --metadata=command=<cmd> because gcloud splits --metadata values on
    # commas to find KEY=VAL pairs, so any comma in the cmd would break it.
    metadata_files = {"startup-script": f"{_repo_root()}/bin/_startup.sh"}
    cmd_file: tempfile._TemporaryFileWrapper | None = None
    if command:
        cmd_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".cmd", encoding="utf-8")
        cmd_file.write(command)
        cmd_file.close()
        metadata_files["command"] = cmd_file.name

    args = [
        "gcloud",
        "compute",
        "instances",
        "create",
        config.name,
        "--project=ml-ai-420606",
        f"--zone={zone or config.zone}",
        f"--machine-type={config.machine_type}",
        "--network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default",
        f"--metadata-from-file={','.join(f'{k}={v}' for k, v in metadata_files.items())}",
        "--maintenance-policy=TERMINATE",
        "--provisioning-model=FLEX_START",
        "--request-valid-for-duration=1h",
        "--instance-termination-action=DELETE",
        f"--max-run-duration={config.max_run}",
        "--service-account=996102297610-compute@developer.gserviceaccount.com",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        (
            f"--create-disk=auto-delete=yes,boot=yes,device-name={config.name},"
            f"image={_IMAGE},mode=rw,size=200,type=pd-balanced"
        ),
        "--no-shielded-secure-boot",
        "--shielded-vtpm",
        "--shielded-integrity-monitoring",
        f"--reservation-affinity={config.reservation_affinity}",
    ]
    if config.install_nvidia_driver:
        args.append("--metadata=enable-osconfig=TRUE,install-nvidia-driver=True")
    if config.accelerator:
        args.append(f"--accelerator={config.accelerator}")
    if not config.wait:
        args.append("--async")

    try:
        logger.info(f"Creating {config.name} in {zone or config.zone}")
        subprocess.run(args, check=True)
    finally:
        if cmd_file is not None:
            os.unlink(cmd_file.name)


def l4_eval(eval_command: str) -> None:
    """
    Create the L4 eval instance, passing the eval command via instance metadata
    (read by _startup.sh after setup completes).
    """
    flex("l4", command=eval_command)
    logger.info("Created l4-eval instance with eval poller in startup script")


def remote(
    gpu: GpuType,
    ddp: bool = False,
    zone: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    """
    Launch a remote GPU instance and re-run the current Python invocation on it
    (i.e. `python sys.argv[0] <args>`, with `--gpu` and `--zone` stripped).

    ddp: prefix the command with NCCL env vars and use `accelerate launch --multi_gpu`.
    zone: override the default GCP zone for this gpu type.
    extra_env: additional env vars to set on the remote command (e.g. forwarding a
        run_name generated locally so the remote re-uses it instead of generating
        its own).

    Caller should `return` immediately after invoking this — the remote instance
    runs the actual workload. Callers should also check `is_on_remote()` before
    deciding to launch, so a CUDA-detection misfire on the remote can't trigger
    a recursive launch.
    """
    if is_on_remote():
        raise RuntimeError(
            f"remote() called while {_REMOTE_ENV_VAR} is set — likely a recursion bug. "
            "Callers should guard the launch with `not gt.launch.is_on_remote()`."
        )

    repo_root = _repo_root()
    argv_remote = _strip_flags(sys.argv[1:], ("--gpu", "--zone"))
    script_path = os.path.relpath(os.path.abspath(sys.argv[0]), repo_root)
    env_pairs = [f"{_REMOTE_ENV_VAR}=1"]
    for key, value in (extra_env or {}).items():
        env_pairs.append(f"{key}={shlex.quote(value)}")
    env_prefix = " ".join(env_pairs)
    if ddp:
        command = (
            f"{env_prefix} NCCL_NET=Socket LD_LIBRARY_PATH= "
            f"accelerate launch --multi_gpu {script_path} {shlex.join(argv_remote)}"
        )
    else:
        command = f"{env_prefix} python {script_path} {shlex.join(argv_remote)}"
    logger.info(f"Launching {gpu} with remote command:\n  {command}")
    flex(gpu, command=command, zone=zone)


if __name__ == "__main__":
    tapify(flex)
