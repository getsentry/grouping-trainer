#!/bin/bash
# Sets up the environment for grouping-trainer instances.
# Normally invoked as root via GCP instance startup.
# To step through manually after SSH'ing in, run
#   sudo -i
#first so $HOME=/root and paths line up.

set -euo pipefail

# GCP's metadata script runner doesn't export HOME
export HOME="${HOME:-/root}"

# Set by gt.launch.gce_vm
GROUPING_TRAINER_BUCKET=$(curl -fsS -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket)
export GROUPING_TRAINER_BUCKET


# ----------------------------------------------------------------------------------------------------------------------
# Set up python env
# ----------------------------------------------------------------------------------------------------------------------

# Install uv (manages its own Python, respects .python-version in the repo).
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR="/root/grouping-trainer"

WANDB_API_KEY=$(gcloud secrets versions access latest --secret=wandb-api-key)
export WANDB_API_KEY

git clone https://github.com/getsentry/grouping-trainer.git "$REPO_DIR"
cd "$REPO_DIR"

uv sync --locked
# shellcheck disable=SC1091
source .venv/bin/activate  # so the eval $COMMAND below finds python/accelerate


# ----------------------------------------------------------------------------------------------------------------------
# Download data
# ----------------------------------------------------------------------------------------------------------------------

gcloud storage cp -r "gs://${GROUPING_TRAINER_BUCKET}/final_csvs/" .


# ----------------------------------------------------------------------------------------------------------------------
# SSH niceties
# ----------------------------------------------------------------------------------------------------------------------

# Auto-cd into the repo, put uv on PATH, and activate the venv on `sudo -i`
{
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "cd $REPO_DIR && source .venv/bin/activate"
} >> /root/.bashrc

# logs cmd = shortcut for tailing the run log from any SSH session
cat > /usr/local/bin/logs <<'EOF'
#!/bin/bash
exec sudo tail -n 50 -f /var/log/grouping_trainer_run.log "$@"
EOF
chmod +x /usr/local/bin/logs

# screen -S run
# ctrl+a d
# ...
# screen -r

echo "Setup complete."


# ----------------------------------------------------------------------------------------------------------------------
# Run the command passed via instance metadata (set by gt.launch.gce_vm) and shutdown
# ----------------------------------------------------------------------------------------------------------------------

# No-op when the attribute is unset or the metadata server is unreachable.
COMMAND=$(curl -fsS -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/command 2>/dev/null || true)
if [ -n "$COMMAND" ]; then
    LOG_FILE="/var/log/grouping_trainer_run.log"
    echo "Running command, output -> $LOG_FILE"
    eval "$COMMAND" >>"$LOG_FILE" 2>&1 || true
    shutdown -h now
fi
