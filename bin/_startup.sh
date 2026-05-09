#!/bin/bash
# Sets up the environment for grouping-trainer instances.
# Normally invoked as root via GCP instance startup. To step through manually
# after SSH'ing in, run `sudo -i` first so $HOME=/root and paths line up.
set -euo pipefail

# Install uv (manages its own Python; respects .python-version in the repo).
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR="/root/grouping-trainer"

GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token-grouping-trainer-temp --project=996102297610)
export GITHUB_TOKEN
WANDB_API_KEY=$(gcloud secrets versions access latest --secret=wandb-api-key --project=996102297610)
export WANDB_API_KEY

git clone "https://${GITHUB_TOKEN}@github.com/getsentry/grouping-trainer.git" "$REPO_DIR"
cd "$REPO_DIR"

# Sometimes the Huggingface API hangs when we download the base model, so download from GCS:
mkdir -p lightonai/modernbert-embed-large
gcloud storage cp -r gs://grouping-data/base_models/lightonai/modernbert-embed-large/* lightonai/modernbert-embed-large

uv sync --locked

gcloud storage cp -r gs://grouping-data/final_csvs/ .

# Auto-cd into the repo, put uv on PATH, and activate the venv on `sudo -i`.
{
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "cd $REPO_DIR && source .venv/bin/activate"
} >> /root/.bashrc

# screen -S run
# ctrl+a d
# ...
# screen -r

echo "Setup complete."

# Run any command passed via instance metadata (set by gt.launch.flex).
# No-op when the attribute is unset or the metadata server is unreachable.
COMMAND=$(curl -fsS -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/command 2>/dev/null || true)
if [ -n "$COMMAND" ]; then
    LOG_FILE="/var/log/grouping_trainer_run.log"
    echo "Running command, output → $LOG_FILE"
    eval "$COMMAND" >>"$LOG_FILE" 2>&1 || true
    shutdown -h now
fi
# To follow the log:
# sudo tail -f /var/log/grouping_trainer_run.log
