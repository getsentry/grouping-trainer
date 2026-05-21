#!/bin/bash
# Sets up the environment for grouping-trainer instances.
# Normally invoked as root via GCP instance startup.
# To step through manually after SSH'ing in, run
#   sudo -i
# first so $HOME=/root and paths line up.

set -euo pipefail

# GCP's metadata script runner doesn't export HOME
export HOME="${HOME:-/root}"

# Set by gt.launch.gce_vm
GROUPING_TRAINER_BUCKET=$(curl -fsS -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket)
export GROUPING_TRAINER_BUCKET


# ----------------------------------------------------------------------------------------------------------------------
# Multi-flex-start lock race
# ----------------------------------------------------------------------------------------------------------------------
# When gt.launch fans out flex-start submits across multiple zones (--multi_flex_start), every sibling VM gets the same
# `launch-id` metadata. The first to reach this point claims the GCS object atomically via --if-generation-match=0;
# losers see a 412 and self-delete.
#
# Any non-zero `gcloud storage cp` (412 race-loss or a transient gcloud error) self-deletes. Prefer over-deleting (user
# needs to retry the launch) to under-deleting (bunch of colliding work).
LAUNCH_ID=$(curl -fsS -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/launch-id 2>/dev/null || true)
if [ -n "$LAUNCH_ID" ]; then
    LOCK_PATH="gs://${GROUPING_TRAINER_BUCKET}/launches/${LAUNCH_ID}/winner"
    LOCK_TMP=$(mktemp)
    hostname > "$LOCK_TMP"
    if gcloud storage cp "$LOCK_TMP" "$LOCK_PATH" --if-generation-match=0; then
        echo "Congratulations! You won the race to $LOCK_PATH"
    else
        echo "You lost the race to $LOCK_PATH. Self-deleting. Better luck next time"
        # || true so a metadata-server hiccup doesn't trip `set -e` before we get to `gcloud compute instances delete`.
        NAME=$(curl -fsS -H 'Metadata-Flavor: Google' \
            http://metadata.google.internal/computeMetadata/v1/instance/name) || NAME=$(hostname)
        ZONE=$(curl -fsS -H 'Metadata-Flavor: Google' \
            http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null \
            | awk -F/ '{print $NF}') || ZONE=""
        gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
        exit 0
    fi
fi


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

# Auto-cd into the repo, put uv on PATH, activate the venv, and re-export the
# env vars set above so interactive SSH sessions see them.
{
    echo "export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo "export GROUPING_TRAINER_BUCKET=$GROUPING_TRAINER_BUCKET"
    echo "export WANDB_API_KEY=$WANDB_API_KEY"
    echo "cd $REPO_DIR && source .venv/bin/activate"
} >> /root/.bashrc

# logs cmd = shortcut for tailing the run log from any SSH session
cat > /usr/local/bin/logs <<'EOF'
#!/bin/bash
exec sudo tail -n 50 -f /var/log/grouping_trainer_run.log "$@"
EOF
chmod +x /usr/local/bin/logs

# startup-logs cmd = shortcut for viewing the GCE startup script's output (e.g.,
# when /var/log/grouping_trainer_run.log doesn't exist because startup failed).
cat > /usr/local/bin/startup-logs <<'EOF'
#!/bin/bash
exec sudo journalctl -u google-startup-scripts.service --no-pager "$@"
EOF
chmod +x /usr/local/bin/startup-logs

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
