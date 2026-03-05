#!/bin/bash
# Shared startup script for training (H100) and eval (L4) machines.
# Passed via --metadata-from-file startup-script=bin/_startup.sh to gcloud compute instances create.
# Does environment setup only — does NOT run train.py or eval/eval_poller.py.
set -euo pipefail

# On GCP deep learning VMs, the first non-root user with a home dir is the login user.
USER_HOME=$(getent passwd 1000 | cut -d: -f6)
USERNAME=$(getent passwd 1000 | cut -d: -f1)
REPO_DIR="$USER_HOME/grouping-trainer"
MARKER="/tmp/setup-done"

GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token-grouping-trainer-temp --project=996102297610)
export GITHUB_TOKEN
WANDB_API_KEY=$(gcloud secrets versions access latest --secret=wandb-api-key --project=996102297610)
export WANDB_API_KEY

# Persist WANDB_API_KEY for the user's shell sessions
echo "export WANDB_API_KEY=$WANDB_API_KEY" >> "$USER_HOME/.bashrc"

git clone "https://${GITHUB_TOKEN}@github.com/getsentry/grouping-trainer.git" "$REPO_DIR"
chown -R "$USERNAME:$USERNAME" "$REPO_DIR"
gsutil -m cp -r gs://grouping-data/final_csvs "$REPO_DIR/"
chown -R "$USERNAME:$USERNAME" "$REPO_DIR/final_csvs"

# conda is pre-installed on the deep learning VM image
su - "$USERNAME" -c "
    source /opt/conda/etc/profile.d/conda.sh
    conda create -n gt-env python=3.10 -y
    conda activate gt-env
    cd $REPO_DIR
    pip install -e .
"

touch "$MARKER"
echo "Setup complete. Marker written to $MARKER"
