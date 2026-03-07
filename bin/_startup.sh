#!/bin/bash
# Shared startup script for training (H100) and eval (L4) machines.
# Passed via --metadata-from-file startup-script=bin/_startup.sh to gcloud compute instances create.
# Does environment setup only — does NOT run train.py or eval/eval_poller.py.
set -euo pipefail

REPO_DIR="/root/grouping-trainer"

GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token-grouping-trainer-temp --project=996102297610)
export GITHUB_TOKEN
WANDB_API_KEY=$(gcloud secrets versions access latest --secret=wandb-api-key --project=996102297610)
export WANDB_API_KEY

git clone "https://${GITHUB_TOKEN}@github.com/getsentry/grouping-trainer.git" "$REPO_DIR"
gcloud storage cp -r gs://grouping-data/final_csvs "$REPO_DIR/"

# conda is pre-installed on the deep learning VM image at /opt/conda
source /opt/conda/etc/profile.d/conda.sh
conda create -n gt-env python=3.10 -y
conda activate gt-env
cd "$REPO_DIR"
pip install -e .

echo "Setup complete."
