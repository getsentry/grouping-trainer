# grouping-trainer

Training code for Sentry's AI grouping model.

(Sampling and labeling is still in [data-analysis](https://github.com/getsentry/data-analysis/tree/main/grouping/data). Will migrate that over here.)


## Usage

```bash
gcloud compute instances create h100-flex --project=ml-ai-420606 --zone=us-central1-a --machine-type=a3-highgpu-1g --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default --metadata=enable-osconfig=TRUE --maintenance-policy=TERMINATE --provisioning-model=FLEX_START --instance-termination-action=DELETE --max-run-duration=172800s --service-account=996102297610-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --accelerator=count=1,type=nvidia-h100-80gb --create-disk=auto-delete=yes,boot=yes,device-name=h100-flex,image=projects/ml-images/global/images/c0-deeplearning-common-cu124-v20250325-debian-11-py310-conda,mode=rw,size=200,type=pd-balanced --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --labels=goog-ops-agent-policy=v2-x86-template-1-4-0,goog-ec-src=vm_add-gcloud --reservation-affinity=none && printf 'agentsRule:\n  packageState: installed\n  version: latest\ninstanceFilter:\n  inclusionLabels:\n  - labels:\n      goog-ops-agent-policy: v2-x86-template-1-4-0\n' > config.yaml && gcloud compute instances ops-agents policies create goog-ops-agent-v2-x86-template-1-4-0-us-central1-a --project=ml-ai-420606 --zone=us-central1-a --file=config.yaml && gcloud compute resource-policies create snapshot-schedule default-schedule-1 --project=ml-ai-420606 --region=us-central1 --max-retention-days=14 --on-source-disk-delete=keep-auto-snapshots --daily-schedule --start-time=20:00 && gcloud compute disks add-resource-policies h100-flex --project=ml-ai-420606 --zone=us-central1-a --resource-policies=projects/ml-ai-420606/regions/us-central1/resourcePolicies/default-schedule-1
```

```bash
export GITHUB_TOKEN=$(gcloud secrets versions access latest --secret=github-token-grouping-trainer-temp --project=996102297610)
export WANDB_API_KEY=$(gcloud secrets versions access latest --secret=wandb-api-key --project=996102297610)
```

```bash
git clone https://${GITHUB_TOKEN}@github.com/getsentry/grouping-trainer.git
```

```bash
gsutil -m -o GSUtil:check_hashes=never cp -r gs://grouping-data/final_csvs .
```

Make a venv

```bash
conda deactivate

conda create -n gt-env python=3.10 -y

conda activate gt-env

pip install -e .
```

Run using `screen`

```bash
gsutil -m cp -r wandb gs://grouping-data/runs/{OUTPUT_DIR}
```

```bash
gsutil -m rsync -r {OUTPUT_DIR} gs://grouping-data/runs/{OUTPUT_DIR}/training
```

```bash
gsutil -m cp -r train.py gs://grouping-data/runs/{OUTPUT_DIR}
```

TODO: enable DDP training. Finetuning [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) w/
sdpa takes ~5 hours w/ 1 A100 80 GB. Our quota in
[us-central1](https://console.cloud.google.com/iam-admin/quotas?project=ml-ai-420606&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22NVIDIA%2520A100%252080GB%2520GPUs_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayName_22%257D_2C%257B_22k_22_3A_22Dimensions%2520%2528e.g.%2520location%2529_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22region_3Aus-central1_5C_22_22_2C_22i_22_3A_22displayDimensions_22%257D%255D%22,%22p%22:0,%22r%22:200)))
and
[europe-west4](https://console.cloud.google.com/iam-admin/quotas?project=ml-ai-420606&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Dimensions%2520%2528e.g.%2520location%2529_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22region_3Aeurope-west4_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayDimensions_22%257D_2C%257B_22k_22_3A_22Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22NVIDIA%2520A100%252080GB%2520GPUs_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayName_22%257D%255D%22,%22p%22:0,%22r%22:200)))
is 4.


## Set up locally

```
direnv allow
```

```bash
python3.13 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
python -m pip install -e ".[dev]"
```
