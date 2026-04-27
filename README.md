# grouping-trainer

Training code for Sentry's AI grouping model.

Sampling and labeling is in [data-analysis](https://github.com/getsentry/data-analysis/tree/main/grouping/data).


## Set up locally

```bash
./bin/set_up_local.sh
```


## Usage

### Train

```bash
python train.py --gpu h100 --run_shortname my-run
```

Train and eval metrics are logged to https://wandb.ai/sentry-seer/grouping-trainer.

For DDP:

```bash
python train.py --gpu a100-ddp --run_shortname my-run
```

Finetuning [ModernBERT large](https://huggingface.co/lightonai/modernbert-embed-large) takes 11 hours w/ 1 A100 80
GB GPU, 7 hours w/ 1 H100, and 7 hours w/ 2 A100 80 GB GPUs w/ DDP.

You can use 4 A100 80 GB GPUs for DDP in the following zones:
- [us-central1-a](https://console.cloud.google.com/iam-admin/quotas?project=ml-ai-420606&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22NVIDIA%2520A100%252080GB%2520GPUs_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayName_22%257D_2C%257B_22k_22_3A_22Dimensions%2520%2528e.g.%2520location%2529_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22region_3Aus-central1_5C_22_22_2C_22i_22_3A_22displayDimensions_22%257D%255D%22,%22p%22:0,%22r%22:200)))
- [europe-west4-a](https://console.cloud.google.com/iam-admin/quotas?project=ml-ai-420606&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Dimensions%2520%2528e.g.%2520location%2529_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22region_3Aeurope-west4_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayDimensions_22%257D_2C%257B_22k_22_3A_22Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22NVIDIA%2520A100%252080GB%2520GPUs_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayName_22%257D%255D%22,%22p%22:0,%22r%22:200)))
- [us-east4-c](https://console.cloud.google.com/iam-admin/quotas?project=ml-ai-420606&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22NVIDIA%2520A100%252080GB%2520GPUs_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayName_22%257D_2C%257B_22k_22_3A_22_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22region_3Aus-east4_5C_22_22%257D%255D%22,%22p%22:0,%22r%22:200))).

You can use many H100s simultaneously as well. Takes 3 hours to train on 4 H100s. The DDP implementation blocks a bit
more than it needs to.


### Launch a GPU

If you'd like to launch a bare instance to ssh into:

```bash
python -m grouping_trainer.launch --gpu h100
```


### Eval

<details>
<summary>Launch a GPU to run a model on held-out projects</summary>

```bash
python eval/save_embeddings.py \
    --run_gcs_dir gs://grouping-data/runs/2026-04-10-12-39-45-large-no-prefix \
    --truncate_dims 64 128 256 512 768 \
    --use_compiled
```

</details>

<details>
<summary>Compare two models head-to-head on held-out projects</summary>

```bash
python eval/compare.py \
    --name_model1 v1 \
    --name_model2 large-no-prefix \
    --gcs_model1 gs://grouping-data/runs/issue_grouping_v1/similarities/test_full2 \
    --gcs_model2 gs://grouping-data/runs/2026-04-10-12-39-45-large-no-prefix/similarities/test_full2 \
    --threshold_model1 0.99 \
    --threshold_model2 0.90 \
    --dim_model2 64 \
    --overwrite
```

</details>
