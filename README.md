# grouping-trainer

Training code for Sentry's AI grouping model.

Sampling and labeling is in [data-analysis](https://github.com/getsentry/data-analysis/tree/main/grouping/data).


## Usage

```bash
./bin/flex_a100.sh
```

Finetuning [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) w/
sdpa takes ~5 hours w/ 1 A100 80 GB.

For DDP:

```bash
./bin/flex_a100_ddp.sh
```

Our quota in
[us-central1](https://console.cloud.google.com/iam-admin/quotas?project=ml-ai-420606&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22NVIDIA%2520A100%252080GB%2520GPUs_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayName_22%257D_2C%257B_22k_22_3A_22Dimensions%2520%2528e.g.%2520location%2529_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22region_3Aus-central1_5C_22_22_2C_22i_22_3A_22displayDimensions_22%257D%255D%22,%22p%22:0,%22r%22:200)))
and
[europe-west4](https://console.cloud.google.com/iam-admin/quotas?project=ml-ai-420606&pageState=(%22allQuotasTable%22:(%22f%22:%22%255B%257B_22k_22_3A_22Dimensions%2520%2528e.g.%2520location%2529_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22region_3Aeurope-west4_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayDimensions_22%257D_2C%257B_22k_22_3A_22Name_22_2C_22t_22_3A10_2C_22v_22_3A_22_5C_22NVIDIA%2520A100%252080GB%2520GPUs_5C_22_22_2C_22s_22_3Atrue_2C_22i_22_3A_22displayName_22%257D%255D%22,%22p%22:0,%22r%22:200)))
is 4.


## Set up locally

```bash
./bin/set_up_local.sh
```
