import math
import warnings

from datetime import datetime
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import MultiDatasetBatchSamplers
import torch
import wandb

import grouping_trainer as gt

assert torch.cuda.is_available()

wandb.login()

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

RUN_SHORTNAME = "gte-no-synthetic-semi-easy-negatives"

SAMPLE_TRAIN: int | None = None if torch.cuda.is_available() else 30
SAMPLE_VAL: int | None = 8000 if torch.cuda.is_available() else 20

PER_DEVICE_TRAIN_BATCH_SIZE = 256 if torch.cuda.is_available() else 2
GRADIENT_ACCUMULATION_STEPS = 1
GRADIENT_CHECKPOINTING = not torch.cuda.is_available()  # turn off if model supports flash attn

PER_DEVICE_EVAL_BATCH_SIZE = 2
EVAL_STEPS = 300
PER_DEVICE_TOKEN_BUDGET = 8192 * 4  # increased for A100 80GB

OUTPUT_DIR = f"./{timestamp}-{RUN_SHORTNAME}-output"
GCS_DIR = f"gs://grouping-data/runs/{timestamp}-{RUN_SHORTNAME}"

wandb.init(project="grouping-trainer", name=f"{timestamp}-{RUN_SHORTNAME}")

assert (EVAL_STEPS % 5) == 0, "pls for sanity make it divisible by 5"

gt.utils._cuda_empty_cache()

# model_path = "issue_grouping_v1/embeddings"
# # model_path = "/Users/kdubey/projects/seer/models/issue_grouping_v1/embeddings"
# model = gt.utils.SentenceTransformer(
#     str(model_path),
#     trust_remote_code=True,
#     # model_kwargs=dict(
#     #     dtype=torch.bfloat16,
#     #     attn_implementation="sdpa",  # not possible for jina-ai :-(
#     # )
# )
# model.device

model = gt.utils.SentenceTransformer(
    "Alibaba-NLP/gte-modernbert-base",
    model_kwargs=dict(
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ),
)

assert "layernorm" in repr(model[0].auto_model).lower()
assert "batch" not in repr(model[0].auto_model).lower()

# Don't have batch norm. That could mess up stuff for the deduplication strategy.

_ = model.encode("test")

dataset_val = gt.train.df_to_dataset(gt.data.load_val_df(sample_size=SAMPLE_VAL))
len(dataset_val)

dataset_dict_train, frac_positive = gt.data.load_train_dataset_dict(
    sample_size=SAMPLE_TRAIN,
    min_dataset_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    paths=("final_csvs/train.csv",),
    # paths=("final_csvs/train-no-sentry.csv", "final_csvs/synthetic-semi-easy-negatives.csv"),
)
len(dataset_dict_train)

sum(dataset_dict_train.num_rows.values())

def init_bias(frac_positive: float):
    return math.log(frac_positive / (1 - frac_positive))


trainer = gt.train.Trainer(
    model=model,
    args=SentenceTransformerTrainingArguments(
        # These should prolly be unchanged
        output_dir=OUTPUT_DIR,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        num_train_epochs=1,
        # Save memory
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        #
        # Datalaoder
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        # Each iter, pick a project randomly, sample from it.
        # Next iter, pick another project randomly, sample from it, etc.
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        seed=42,  # passed to batch sampler
        #
        # Optimizer
        learning_rate=1e-4,
        learning_rate_mapping={
            # These are important to tune. Higher so that training doesn't get stuck. TODO: check
            r"^log_scale$": 2e-4,
            r"^bias$": 2e-4,
        },
        weight_decay=0.01,
        warmup_ratio=0.1,
        #
        # Eval
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        #
        # Logging
        logging_strategy="steps",
        logging_steps=EVAL_STEPS // 10,  # train loss alongside metrics table
        run_name=f"{timestamp}-{RUN_SHORTNAME}",
        report_to="wandb",
        #
        # Checkpointing
        save_strategy="steps",
        save_steps=EVAL_STEPS // 2,
        save_total_limit=2,
    ),
    #
    # Training
    loss=gt.train.SigmoidPairwiseLoss(
        model,
        bias_init=init_bias(frac_positive),
        log_of_scale_init=torch.tensor(5).log(),
        matryoshka_dims=[768, 512, 256, 128, 64],
        matryoshka_weights=[2, 1, 1, 0.5, 0.25],
        n_dims_per_step=2,
    ),
    data_collator=gt.train.DefaultDataCollator(tokenize_fn=model.tokenize),
    train_dataset=dataset_dict_train,
    shuffle_within_dataset=False,  # more cache hits in each forward
    per_device_token_budget=PER_DEVICE_TOKEN_BUDGET,
    #
    # Eval (val loss only; precision/recall eval runs async on a separate machine)
    eval_dataset=dataset_val,
    #
    # Callbacks
    callbacks=[gt.train.GCSCheckpointUploadCallback(gcs_dir=GCS_DIR)],
)

print(f"\nRun on L4: python eval/eval_poller.py --gcs-dir {GCS_DIR} --wandb-run-id {wandb.run.id}\n")

warnings.filterwarnings(
    "ignore",
    message=".*torch.utils.checkpoint: the use_reentrant parameter.*",
    category=UserWarning,
)

train_output = trainer.train()

trainer.save_model()
