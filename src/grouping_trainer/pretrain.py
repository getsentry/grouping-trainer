"""
Continue MLM pretraining on full grouping labels: `prompt[SEP]thinking_output[SEP]response_output`
"""

import gc
import logging
from datetime import datetime
from typing import Any

import polars as pl
import torch
from datasets import Dataset
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Sampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def load_texts(
    sep_token: str,
    paths: tuple[str, ...] = gt.data.DEFAULT_TRAIN_PATHS_NO_SYNTHETIC,
    sample_size: int | None = None,
    n_rows_per_csv: int | None = None,
) -> list[str]:
    """
    Returns a list of these texts:

    ```
    prompt[SEP]thinking_output[SEP]response_output
    ```
    """
    df = gt.data.load_train_df(paths=paths, sample_size=sample_size, n_rows_per_csv=n_rows_per_csv)

    columns = ("prompt", "thinking_output", "response_output")
    bad = df.filter(
        pl.any_horizontal((pl.col(column).is_null() | (pl.col(column).str.len_chars() == 0)) for column in columns)
        & ~pl.col("source").str.starts_with("synthetic-")
    )
    assert bad.is_empty(), f"non-synthetic rows have empty values:\n{bad}"

    return (
        df.select(pl.concat_str([pl.col(column) for column in columns], separator=sep_token).alias("text"))
        .unique(maintain_order=True)["text"]
        .to_list()
    )


def tokenize_for_mlm(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> Dataset:
    """
    Tokenize `texts` into a HuggingFace `Dataset` with `input_ids`, `attention_mask`, and `special_tokens_mask`.
    No padding here — the MLM data collator pads dynamically per batch.
    """

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )

    return Dataset.from_dict({"text": texts}).map(tokenize, batched=True, remove_columns=["text"])  # type: ignore[bad-return]


class PretrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_shortname: str
    base_model: str

    # MLM
    mlm_probability: float = 0.3  # ModernBERT was pretrained with 30%
    max_seq_length: int = 8192  # ModernBERT's native context length

    # Training args
    global_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4  # 5e-5 was much slower. ModernBERT started w/ 5e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    num_train_epochs: float = 1.0
    gradient_checkpointing: bool = True  # ~free wall-clock with seq_len up to 8192 on ModernBERT-large

    # Data
    training_csvs: tuple[str, ...] = gt.data.DEFAULT_TRAIN_PATHS_NO_SYNTHETIC
    sample_size: int | None = None  # uniform downsample, applied after the full data is loaded
    n_rows_per_csv: int | None = None  # laptop-sanity prefix cap on `pl.read_csv`; biased, only for tiny_runs
    # stress-probe mode: train iterates longest sequences first; pair w/ max_steps
    sort_by_seq_length_desc: bool = False
    # MLM eval on held-out val.csv every save_steps. None disables eval. Sync to training process — keep small.
    eval_sample_size: int | None = 2000

    # Logging / checkpointing
    num_logs: int = 2000
    num_checkpoints: int = 50


def _load_model_and_tokenizer(base_model: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if gt.utils.is_gcs_uri(base_model):
        base_model = gt.utils.download_base_model_from_gcs(base_model)
    kwargs_model: dict[str, Any] = {}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        kwargs_model |= dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    model = AutoModelForMaskedLM.from_pretrained(base_model, **kwargs_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return model, tokenizer


class _PretrainTrainer(Trainer):
    def __init__(self, *args, sort_by_seq_length_desc: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sort_by_seq_length_desc = sort_by_seq_length_desc

    def _get_train_sampler(self, train_dataset=None) -> Sampler | None:
        if not self.sort_by_seq_length_desc:
            return super()._get_train_sampler(train_dataset)

        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if self.args.world_size > 1:
            return DistributedSampler(dataset, shuffle=False)  # type: ignore[bad-argument-type]
        return SequentialSampler(dataset)  # type: ignore[bad-argument-type]


def _sort_dataset_by_length_desc(dataset: Dataset) -> Dataset:
    dataset_with_length = dataset.map(
        lambda batch: {"length": [len(input_ids) for input_ids in batch["input_ids"]]},
        batched=True,
    )
    assert isinstance(dataset_with_length, Dataset)
    sorted_dataset = dataset_with_length.sort("length", reverse=True).remove_columns("length")
    assert isinstance(sorted_dataset, Dataset)
    return sorted_dataset  # type: ignore[bad-return]


def make_pretrainer(
    pretraining_config: PretrainingConfig,
    run_name: str | None = None,
    is_resumed: bool = False,
) -> Trainer:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_name = f"{timestamp}-{pretraining_config.run_shortname}"

    num_devices = max(1, torch.cuda.device_count())
    per_device_train_batch_size = max(1, pretraining_config.global_train_batch_size // num_devices)

    gc.collect()
    torch.cuda.empty_cache()
    # TODO: when is_resumed, load weights from the checkpoint instead so we skip a redundant gs:// base-model rsync.
    model, tokenizer = _load_model_and_tokenizer(pretraining_config.base_model)
    assert tokenizer.sep_token is not None, f"Tokenizer for {pretraining_config.base_model} has no sep_token"

    texts = load_texts(
        sep_token=tokenizer.sep_token,
        paths=pretraining_config.training_csvs,
        sample_size=pretraining_config.sample_size,
        n_rows_per_csv=pretraining_config.n_rows_per_csv,
    )
    logger.info(f"Pretraining corpus: {len(texts):,} unique texts")
    dataset_train = tokenize_for_mlm(texts, tokenizer, max_seq_length=pretraining_config.max_seq_length)

    if pretraining_config.sort_by_seq_length_desc:
        logger.info("Stress mode: sorting dataset by sequence length descending")
        dataset_train = _sort_dataset_by_length_desc(dataset_train)

    # Disable eval in stress-probe mode to keep the probe focused on the first few train steps
    eval_enabled = pretraining_config.eval_sample_size is not None and not pretraining_config.sort_by_seq_length_desc
    dataset_val: Dataset | None = None
    if eval_enabled:
        eval_texts = load_texts(
            sep_token=tokenizer.sep_token,
            paths=gt.data.DEFAULT_VAL_PATHS,
            sample_size=pretraining_config.eval_sample_size,
            n_rows_per_csv=pretraining_config.n_rows_per_csv,
        )
        logger.info(f"Eval corpus: {len(eval_texts):,} unique texts from val.csv")
        dataset_val = tokenize_for_mlm(eval_texts, tokenizer, max_seq_length=pretraining_config.max_seq_length)

    steps_total, logging_steps, save_steps = gt.utils.compute_train_steps(
        num_rows=len(dataset_train),
        num_devices=num_devices,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=pretraining_config.gradient_accumulation_steps,
        num_logs=pretraining_config.num_logs,
        num_checkpoints=pretraining_config.num_checkpoints,
        num_train_epochs=pretraining_config.num_train_epochs,
    )
    logger.info(f"Estimated {steps_total:,} optimizer steps, logging every {logging_steps}, saving every {save_steps}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=pretraining_config.mlm_probability,
    )
    return _PretrainTrainer(
        sort_by_seq_length_desc=pretraining_config.sort_by_seq_length_desc,
        model=model,
        args=TrainingArguments(
            output_dir=f"./{run_name}",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=False,
            dataloader_pin_memory=torch.cuda.is_available(),
            num_train_epochs=pretraining_config.num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=pretraining_config.gradient_accumulation_steps,
            gradient_checkpointing=pretraining_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=pretraining_config.learning_rate,
            weight_decay=pretraining_config.weight_decay,
            warmup_ratio=pretraining_config.warmup_ratio,
            seed=42,
            logging_strategy="steps",
            logging_steps=logging_steps,
            run_name=run_name,
            report_to="wandb",
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            ddp_find_unused_parameters=False,
            eval_strategy="steps" if eval_enabled else "no",
            eval_steps=save_steps if eval_enabled else None,
            eval_on_start=eval_enabled and not is_resumed,
            per_device_eval_batch_size=per_device_train_batch_size,
        ),
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
