import logging
import math
import os
import subprocess
from collections.abc import Iterable
from typing import Literal, cast, overload

import numpy as np
import polars as pl
import torch
from polars._typing import ConcatMethod
from sentence_transformers import SentenceTransformer as SentenceTransformerOriginal
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_GCS_PREFIX = "gs://"
_DIR_BASE_MODELS_LOCAL = "_base_models"


def concat_vertical_unordered(
    dfs: Iterable[pl.DataFrame],
    how: ConcatMethod = "vertical",
    rechunk: bool = False,
    parallel: bool = True,
) -> pl.DataFrame:
    """
    Polars doesn't have a mode for vertical concatenation when the same columns are in different orders.
    """
    dfs_iter = iter(dfs)
    df_first = next(dfs_iter)
    columns = set(df_first.columns)
    dfs_ordered = [df_first]
    for df in dfs_iter:
        if columns != set(df.columns):
            raise ValueError(f"Columns are not the same: {sorted(columns)} != {sorted(df.columns)}")
        dfs_ordered.append(df.select(df_first.columns))
    return pl.concat(dfs_ordered, how=how, rechunk=rechunk, parallel=parallel)


def deduplicate_pairs(
    df: pl.DataFrame,
    column1: str = "query_stacktrace_string",
    column2: str = "candidate_stacktrace_string",
) -> pl.DataFrame:
    """
    Keeps the first occurrence of each pair of `column1` and `column2`, even if their values appear as `column2` and
    `column1`.

    Grouping is symmetric.
    """
    assert "_pair_first" not in df.columns and "_pair_second" not in df.columns, (
        "input df must not have columns named '_pair_first' or '_pair_second' (used as scratch)"
    )
    return (
        df.with_columns(
            _pair_first=pl.min_horizontal(column1, column2),
            _pair_second=pl.max_horizontal(column1, column2),
        )
        .unique(subset=["_pair_first", "_pair_second"], keep="first", maintain_order=True)
        .drop(["_pair_first", "_pair_second"])
        .select(df.columns)
    )


class SentenceTransformer(SentenceTransformerOriginal):
    """
    `SentenceTransformer` which deduplicates texts during inference and retries OOMs once.
    """

    def __init__(self, *args, text_prefix: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text_prefix = text_prefix

    # The getter and setter below are just for type hints. SentenceTransformer annotates it as Any

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return super().tokenizer

    @tokenizer.setter
    def tokenizer(self, value: PreTrainedTokenizerBase) -> None:
        self._first_module().tokenizer = value  # type: ignore[assignment]

    def tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], **kwargs) -> dict[str, torch.Tensor]:
        if self.text_prefix:
            if isinstance(texts, list):
                assert all(isinstance(text, str) for text in texts)
                texts = cast(list[str], texts)
                texts = [self.text_prefix + text for text in texts]
            else:
                raise ValueError(f"Not sure how to add the prefix for the input text type: {type(texts)}")
        return super().tokenize(texts, **kwargs)

    @overload
    def encode(  # type: ignore[bad-override]
        self,
        sentences: str,
        **kwargs,
    ) -> torch.Tensor: ...

    @overload
    def encode(
        self,
        sentences: list[str] | np.ndarray,
        *,
        convert_to_numpy: Literal[True],
        convert_to_tensor: bool = False,
        **kwargs,
    ) -> np.ndarray: ...

    @overload
    def encode(
        self,
        sentences: list[str] | np.ndarray,
        *,
        convert_to_numpy: bool = False,
        convert_to_tensor: Literal[True],
        **kwargs,
    ) -> torch.Tensor: ...

    def encode(self, sentences: str | list[str] | np.ndarray, **kwargs) -> np.ndarray | torch.Tensor:
        texts = sentences
        if isinstance(texts, str | np.ndarray):
            return super().encode(texts, **kwargs)

        unique = list(dict.fromkeys(texts))  # preserve order
        text_to_idx = {text: idx for idx, text in enumerate(unique)}
        embeddings = super().encode(unique, **kwargs)
        return embeddings[[text_to_idx[text] for text in texts]]  # assume numpy or torch


def is_gcs_uri(uri: str) -> bool:
    return uri.startswith(_GCS_PREFIX)


def assert_gcs_path_exists(uri: str) -> None:
    """
    Raises `CalledProcessError` if `uri` doesn't exist (or matches no objects) in GCS.
    """
    subprocess.run(
        ["gcloud", "storage", "ls", uri.rstrip("/") + "/"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _download_base_model_from_gcs(uri: str) -> str:
    """
    Rsync `uri` (a `gs://...` directory) into `_base_models/<basename>/` relative to CWD and return the local path.
    """
    basename = uri.rstrip("/").rsplit("/", 1)[-1]
    path_local = os.path.join(_DIR_BASE_MODELS_LOCAL, basename)
    logger.info(f"Downloading base model: {uri} -> {path_local}")
    subprocess.run(["gcloud", "storage", "rsync", "-r", uri.rstrip("/"), path_local], check=True)
    return path_local


def encoder_from_base(base_model: str, use_text_prefix: bool = False) -> SentenceTransformer:
    """
    Build a SentenceTransformer encoder with standard dtype/attention settings.

    `base_model` is a HuggingFace model ID, a local path, or a `gs://...` path to a custom model directory. gs:// models
    are downloaded into `_base_models/` (relative to CWD) on first call.

    Handles model-specific quirks (e.g. jina v5's config_kwargs and trust_remote_code) and enables bfloat16 + SDPA when
    supported.
    """
    if is_gcs_uri(base_model):
        base_model = _download_base_model_from_gcs(base_model)

    if base_model == "jinaai/jina-embeddings-v5-text-nano-text-matching":
        return SentenceTransformer(
            base_model,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
            config_kwargs={"_attn_implementation": "sdpa"},
        )

    model_kwargs = None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")

    text_prefix = ""
    if base_model == "lightonai/modernbert-embed-large" and use_text_prefix:
        # https://huggingface.co/lightonai/modernbert-embed-large#usage
        text_prefix = "clustering: "

    return SentenceTransformer(
        base_model,
        model_kwargs=model_kwargs,
        text_prefix=text_prefix,
    )


def compute_train_steps(
    *,
    num_rows: int,
    num_devices: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_logs: int,
    num_checkpoints: int,
    num_train_epochs: float = 1.0,
) -> tuple[int, int, int]:
    """
    Returns `steps_total, logging_steps, save_steps` for HF `Trainer`s.
    """
    rows_per_device = math.ceil(num_rows / num_devices)  # DistributedSampler pads
    num_batches = math.ceil(rows_per_device / per_device_train_batch_size)
    steps_per_epoch = math.ceil(num_batches / gradient_accumulation_steps)
    steps_total = max(1, math.ceil(steps_per_epoch * num_train_epochs))
    logging_steps = max(1, steps_total // num_logs)
    save_steps = max(1, steps_total // num_checkpoints)
    return steps_total, logging_steps, save_steps
