import gc
from collections.abc import Iterable
from functools import wraps

import polars as pl
import torch
from polars._typing import ConcatMethod
from sentence_transformers import SentenceTransformer as SentenceTransformerOriginal
from transformers import PreTrainedTokenizerBase


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


def _cuda_empty_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _retry_cuda_errors_once(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "CUDA" not in str(e):
                raise
            _cuda_empty_cache()
            return func(*args, **kwargs)

    return wrapper


class SentenceTransformer(SentenceTransformerOriginal):
    """
    `SentenceTransformer` which deduplicates texts during inference and retries OOMs once.
    """

    def __init__(self, *args, text_prefix: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text_prefix = text_prefix

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return super().tokenizer

    @tokenizer.setter
    def tokenizer(self, value: PreTrainedTokenizerBase) -> None:
        self._first_module().tokenizer = value

    # The getter and setter above are just for type hints. SentenceTransformer annotates it as Any

    def tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], **kwargs) -> dict[str, torch.Tensor]:
        if self.text_prefix:
            if isinstance(texts, list) and all(isinstance(text, str) for text in texts):
                texts = [self.text_prefix + text for text in texts]
            else:
                raise ValueError(f"Not sure how to add the prefix for the input text type: {type(texts)}")
        return super().tokenize(texts, **kwargs)

    @_retry_cuda_errors_once
    def encode(self, texts: str | list[str], **kwargs):
        if isinstance(texts, str):
            return super().encode(texts, **kwargs)

        unique = list(dict.fromkeys(texts))  # preserve order
        text_to_idx = {text: idx for idx, text in enumerate(unique)}
        embeddings = super().encode(unique, **kwargs)
        return embeddings[[text_to_idx[text] for text in texts]]  # assume numpy or torch


def encoder_from_base(base_model: str, use_text_prefix: bool = True) -> SentenceTransformer:
    """
    Build a SentenceTransformer encoder with standard dtype/attention settings.

    Handles model-specific quirks (e.g. jina v5's config_kwargs and trust_remote_code) and enables bfloat16 + SDPA when
    CUDA supports it.
    """
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
