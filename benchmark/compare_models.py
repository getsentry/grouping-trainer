"""
Runs each model across a grid of texts w/ different token lengths.

uv run python benchmark/compare_models.py
"""

import gc
import logging
import random
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

import polars as pl
import torch
from tap import tapify
from tqdm.auto import tqdm

import grouping_trainer as gt

logger = logging.getLogger(__name__)

MODEL_NAMES: tuple[str, ...] = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-small-v2",
    "Alibaba-NLP/gte-modernbert-base",
    "lightonai/modernbert-embed-large",
)

STEP_BETWEEN_INPUT_TOKENS = 16
MIN_TOKENS = 8
DEFAULT_COMPILE_TOKEN_BUCKETS: tuple[int, ...] = (64, 128, 256, 512, 1024)

type Version = Literal["base", "compiled"]


def _load_with_sdpa_fallback[T: gt.utils.SentenceTransformer](cls: type[T], model_name: str) -> T:
    model_kwargs = {}
    if torch.cuda.is_bf16_supported():
        model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    try:
        return cls(model_name, model_kwargs=model_kwargs)
    except ValueError as e:
        if "scaled_dot_product_attention" not in str(e):
            raise
        logger.warning(f"[{model_name}] SDPA not supported. Falling back to eager.")
        model_kwargs_eager = {k: v for k, v in model_kwargs.items() if k != "attn_implementation"}
        return cls(model_name, model_kwargs=model_kwargs_eager)


def _input_token_lengths(max_seq_length: int, step: int = STEP_BETWEEN_INPUT_TOKENS) -> list[int]:
    """
    Return sorted token-length targets: `step` grid plus B-1, B, B+1 for each bucket B.
    """
    grid = set(range(MIN_TOKENS, max_seq_length + 1, step))
    for bucket in DEFAULT_COMPILE_TOKEN_BUCKETS:
        if bucket > max_seq_length:
            continue
        for offset in (-1, 0, 1):
            value = bucket + offset
            if MIN_TOKENS <= value <= max_seq_length:
                grid.add(value)
    return sorted(grid)


def _generate_texts(
    tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]], input_token_lengths: list[int]
) -> dict[int, tuple[str, int]]:
    """
    Returns a dict mapping target input token lengths to (generated text, actual number of tokens in text).
    """
    target_num_tokens_to_text_and_actual_num: dict[int, tuple[str, int]] = {}
    for target_num_tokens in input_token_lengths:
        text = gt.compiled._create_text_with_num_tokens(target_num_tokens, tokenize_fn)
        actual_num_tokens = tokenize_fn([text])["input_ids"].shape[1]
        target_num_tokens_to_text_and_actual_num[target_num_tokens] = (text, actual_num_tokens)
    return target_num_tokens_to_text_and_actual_num


def _time_func[**P](func: Callable[P, object], *args: P.args, **kwargs: P.kwargs) -> float:
    start = time.monotonic()
    func(*args, **kwargs)
    return time.monotonic() - start


class Record(TypedDict):
    model_name: str
    version: Version
    phase: Literal["warmup", "run"]
    num_tokens_target: int | None  # None for warmup
    num_tokens_actual: int | None  # None for warmup
    latency_sec: float


def _run_model(
    model: gt.utils.SentenceTransformer,
    version: Version,
    model_name: str,
    target_num_tokens_to_text_and_actual_num: dict[int, tuple[str, int]],
) -> list[Record]:
    """
    Time the encode() call for each target.
    """
    records: list[Record] = []
    for target_num_tokens, (text, actual_num_tokens) in tqdm(
        target_num_tokens_to_text_and_actual_num.items(), desc=f"{model_name} {version}"
    ):
        latency = _time_func(model.encode, text, convert_to_numpy=True, show_progress_bar=False)
        records.append(
            {
                "model_name": model_name,
                "version": version,
                "phase": "run",
                "num_tokens_target": target_num_tokens,
                "num_tokens_actual": actual_num_tokens,
                "latency_sec": latency,
            }
        )
    return records


def _clear_context():
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()


def _target_num_tokens_to_text_and_actual_num(
    model_name: str, rng_for_input_order: random.Random
) -> dict[int, tuple[str, int]]:
    model = gt.utils.SentenceTransformer(model_name)
    input_token_lengths = _input_token_lengths(model.max_seq_length)
    target_num_tokens_to_text_and_actual_num = _generate_texts(model.tokenize, input_token_lengths)
    target_num_tokens_order = list(target_num_tokens_to_text_and_actual_num.keys())
    rng_for_input_order.shuffle(target_num_tokens_order)
    return {
        target_num_tokens: target_num_tokens_to_text_and_actual_num[target_num_tokens]
        for target_num_tokens in target_num_tokens_order
    }


def _benchmark_model(
    model_name: str,
    target_num_tokens_to_text_and_actual_num: dict[int, tuple[str, int]],
    *,
    version: Version,
) -> list[Record]:
    _clear_context()

    logger.info(f"[{model_name}] loading")
    model = _load_with_sdpa_fallback(
        gt.compiled.SentenceTransformer if version == "compiled" else gt.utils.SentenceTransformer,
        model_name,
    )
    if isinstance(model, gt.compiled.SentenceTransformer):
        warmup_sec = _time_func(model.compile_and_warm_up)
    else:
        warmup_sec = _time_func(model.encode, "warm up")

    records: list[Record] = []
    records.append(
        {
            "model_name": model_name,
            "version": version,
            "phase": "warmup",
            "num_tokens_target": None,
            "num_tokens_actual": None,
            "latency_sec": warmup_sec,
        }
    )
    records.extend(_run_model(model, version, model_name, target_num_tokens_to_text_and_actual_num))

    _clear_context()
    return records


def _add_bucket_label(df: pl.DataFrame, buckets: tuple[int, ...], column: str) -> pl.DataFrame:
    """
    Returns a df w/ a new column `bucket` which categorizes the values in `column` into buckets defined by `buckets`.
    """
    labels = [f"<={buckets[0]}"]
    for i in range(1, len(buckets)):
        labels.append(f"{buckets[i - 1] + 1}-{buckets[i]}")
    labels.append(f">{buckets[-1]}")
    return df.with_columns(bucket=pl.col(column).cut(breaks=list(buckets), labels=labels))


def _df_to_markdown(df: pl.DataFrame) -> str:
    with pl.Config(
        tbl_formatting="MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=10000,
        fmt_str_lengths=1000,
    ):
        return str(df)


def _summary_per_model_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pivot `df`'s `phase="run"` rows wide on `version` and aggregate per `(model_name, bucket)`.
    """
    df = df.filter(pl.col("phase") == "run").drop_nulls(["num_tokens_actual"])
    df = _add_bucket_label(df, buckets=DEFAULT_COMPILE_TOKEN_BUCKETS, column="num_tokens_actual")
    df = df.pivot(
        on="version",
        index=["model_name", "bucket", "num_tokens_target", "num_tokens_actual"],
        values="latency_sec",
    )
    df = (
        df.group_by(["model_name", "bucket"], maintain_order=False)
        .agg(
            pl.len().alias("n"),
            pl.col("num_tokens_actual").median().alias("tok_p50"),
            (pl.col("base").median() * 1000).round(2).alias("base_ms_p50"),
            (pl.col("compiled").median() * 1000).round(2).alias("compiled_ms_p50"),
            (pl.col("base").quantile(0.9) * 1000).round(2).alias("base_ms_p90"),
            (pl.col("compiled").quantile(0.9) * 1000).round(2).alias("compiled_ms_p90"),
            (pl.col("base").median() / pl.col("compiled").median()).round(2).alias("speedup_p50"),
        )
        .sort(["model_name", "tok_p50"])
    )
    return df


def _warmup_summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    One row per model with the compiled-version `compile_and_warm_up()` wall time.
    """
    return (
        df.filter((pl.col("phase") == "warmup") & (pl.col("version") == "compiled"))
        .select(
            pl.col("model_name"),
            pl.col("latency_sec").round(1).alias("compile_and_warmup_sec"),
        )
        .sort("model_name")
    )


def main(
    output_path: Path | None = None,
    seed: int = 0,
    model_names: tuple[str, ...] = MODEL_NAMES,
):
    """
    Benchmark gt.utils.SentenceTransformer vs gt.compiled.SentenceTransformer across multiple models.

    Parameters
    ----------
    output_path
        CSV output path. Defaults to benchmark/results/multi_model_<timestamp>.csv.
    seed
        Random seed for the per-model input-order shuffle.
    model_names
        HuggingFace model IDs to benchmark. Defaults to MODEL_NAMES at the top of this file.
    """
    gt.logging.configure_logging(process_type="benchmark_multi_model")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (gt.compiled.SentenceTransformer uses CUDA graphs).")

    # Set output path
    if output_path is None:
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_path = Path("benchmark/results") / f"multi_model_{stamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run models on benchmark
    rng_for_input_order = random.Random(seed)
    records_for_all_models: list[Record] = []
    for model_name in model_names:
        target_num_tokens_to_text_and_actual_num = _target_num_tokens_to_text_and_actual_num(
            model_name, rng_for_input_order
        )
        for version in ("base", "compiled"):
            records_for_all_models.extend(
                _benchmark_model(model_name, target_num_tokens_to_text_and_actual_num, version=version)
            )

    # Write results
    df = pl.DataFrame(records_for_all_models)
    df.write_csv(output_path)
    logger.info(f"Wrote {len(df):,} records to {output_path}")
    # Should be small enough to print
    print()
    print("=== Per-(model, bucket) latency (medians in ms, speedup = base / compiled) ===")
    print(_df_to_markdown(_summary_per_model_bucket(df)))
    print()
    print("=== compile_and_warm_up() wall time per model ===")
    print(_df_to_markdown(_warmup_summary(df)))
    print()


if __name__ == "__main__":
    tapify(main)
