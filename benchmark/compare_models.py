"""
Multi-model benchmark: gt.utils.SentenceTransformer vs gt.compiled.SentenceTransformer.

For each model, sweeps token lengths from MIN_TOKENS to max_seq_length (step SWEEP_STEP, plus
explicit points adjacent to each compiled bucket boundary). For each (model, version, num_tokens),
times one encode() call in a randomized order. Writes raw per-call latencies to CSV and prints a
per-(model, bucket) speedup summary plus warmup times and any failures.

batch_size = 1 throughout. Single-shot timings (no median over repeats) so cold-start / cache-miss
costs aren't hidden. Same shuffled sweep order is used for both versions of each model.

Example:

    uv run python benchmark/compare_models.py --output_path benchmark/results/multi_model.csv
"""

import logging
import random
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
from accelerate.utils import release_memory
from tap import tapify
from tqdm.auto import tqdm

import grouping_trainer as gt

logger = logging.getLogger(__name__)

MODELS: tuple[str, ...] = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-small-v2",
    "lightonai/modernbert-embed-large",
)

SWEEP_STEP = 16
MIN_TOKENS = 8
DEFAULT_BUCKETS: tuple[int, ...] = (64, 128, 256, 512, 1024)


def _model_kwargs() -> dict:
    """
    Match `gt.utils.encoder_from_base`'s dtype/attention settings when bf16 is supported.
    """
    if torch.cuda.is_bf16_supported():
        return dict(dtype=torch.bfloat16, attn_implementation="sdpa")
    return {}


def _load_with_sdpa_fallback[T: gt.utils.SentenceTransformer](cls: type[T], model_name: str) -> T:
    """
    Construct `cls(model_name, ...)` with the standard model_kwargs. If the model's architecture
    doesn't have an SDPA implementation in transformers (e.g. MPNet), retry without
    `attn_implementation` so it falls back to eager attention.
    """
    model_kwargs = _model_kwargs()
    try:
        return cls(model_name, model_kwargs=model_kwargs)
    except ValueError as e:
        if "scaled_dot_product_attention" not in str(e):
            raise
        logger.warning(f"[{model_name}] SDPA not supported; retrying with eager attention")
        model_kwargs_eager = {k: v for k, v in model_kwargs.items() if k != "attn_implementation"}
        return cls(model_name, model_kwargs=model_kwargs_eager)


def _sweep_lengths(max_seq_length: int) -> list[int]:
    """
    Return sorted token-length targets: step-`SWEEP_STEP` grid plus B-1, B, B+1 for each default bucket B.
    """
    grid: set[int] = set(range(MIN_TOKENS, max_seq_length + 1, SWEEP_STEP))
    for bucket in DEFAULT_BUCKETS:
        if bucket > max_seq_length:
            continue
        for offset in (-1, 0, 1):
            value = bucket + offset
            if MIN_TOKENS <= value <= max_seq_length:
                grid.add(value)
    return sorted(grid)


def _generate_texts(
    tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]],
    sweep: list[int],
) -> dict[int, tuple[str, int]]:
    """
    Build `{target_num_tokens: (text, actual_num_tokens)}` using `tokenize_fn`.
    """
    texts: dict[int, tuple[str, int]] = {}
    for target in sweep:
        text = gt.compiled._create_text_with_num_tokens(target, tokenize_fn)
        actual = tokenize_fn([text])["input_ids"].shape[1]
        texts[target] = (text, actual)
    return texts


def _encode_once(model: gt.utils.SentenceTransformer, text: str) -> float:
    """
    Wall time of a single encode(). The implicit device->host transfer in .encode() syncs CUDA.
    """
    start = time.monotonic()
    _ = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return time.monotonic() - start


def _benchmark_model(model_name: str, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """
    Run compiled then base for one model. Compiled-first matches benchmark/run.py and avoids letting a
    prior base encode dirty the CUDA allocator state that `mode="reduce-overhead"` needs for cudagraph
    capture. Returns (rows_for_csv, failures).
    """
    rows: list[dict] = []
    failures: list[dict] = []
    sweep_order: list[int] | None = None
    texts: dict[int, tuple[str, int]] | None = None
    warmup_sec: float = 0.0
    model_compiled: gt.compiled.SentenceTransformer | None = None

    try:
        logger.info(f"[{model_name}] loading compiled")
        model_compiled = _load_with_sdpa_fallback(gt.compiled.SentenceTransformer, model_name)
    except Exception as e:
        logger.exception(f"[{model_name}] compiled load failed")
        failures.append({"model_name": model_name, "version": "compiled", "reason": f"{type(e).__name__}: {e}"})

    if model_compiled is not None:
        try:
            logger.info(f"[{model_name}] compile_and_warm_up")
            start = time.monotonic()
            model_compiled.compile_and_warm_up()
            warmup_sec = time.monotonic() - start
        except Exception as e:
            logger.exception(f"[{model_name}] compile_and_warm_up failed")
            failures.append(
                {
                    "model_name": model_name,
                    "version": "compiled",
                    "reason": f"compile_and_warm_up: {type(e).__name__}: {e}",
                }
            )
            (model_compiled,) = release_memory(model_compiled)
            model_compiled = None

    if model_compiled is not None:
        max_seq = model_compiled.max_seq_length
        sweep = _sweep_lengths(max_seq)
        logger.info(f"[{model_name}] max_seq={max_seq}, sweep={len(sweep)} lengths")

        # Fresh local so the closure below doesn't widen the capture back to `... | None`.
        model_compiled_narrowed = model_compiled

        # Bypass compiled.SentenceTransformer.tokenize (which pads to buckets) so the calibration in
        # _create_text_with_num_tokens sees true token counts.
        def tokenize_unpadded(batch: list[str]) -> dict[str, torch.Tensor]:
            return gt.utils.SentenceTransformer.tokenize(model_compiled_narrowed, batch)

        texts = _generate_texts(tokenize_unpadded, sweep)
        sweep_order = list(sweep)
        rng.shuffle(sweep_order)

        rows.append(
            {
                "model_name": model_name,
                "version": "compiled",
                "phase": "warmup",
                "num_tokens_target": None,
                "num_tokens_actual": None,
                "latency_sec": warmup_sec,
                "iteration": 0,
            }
        )

        for target in tqdm(sweep_order, desc=f"{model_name} compiled"):
            text, actual = texts[target]
            latency = _encode_once(model_compiled, text)
            rows.append(
                {
                    "model_name": model_name,
                    "version": "compiled",
                    "phase": "sweep",
                    "num_tokens_target": target,
                    "num_tokens_actual": actual,
                    "latency_sec": latency,
                    "iteration": 0,
                }
            )

        (model_compiled,) = release_memory(model_compiled)

    try:
        logger.info(f"[{model_name}] loading base")
        model_base = _load_with_sdpa_fallback(gt.utils.SentenceTransformer, model_name)
    except Exception as e:
        logger.exception(f"[{model_name}] base load failed")
        failures.append({"model_name": model_name, "version": "base", "reason": f"{type(e).__name__}: {e}"})
        return rows, failures

    if sweep_order is None or texts is None:
        max_seq = model_base.max_seq_length
        sweep = _sweep_lengths(max_seq)
        logger.info(f"[{model_name}] max_seq={max_seq}, sweep={len(sweep)} lengths")
        texts = _generate_texts(model_base.tokenize, sweep)
        sweep_order = list(sweep)
        rng.shuffle(sweep_order)

    _ = model_base.encode("warm up", convert_to_numpy=True, show_progress_bar=False)

    for target in tqdm(sweep_order, desc=f"{model_name} base"):
        text, actual = texts[target]
        latency = _encode_once(model_base, text)
        rows.append(
            {
                "model_name": model_name,
                "version": "base",
                "phase": "sweep",
                "num_tokens_target": target,
                "num_tokens_actual": actual,
                "latency_sec": latency,
                "iteration": 0,
            }
        )

    (model_base,) = release_memory(model_base)
    return rows, failures


def _bucketize(df: pl.DataFrame, edges: tuple[int, ...], column: str) -> pl.DataFrame:
    """
    Add a `bucket` column whose categories are <=e0, e0+1..e1, ..., >e_last.
    """
    labels = [f"<={edges[0]}"]
    for i in range(1, len(edges)):
        labels.append(f"{edges[i - 1] + 1}-{edges[i]}")
    labels.append(f">{edges[-1]}")
    return df.with_columns(bucket=pl.col(column).cut(breaks=list(edges), labels=labels))


def _df_to_markdown(df: pl.DataFrame) -> str:
    """
    Render a Polars DataFrame as a GitHub-flavored markdown table.
    """
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
    Pivot sweep rows wide on `version` and aggregate per (model, bucket).
    """
    df_sweep = df.filter(pl.col("phase") == "sweep").drop_nulls(["num_tokens_actual"])
    df_sweep = _bucketize(df_sweep, edges=DEFAULT_BUCKETS, column="num_tokens_actual")

    df_wide = df_sweep.pivot(
        on="version",
        index=["model_name", "bucket", "num_tokens_target", "num_tokens_actual"],
        values="latency_sec",
    )
    for col in ("base", "compiled"):
        if col not in df_wide.columns:
            df_wide = df_wide.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return (
        df_wide.group_by(["model_name", "bucket"], maintain_order=False)
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
    models: tuple[str, ...] = MODELS,
):
    """
    Benchmark gt.utils.SentenceTransformer vs gt.compiled.SentenceTransformer across multiple models.

    Parameters
    ----------
    output_path
        CSV output path. Defaults to benchmark/results/multi_model_<timestamp>.csv.
    seed
        Random seed for the per-model sweep-order shuffle.
    models
        HuggingFace model IDs to benchmark. Defaults to MODELS at the top of this file.
    """
    gt.logging.configure_logging(process_type="benchmark_multi_model")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (gt.compiled.SentenceTransformer uses CUDA graphs).")

    if output_path is None:
        stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_path = Path("benchmark/results") / f"multi_model_{stamp}.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rows_all: list[dict] = []
    failures_all: list[dict] = []

    for model_name in models:
        # Dynamo's submodule caches (Pooling, Normalize, ...) are keyed by function identity, so
        # they accumulate entries across models in one process and silently overflow the default
        # cache_size_limit of 8. Reset between models so each one compiles from a clean slate.
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        rows, failures = _benchmark_model(model_name, rng)
        rows_all.extend(rows)
        failures_all.extend(failures)

    if not rows_all:
        logger.error("No rows collected; nothing to write.")
        return

    df = pl.DataFrame(rows_all)
    df.write_csv(output_path)
    logger.info(f"Wrote {len(df):,} rows to {output_path}")

    print()
    print("=== Per-(model, bucket) latency (medians in ms, speedup = base / compiled) ===")
    print(_df_to_markdown(_summary_per_model_bucket(df)))
    print()
    print("=== compile_and_warm_up() wall time per model ===")
    print(_df_to_markdown(_warmup_summary(df)))
    print()
    if failures_all:
        print("=== Failures ===")
        print(_df_to_markdown(pl.DataFrame(failures_all)))
    else:
        print("=== No failures ===")


if __name__ == "__main__":
    tapify(main, description=__doc__)
