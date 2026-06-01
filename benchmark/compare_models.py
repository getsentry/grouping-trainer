"""
Runs each model across a grid of texts w/ different token lengths, comparing the regular SentenceTransformer vs its CUDA
graph-compiled version.

uv run python benchmark/compare_models.py
"""

import gc
import logging
import os
import random
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from string import ascii_lowercase
from typing import Literal, TypedDict, cast

import numpy as np
import polars as pl
import torch
from tap import tapify
from tqdm.auto import tqdm

import grouping_trainer as gt

logger = logging.getLogger(__name__)

MODEL_NAMES: tuple[str, ...] = (
    # Add more models here:
    # ...
    # All models below achieve the expected speedup. To test new models, add them above so they're tested first.
    "lightonai/modernbert-embed-large",
    # "Alibaba-NLP/gte-modernbert-base",  # TODO: seems to drift w/ compilation
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-small-v2",
)
MODEL_NAME_TO_ATOL: dict[str, float] = {
    "lightonai/modernbert-embed-large": 6e-3,
}

STEP_BETWEEN_INPUT_TOKENS = 8
MIN_TOKENS = 8
DEFAULT_COMPILE_TOKEN_BUCKETS: tuple[int, ...] = (64, 128, 256, 512, 1024)

type Version = Literal["base", "compiled"]


def _load_sdpa_with_eager_fallback[T: gt.utils.SentenceTransformer](cls: type[T], model_name: str) -> T:
    model_kwargs = {"attn_implementation": "sdpa"}
    if torch.cuda.is_bf16_supported():
        # The benefit of compilation positively interacts w/ bfloat16. CUDA compute is much faster b/c of tensor cores,
        # and memory movement is also faster. So Python overhead is relatively higher.
        model_kwargs |= {"dtype": torch.bfloat16}
    try:
        return cls(model_name, model_kwargs=model_kwargs)
    except ValueError as exception:
        if "scaled_dot_product_attention" not in str(exception):
            raise exception
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


def _create_random_text_with_num_tokens(
    target_num_tokens: int,
    tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]],
    seed: int | None = 42,
    **tokenize_kwargs,
) -> str:
    """
    Return a text of random single-character words that tokenizes to exactly `target_num_tokens` tokens.
    """

    # Random chars should give more diverse embeddings than repeating. Want to avoid numerical drift flying under the
    # radar in the correctness check. Ideally we have a small corpus of single-token words to pick from, but it's not
    # clear to me that's not a marginal improvement.
    def count_tokens(text):
        return tokenize_fn([text], **tokenize_kwargs)["input_ids"].shape[1]

    num_specials = count_tokens("")
    single_character_tokens = [
        character for character in ascii_lowercase if (count_tokens(" " + character) - num_specials) == 1
    ]
    if not single_character_tokens:
        raise RuntimeError("no single-token characters found. Weird tokenizer?")

    rng = random.Random(seed)
    text = "".join(" " + rng.choice(single_character_tokens) for _ in range(target_num_tokens - num_specials))

    if count_tokens(text) != target_num_tokens:
        raise RuntimeError("target not reachable by single-char steps")

    return text


def _generate_texts(
    tokenize_fn: Callable[[list[str]], dict[str, torch.Tensor]], input_token_lengths: list[int]
) -> dict[int, str]:
    """
    Returns a dict mapping target input token lengths to the generated text.
    """
    return {
        target_num_tokens: _create_random_text_with_num_tokens(target_num_tokens, tokenize_fn)
        for target_num_tokens in input_token_lengths
    }


def _time_func[**P, R](func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
    start = time.monotonic()
    out = func(*args, **kwargs)
    latency_sec = time.monotonic() - start
    return out, latency_sec


class Record(TypedDict):
    model_name: str
    version: Version
    phase: Literal["warmup", "run"]
    num_tokens: int | None  # None for warmup
    latency_sec: float


@dataclass(kw_only=True, frozen=True)
class BenchmarkModelResult:
    records: list[Record]
    embeddings: np.ndarray


def _run_model(
    model: gt.utils.SentenceTransformer,
    version: Version,
    model_name: str,
    target_num_tokens_to_text: dict[int, str],
) -> BenchmarkModelResult:
    """
    Time the encode() call for each target.
    """
    assert target_num_tokens_to_text
    records: list[Record] = []
    embeddings: list[np.ndarray] = []
    for target_num_tokens, text in tqdm(target_num_tokens_to_text.items(), desc=f"{model_name} {version}"):
        embedding, latency_sec = _time_func(model.encode, text, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(cast(np.ndarray, embedding))
        records.append(
            {
                "model_name": model_name,
                "version": version,
                "phase": "run",
                "num_tokens": target_num_tokens,
                "latency_sec": latency_sec,
            }
        )
    return BenchmarkModelResult(records=records, embeddings=np.array(embeddings))


def _clear_context():
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()


def _target_num_tokens_to_text(model_name: str) -> dict[int, str]:
    model = gt.utils.SentenceTransformer(model_name)
    input_token_lengths = _input_token_lengths(model.max_seq_length)
    target_num_tokens_to_text = _generate_texts(model.tokenize, input_token_lengths)
    target_num_tokens_order = list(target_num_tokens_to_text.keys())
    random.Random(42).shuffle(target_num_tokens_order)
    return {
        target_num_tokens: target_num_tokens_to_text[target_num_tokens] for target_num_tokens in target_num_tokens_order
    }


def _benchmark_model_version(
    model_name: str,
    target_num_tokens_to_text: dict[int, str],
    *,
    version: Version,
) -> BenchmarkModelResult:
    _clear_context()

    logger.info(f"[{model_name}] loading")
    model = _load_sdpa_with_eager_fallback(
        gt.compiled.SentenceTransformer if version == "compiled" else gt.utils.SentenceTransformer,
        model_name,
    )
    if isinstance(model, gt.compiled.SentenceTransformer):
        _, warmup_sec = _time_func(model.compile_and_warm_up)
    else:
        _, warmup_sec = _time_func(model.encode, "warm up")

    records: list[Record] = []
    records.append(
        {
            "model_name": model_name,
            "version": version,
            "phase": "warmup",
            "num_tokens": None,
            "latency_sec": warmup_sec,
        }
    )
    benchmark_model_result = _run_model(model, version, model_name, target_num_tokens_to_text)
    records.extend(benchmark_model_result.records)

    _clear_context()
    return BenchmarkModelResult(records=records, embeddings=benchmark_model_result.embeddings)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


def _benchmark_model(model_name: str, versions: tuple[Version, ...] = ("base", "compiled")) -> list[Record]:
    target_num_tokens_to_text = _target_num_tokens_to_text(model_name)
    records: list[Record] = []
    version_to_embeddings: dict[Version, np.ndarray] = {}

    for version in versions:
        benchmark_model_result = _benchmark_model_version(
            model_name,
            target_num_tokens_to_text,
            version=version,
        )
        records.extend(benchmark_model_result.records)
        version_to_embeddings[version] = benchmark_model_result.embeddings

    # Sanity check correctness by comparing cos sim. PyTorch's huggingface dynamo bench does allclose(eager_bf16,
    # compiled_bf16) at tol=1e-3 (bumped to 4e-3 for known-noisy models) for bf16+compile AMP inference checking, with
    # an fp64 reference as a second-chance fallback when allclose fails.
    # https://github.com/pytorch/pytorch/blob/19ecfe58b45fe56afcd9155ad721dcf9a7569339/benchmarks/dynamo/huggingface.py#L529
    default_atol = 1e-3 if torch.cuda.is_bf16_supported() else 1e-4
    atol = MODEL_NAME_TO_ATOL.get(model_name, default_atol)
    for version1, version2 in combinations(versions, 2):
        cos_sim = _cos_sim(version_to_embeddings[version1], version_to_embeddings[version2])
        diag = np.diag(cos_sim)
        # rtol contributes ~atol on top since target ≈ 1.0 (effective tolerance ~2e-3), following
        # torch.testing.assert_close's convention of `rtol == atol` for reduced-precision dtypes.
        assert np.allclose(diag, 1.0, atol=atol, rtol=atol), (
            f"Cos sim isn't always numerically close to 1.0 for {version1} vs {version2}. "
            f"Observed range: [{diag.min()}, {diag.max()}]"
        )

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
    df = df.filter(pl.col("phase") == "run")
    df = _add_bucket_label(df, buckets=DEFAULT_COMPILE_TOKEN_BUCKETS, column="num_tokens")
    df = df.pivot(
        on="version",
        index=["model_name", "bucket", "num_tokens"],
        values="latency_sec",
    )
    df = (
        df.group_by(["model_name", "bucket"], maintain_order=False)
        .agg(
            pl.len().alias("n"),
            pl.col("num_tokens").median().alias("tok_p50"),
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
    model_names: tuple[str, ...] = MODEL_NAMES,
    *,
    no_gpu: bool = False,
    zone: str | None = None,
):
    """
    Benchmark gt.utils.SentenceTransformer vs gt.compiled.SentenceTransformer across multiple models.

    Parameters
    ----------
    model_names
        Space-separated list of HuggingFace model IDs to benchmark.
        Example: "BAAI/bge-small-en-v1.5 BAAI/bge-base-en-v1.5 sentence-transformers/all-MiniLM-L6-v2"
    no_gpu
        Don't sync-start an L4 and run this same invocation there, instead run it locally.
    zone
        Override the default GCP zone for the L4 when launching the GPU instance. Useful when capacity is dry in the
        default zone.
    """
    gt.logging.configure_logging(process_type="benchmark_multi_model")

    if not (no_gpu or gt.launch.is_on_remote()):
        run_name = gt.launch.run_name_from_shortname("compare-models")
        gt.launch.run_argv_remotely(
            gpu="l4",
            job_type=gt.launch.JobType.BENCHMARK,
            run_name=run_name,
            zone=zone,
        )
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required b/c gt.compiled.SentenceTransformer uses CUDA graphs.")

    df = pl.DataFrame(
        [
            record
            for model_name in tqdm(model_names, desc="Benchmarking models")
            for record in _benchmark_model(model_name)
        ]
    )

    print()
    print("=== Per-(model, bucket) latency (medians in ms, speedup = base / compiled) ===")
    print(_df_to_markdown(_summary_per_model_bucket(df)))
    print()
    print("=== compile_and_warm_up() wall time per model ===")
    print(_df_to_markdown(_warmup_summary(df)))
    print()

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path_output = Path("benchmark/results") / f"multi_model_{timestamp}.csv"
    path_output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path_output)
    logger.info(f"Wrote {len(df):,} records to {path_output}")

    if gt.launch.is_on_remote():
        path_gcs_output = f"gs://{os.environ['GROUPING_TRAINER_BUCKET']}/perf/multi-model/{timestamp}/multi_model.csv"
        subprocess.run(["gcloud", "storage", "cp", str(path_output), path_gcs_output], check=True)


if __name__ == "__main__":
    tapify(main)
