"""
Benchmark compiled vs non-compiled SentenceTransformer on deduped query stacktraces.

Times one encode() call per unique query_stacktrace_string, for both gt.compiled.SentenceTransformer
and gt.utils.SentenceTransformer, and uploads the results to

gs://grouping-data/perf/{stamp}-{run_id}/{dataset_name}/

Example:

python benchmark/run.py \
    --run_gcs_dir gs://grouping-data/runs/2026-04-10-12-39-45-large-no-prefix
"""

# NOTE: not sure why ONNX didn't play nice w/ ModernBERT. Tried the ONNX backend w/ the latest pre-transformers v5
# versions (onnxruntime-gpu==1.25.0, optimum==1.27.0, including the O2 optimizer pass via export_optimized_onnx_model)
# on an L4 with cu128. ModernBERT support was added in optimum==1.26.0:
# https://github.com/huggingface/optimum/releases/tag/v1.26.0. It was 1.5x slower than the eager bf16+sdpa baseline. Got
# these warnings:
#
# 2026-04-25 19:57:03.793307561 [W:onnxruntime:, transformer_memcpy.cc:111 ApplyImpl] 28 Memcpy nodes are added to the
# graph main_graph for CUDAExecutionProvider. It might have negative impact on performance (including unable to run CUDA
# graph). Set session_options.log_severity_level=1 to see the detail logs before this message.
#
# 2026-04-25 19:57:03.815823108 [W:onnxruntime:, session_state.cc:1359 VerifyEachNodeIsAssignedToAnEp] Some nodes were
# not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g.
# ORT explicitly assigns shape related ops to CPU to improve perf.
#
# 2026-04-25 19:57:03.815865405 [W:onnxruntime:, session_state.cc:1361 VerifyEachNodeIsAssignedToAnEp] Rerunning with
# verbose output on a non-minimal build will show node assignments.
#
# (Digging around w/ Claude) In optimum 1.27.0 and onnxruntime 1.25.0:
# - optimum/onnxruntime/utils.py:111 maps "modernbert" -> "bert", so optimum routes ModernBERT graphs through ORT's
#   BERT optimizer (BertOnnxModel).
# - onnxruntime/python/tools/transformers/onnx_model_bert.py:131 fuses RoPE via FusionRotaryEmbeddings.
# - But the BERT optimizer has no fuser for alternating sliding-window / local attention (zero hits for
#   sliding_window/local_attention/window_attention in onnxruntime's fusion_*.py) and GeGLU FFN (zero hits for
#   geglu/swiglu).

import logging
import os.path
import subprocess
import tempfile
import time
from datetime import datetime

import numpy as np
import polars as pl
import torch
from accelerate.utils import release_memory
from tap import tapify
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def _encode_timed(model: gt.utils.SentenceTransformer, texts: list[str], desc: str) -> list[float]:
    times: list[float] = []
    for text in tqdm(texts, desc=desc):
        start = time.monotonic()
        _ = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        times.append(time.monotonic() - start)
    return times


def _num_tokens(tokenizer: PreTrainedTokenizerBase, text: str, max_seq_length: int) -> int:
    return len(tokenizer(text, truncation=True, max_length=max_seq_length)["input_ids"])


def main(
    run_gcs_dir: str,
    text_prefix: str = "",
    df_path: str = "final_csvs/test_full2.csv",
    sample_size: int | None = None,
    does_not_support_sdpa: bool = False,
    max_seq_length: int = 8192,
):
    """
    Always runs on a GPU — the compiled model path requires CUDA. If CUDA is not locally
    available, this auto-launches an L4 and re-runs the same invocation there.

    Parameters
    ----------
    run_gcs_dir
        GCS path to the training run directory, e.g., gs://grouping-data/runs/my-run
    text_prefix
        String to prepend to every text before tokenization, e.g., for lightonai/modernbert-embed-large "clustering: "
    df_path
        Path to the validation/test CSV file. Only the query_stacktrace_string column is used.
    sample_size
        Number of deduped queries to benchmark. None (default) uses all deduped queries. Useful for smoke tests.
    does_not_support_sdpa
        If True, skip bfloat16 and SDPA attention for models that don't support it.
    max_seq_length
        Maximum sequence length to tokenize. Based on prod.
    """
    if not torch.cuda.is_available() and not gt.launch.is_on_remote():
        gt.launch.remote("l4")
        return

    gt.logging.configure_logging(process_type="benchmark_compiled")

    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_gcs_dir = run_gcs_dir.rstrip("/")
    run_id = os.path.basename(run_gcs_dir)
    path_gcs_inference = f"{run_gcs_dir}/inference"
    name_dataset = os.path.splitext(os.path.basename(df_path))[0]
    dir_gcs_output = f"gs://grouping-data/perf/{stamp}-{run_id}/{name_dataset}"

    df = gt.data.load_val_df(paths=(df_path,))
    texts = df.select("query_stacktrace_string").unique().to_series().to_list()
    if sample_size is not None and sample_size < len(texts):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(texts), size=sample_size, replace=False)
        texts = [texts[int(i)] for i in indices]
    logger.info(f"Benchmarking {len(texts)} unique query stacktraces")

    model_kwargs = {}
    if not does_not_support_sdpa and torch.cuda.is_bf16_supported():
        model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")

    with tempfile.TemporaryDirectory() as dir_tmp:
        logger.info(f"Downloading model from {path_gcs_inference} ...")
        subprocess.run(["gcloud", "storage", "rsync", "-r", path_gcs_inference, dir_tmp], check=True)

        logger.info("Loading compiled model")
        start = time.monotonic()
        model_compiled = gt.compiled.SentenceTransformer(
            dir_tmp, trust_remote_code=True, model_kwargs=model_kwargs, text_prefix=text_prefix
        )
        model_compiled.compile_and_warm_up()
        logger.info(f"Compiled model ready in {time.monotonic() - start:.1f}s")

        num_tokens = [_num_tokens(model_compiled.tokenizer, text, max_seq_length) for text in texts]

        times_compiled = _encode_timed(model_compiled, texts, desc="compiled")
        (model_compiled,) = release_memory(model_compiled)

        logger.info("Loading base model")
        start = time.monotonic()
        model_base = gt.utils.SentenceTransformer(
            dir_tmp, trust_remote_code=True, model_kwargs=model_kwargs, text_prefix=text_prefix
        )
        _ = model_base.encode("warm up")
        logger.info(f"Base model ready in {time.monotonic() - start:.1f}s")

        times_base = _encode_timed(model_base, texts, desc="base")
        (model_base,) = release_memory(model_base)

    df_out = pl.DataFrame(
        {
            "query_stacktrace_string": texts,
            "num_tokens": num_tokens,
            "time_compiled_sec": times_compiled,
            "time_base_sec": times_base,
        }
    )

    median_compiled_ms = float(np.median(times_compiled)) * 1000
    median_base_ms = float(np.median(times_base)) * 1000
    logger.info(
        f"Median compiled={median_compiled_ms:.1f}ms  "
        f"base={median_base_ms:.1f}ms  "
        f"speedup={median_base_ms / median_compiled_ms:.2f}x"
    )

    with tempfile.TemporaryDirectory() as dir_out:
        df_out.write_csv(f"{dir_out}/times.csv")
        with open(f"{dir_out}/run.txt", "w") as f:
            f.write(
                f"run_gcs_dir={run_gcs_dir}\n"
                f"df_path={df_path}\n"
                f"stamp={stamp}\n"
                f"sample_size={len(texts)}\n"
                f"text_prefix={text_prefix!r}\n"
                f"model_kwargs={model_kwargs}\n"
            )

        logger.info(f"Uploading to {dir_gcs_output} ...")
        subprocess.run(["gcloud", "storage", "rsync", "-r", dir_out, dir_gcs_output], check=True)
        logger.info(f"Uploaded to {dir_gcs_output}")


if __name__ == "__main__":
    tapify(main, description=__doc__)
