import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F

import grouping_trainer as gt

logger = logging.getLogger(__name__)

_ForwardFunction = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]


@contextmanager
def _set_float32_matmul_precision(precision: Literal["highest", "high", "medium"]):
    current_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(current_precision)


class SentenceTransformer(gt.utils.SentenceTransformer):
    """
    Python is too slow for small models w/ batch size 1. Rm its overhead by compiling. 1.5-3x speedup for our models.
    Cost: warming up can take minutes.
    """

    def __init__(
        self,
        *args,
        compiled_batch_size: int = 1,
        compiled_token_buckets: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Must be able to pad to use pre-compiled forward")

        self._compiled_batch_size = compiled_batch_size
        self._compiled_token_buckets = tuple(
            sorted({bucket for bucket in compiled_token_buckets if bucket <= self.max_seq_length})
        )
        self._compiled_forward: _ForwardFunction | None = None

    def tokenize(
        self, texts: list[str] | list[dict[Any, Any]] | list[tuple[str, str]], **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Pads tokens to the nearest bucket so encode calls use a pre-compiled CUDA graph.
        """
        encodings = super().tokenize(texts, **kwargs)
        batch_size, num_tokens = encodings["input_ids"].shape

        if batch_size != self._compiled_batch_size:
            logger.error(
                "Input batch size doesn't match the compiled batch size. You should generally only use the compiled "
                "model when the batch size is known beforehand.",
                extra={
                    "compiled_batch_size": self._compiled_batch_size,
                    "batch_size": batch_size,
                    "num_tokens": num_tokens,
                },
            )
            return encodings

        target_num_tokens = num_tokens
        for bucket in self._compiled_token_buckets:
            if bucket >= num_tokens:
                target_num_tokens = bucket
                break

        if target_num_tokens > num_tokens:
            num_padding_tokens = target_num_tokens - num_tokens
            if extra_keys := (set(encodings.keys()) - {"input_ids", "attention_mask", "token_type_ids"}):
                raise ValueError(f"Unexpected encoding keys: {extra_keys}")

            encodings["input_ids"] = F.pad(
                encodings["input_ids"], (0, num_padding_tokens), value=self.tokenizer.pad_token_id
            )
            encodings["attention_mask"] = F.pad(encodings["attention_mask"], (0, num_padding_tokens), value=0)
            if "token_type_ids" in encodings:
                encodings["token_type_ids"] = F.pad(encodings["token_type_ids"], (0, num_padding_tokens), value=0)

        return encodings

    def compile_and_warm_up(self):
        # This method isn't called in __init__ so that the caller can transfer the model to the target device before
        # warming up.

        self._compiled_forward = cast(
            _ForwardFunction,
            torch.compile(super().forward, mode="reduce-overhead", dynamic=False),
        )
        self.eval()

        for target_num_tokens in self._compiled_token_buckets:
            # Create dummy text which is exactly target_num_tokens long.
            num_words = target_num_tokens  # overestimate
            text = "a " * num_words
            num_tokens = super().tokenize([text])["input_ids"].shape[1]
            num_words -= num_tokens - target_num_tokens
            text = "a " * num_words
            texts = [text] * self._compiled_batch_size

            # Check correctness here to avoid silent performance regressions.
            # There are other approaches like creating the encoding ourselves, padding to the target length, and calling
            # .forward() (under inference_mode) ourselves. This approach didn't perform well—maybe b/c of subtle
            # differences in how .encode works. I prefer going through .encode and being loud about missing the target.
            # To debug that other approach, can check which guards fail using TORCH_LOGS="recompiles" in a non-prod
            # env.
            if super().tokenize(texts)["input_ids"].shape[1] != target_num_tokens:
                raise ValueError(f"Tokenization failed for {target_num_tokens=}")

            logger.info(f"Warming up for {target_num_tokens=}")

            for _ in range(4):
                _ = self.encode(texts, show_progress_bar=False)
                # Why repeat 4 times? See these docs:
                #
                # https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html
                # https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/torch-integration.html#stream-capture-api-torch-cuda-graph
                # https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
                #
                # Summary from Gemini 3.1 Pro:
                #
                # Run 1 (Dynamo/Inductor): PyTorch lowers the model to FX graphs, creates Triton kernels, and runs them
                # once. (This is the longest delay).
                #
                # Runs 2 & 3 (Memory Warmup): PyTorch runs the compiled kernels in "eager" mode on a side-stream. This
                # initializes cuBLAS/cuDNN workspaces and forces PyTorch's caching allocator to assign static memory
                # addresses for all intermediate tensors.
                #
                # Run 4 (Capture): PyTorch executes the code within a torch.cuda.graph(g) context. The GPU driver
                # records the exact sequence of kernel launches and memory pointers without actually executing the math.

    @_set_float32_matmul_precision("high")
    def forward(self, input: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        # Only use the compiled forward if the sequence length matches one of our buckets. If we used the compiled forward
        # for one that doesn't hit the bucket, we create a new CUDA graph for every unique sequence length above
        # 2048, which thrashes the cache.

        if self.training:
            raise ValueError("This won't work for training.")

        batch_size, seq_length = input["input_ids"].shape
        if batch_size == self._compiled_batch_size and seq_length in self._compiled_token_buckets:
            if self._compiled_forward is None:
                # Don't fall back to the non-compiled forward. There's no point using this class if it's not warmed up.
                # It'll pad and call the model on padded input for no reason.
                raise ValueError("compile_and_warm_up() must be called before using the compiled forward.")
            return self._compiled_forward(input, **kwargs)
            # model-related kwargs shouldn't be variable across calls
        return super().forward(input, **kwargs)
