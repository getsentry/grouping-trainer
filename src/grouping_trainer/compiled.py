from typing import Callable
import logging

import torch
import torch.nn.functional as F

import grouping_trainer as gt

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


class SentenceTransformer(gt.utils.SentenceTransformer):
    """
    Python is too slow for small models w/ batch size 1. Rm its overhead by compiling.
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
        self._buckets = tuple(sorted({bucket for bucket in compiled_token_buckets if bucket <= self.max_seq_length}))
        self._compiled_forward: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None

    def tokenize(self, texts: list[str], **kwargs) -> dict[str, torch.Tensor]:
        """
        Pads tokens to the nearest bucket so encode calls use a pre-compiled CUDA graph.
        """
        encodings = super().tokenize(texts, **kwargs)
        current_len = encodings["input_ids"].shape[1]
        target_len = current_len
        for bucket in self._buckets:
            if bucket >= current_len:
                target_len = bucket
                break

        if target_len > current_len:
            num_padding_tokens = target_len - current_len
            if extra_keys := (set(encodings.keys()) - {"input_ids", "attention_mask", "token_type_ids"}):
                raise ValueError(f"Unexpected encoding keys: {extra_keys}")

            encodings["input_ids"] = F.pad(
                encodings["input_ids"], (0, num_padding_tokens), value=self.tokenizer.pad_token_id
            )
            encodings["attention_mask"] = F.pad(encodings["attention_mask"], (0, num_padding_tokens), value=0)
            if "token_type_ids" in encodings:
                encodings["token_type_ids"] = F.pad(encodings["token_type_ids"], (0, num_padding_tokens), value=0)

        return encodings

    def warm_up_and_compile(self):
        self._compiled_forward = torch.compile(super().forward, mode="reduce-overhead", dynamic=False)
        self.eval()

        for target_length in self._buckets:
            if target_length > self.max_seq_length:
                continue

            num_prefix_tokens = len(self.tokenizer.tokenize(self.prompt_prefix)) if self.prompt_prefix else 0
            num_words_to_hit_target_length = (
                target_length - self.tokenizer.num_special_tokens_to_add(pair=False) - num_prefix_tokens
            )
            # For BERT: [CLS]...[SEP]
            text = "a " * num_words_to_hit_target_length
            texts = [text] * self._compiled_batch_size

            # Check correctness here to avoid silent performance regressions.
            # There are other approaches like creating the encoding ourselves, padding to the target length, and calling
            # .forward() (under inference_mode) ourselves. This approach didn't perform well—maybe b/c of subtle
            # differences in how .encode works. I prefer going through .encode and being loud about missing the target.
            # To debug that other approach, can check which guards fail using TORCH_LOGS="recompiles" in a non-prod
            # env.
            if super().tokenize(texts)["input_ids"].shape[1] != target_length:
                raise ValueError(f"Tokenization failed for {target_length=}")

            logger.info(f"Warming up for {target_length=}")

            for _ in range(4):
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
                _ = self.encode(texts, show_progress_bar=False)

    def forward(self, input: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """
        Only use the compiled forward if the sequence length matches one of our buckets. If we used the compiled forward
        for one that doesn't hit the bucket, we create a new CUDA graph for every unique sequence length above
        2048, which thrashes the cache.
        """
        if self.training:
            raise ValueError("This won't work for training.")

        batch_size, seq_length = input["input_ids"].shape
        if batch_size == self._compiled_batch_size and seq_length in self._buckets:
            return self._compiled_forward(input, **kwargs)
        return super().forward(input, **kwargs)
