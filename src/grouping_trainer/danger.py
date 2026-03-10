from typing import Callable
import logging

import torch
import torch.nn.functional as F

import grouping_trainer as gt

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


class SentenceTransformer(gt.utils.SentenceTransformer):
    """
    Python is too slow for this small model and batch size 1. So compile.
    Cost: warming up can take 120 seconds for our data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._buckets = (64, 128, 256, 512, 1024, 2048)
        # The vast majority of stacktraces are in this range.
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
            pad_val = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

            if extra_keys := (set(encodings.keys()) - {"input_ids", "attention_mask", "token_type_ids"}):
                raise ValueError(f"Unexpected encoding keys: {extra_keys}")

            encodings["input_ids"] = F.pad(encodings["input_ids"], (0, num_padding_tokens), value=pad_val)
            encodings["attention_mask"] = F.pad(encodings["attention_mask"], (0, num_padding_tokens), value=0)
            if "token_type_ids" in encodings:
                encodings["token_type_ids"] = F.pad(encodings["token_type_ids"], (0, num_padding_tokens), value=0)

        return encodings

    def warmup_and_compile(self):
        self._compiled_forward = torch.compile(super().forward, mode="reduce-overhead", dynamic=False)
        self.eval()

        for target_length in self._buckets:
            if target_length > self.max_seq_length:
                continue

            num_words = target_length - self.tokenizer.num_special_tokens_to_add(pair=False)
            # For BERT: [CLS]...[SEP]
            text = "a " * num_words

            # Check correctness here to avoid silent performance regressions.
            # There are other approaches like creating the encoding ourselves, padding to the target length, and calling
            # .forward() (under inference_mode) ourselves. This approach didn't perform well—maybe b/c of subtle
            # differences in how .encode works. I prefer going through .encode and being loud about missing the target.
            # To debug that other approach, can check which guards fail using TORCH_LOGS="recompiles" in a non-prod
            # env.
            if self.tokenize([text])["input_ids"].shape[1] != target_length:
                raise ValueError(f"Tokenization failed for {target_length=}")

            logger.info(f"Warming up for {target_length=}")
            _ = self.encode(text, show_progress_bar=False)

    def forward(self, input: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """
        Only use the compiled forward if the sequence length matches one of our buckets. If we used the compiled forward
        for one that doesn't hit the bucket, we create a new CUDA graph for every unique sequence length above
        2048, which thrashes the cache.
        """
        seq_length = input["input_ids"].shape[1]
        if seq_length in self._buckets:
            return self._compiled_forward(input, **kwargs)
        return super().forward(input, **kwargs)
