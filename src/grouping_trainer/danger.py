from typing import Callable
import torch
import torch.nn.functional as F

from grouping_trainer.utils import SentenceTransformer as SentenceTransformerGT


class SentenceTransformer(SentenceTransformerGT):
    """
    Python is too slow for this small model and batch size 1. Need to compile.
    Cost: warming up the cached graphs can take 120 seconds for our data.
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
        # dynamic=False forces Dynamo to build strictly static graphs for our buckets. Don't let it generalize shapes.
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

            _ = self.encode(text)

        # TODO: we could also end by encoding a 8192-length sequence to reserve a bunch of memory. Our model isn't that
        # big. There is very likely enough room left to reserve our max seq length after the cached graphs are filled. I
        # think this would be a pretty small optimization.

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
