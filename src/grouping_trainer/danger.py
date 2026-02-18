import torch
import torch.nn.functional as F

from grouping_trainer.utils import SentenceTransformer as SentenceTransformerGT


class SentenceTransformer(SentenceTransformerGT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buckets = [64, 128, 256, 512, 1024, 2048, 4096]

    def tokenize(self, texts: list[str], **kwargs) -> dict[str, torch.Tensor]:
        """
        Pads tokens to the nearest bucket so encode calls use a pre-compiled CUDA graph.
        """
        encodings = super().tokenize(texts, **kwargs)
        current_len = encodings["input_ids"].shape[1]
        target_len = current_len
        for bucket in self.buckets:
            if bucket >= current_len:
                target_len = bucket
                break

        if target_len > current_len:
            num_padding_tokens = target_len - current_len
            pad_val = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

            encodings["input_ids"] = F.pad(encodings["input_ids"], (0, num_padding_tokens), value=pad_val)
            encodings["attention_mask"] = F.pad(encodings["attention_mask"], (0, num_padding_tokens), value=0)
            if "token_type_ids" in encodings:
                encodings["token_type_ids"] = F.pad(encodings["token_type_ids"], (0, num_padding_tokens), value=0)

        return encodings

    def warmup_and_compile(self):
        self.forward = torch.compile(self.forward, mode="reduce-overhead")
        self.eval()

        for target_length in self.buckets:
            if target_length > self.max_seq_length:
                continue

            num_words = target_length - self.tokenizer.num_special_tokens_to_add(pair=False)
            # For BERT: [CLS]...[SEP]
            text = "a " * num_words

            try:
                _ = self.encode(text)
            except Exception as e:
                print(f"Bucket failed {target_length=}: {e}")
