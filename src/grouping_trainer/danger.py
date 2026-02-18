from sentence_transformers import SentenceTransformer as SentenceTransformerOriginal
import torch
import torch.nn.functional as F


class SentenceTransformer(SentenceTransformerOriginal):
    """
    This class was used for a kind of failed experiment where I wanted to see if we could
    get rid of Python overhead. It works in the long run (2x speedup) but the first 25
    calls to the model were too slow. Maybe something simple I'm missing re how I'm warming
    up. I also don't like overriding `tokenize`. Feel like it's easy to silently mess up if
    we use a different model. I tested correctness for the finetuned grouping model.
    """

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
        device = self.device

        for length in self.buckets:
            if length > self.max_seq_length:
                continue
            encodings = {
                "input_ids": torch.zeros((1, length), dtype=torch.long, device=device),
                "attention_mask": torch.ones((1, length), dtype=torch.long, device=device),
            }
            # BERT usually has this.
            if hasattr(self.tokenizer, "model_input_names") and "token_type_ids" in self.tokenizer.model_input_names:
                encodings["token_type_ids"] = torch.zeros((1, length), dtype=torch.long, device=device)
            try:
                with torch.inference_mode():
                    self.forward(encodings)
            except Exception as e:
                print(f"Bucket failed: {length=}: {e}")
