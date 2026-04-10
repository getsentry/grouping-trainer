import torch
import torch.nn.functional as F

import grouping_trainer as gt

model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
base_model = "lightonai/modernbert-embed-large"

encoder = gt.utils.SentenceTransformer(base_model, model_kwargs=model_kwargs)
encoder_compiled = gt.compiled.SentenceTransformer(base_model, model_kwargs=model_kwargs)

for n, p in encoder.named_parameters():
    p_danger = encoder_compiled.get_parameter(n)
    assert torch.allclose(p, p_danger)

test_string = "test string here"
x = encoder.encode(test_string, convert_to_numpy=False, convert_to_tensor=True)
x_danger = encoder_compiled.encode(test_string, convert_to_numpy=False, convert_to_tensor=True)
if not torch.allclose(x, x_danger):
    print("x_danger is different")
    print(x[:20])
    print("-" * 100)
    print(x_danger[:20])

e = encoder.tokenize(test_string)
e_danger = encoder_compiled.tokenize(test_string)

assert torch.all(e_danger["input_ids"][:, :3] == e["input_ids"])
assert torch.all(e_danger["attention_mask"][:, :3] == e["attention_mask"])
assert torch.all(e_danger["input_ids"][:, 3:] == encoder.tokenizer.pad_token_id)
assert torch.all(e_danger["attention_mask"][:, 3:] == 0)

# Test if padding is the source of divergence.
# Can't monkey-patch tokenize (super() is class-bound), so override at the
# instance level with a simple wrapper that pads after the normal tokenize.

pad_to = 64
_orig_tokenize = encoder.tokenize


def _padded_tokenize(texts, **kwargs):
    enc = _orig_tokenize(texts, **kwargs)
    cur = enc["input_ids"].shape[1]
    if cur < pad_to:
        n = pad_to - cur
        enc["input_ids"] = F.pad(enc["input_ids"], (0, n), value=encoder.tokenizer.pad_token_id)
        enc["attention_mask"] = F.pad(enc["attention_mask"], (0, n), value=0)
        if "token_type_ids" in enc:
            enc["token_type_ids"] = F.pad(enc["token_type_ids"], (0, n), value=0)
    return enc


try:
    encoder.tokenize = _padded_tokenize
    x_padded = encoder.encode(test_string, convert_to_numpy=False, convert_to_tensor=True)
finally:
    encoder.tokenize = _orig_tokenize

print("padded vs danger allclose:", torch.allclose(x_padded, x_danger))
print("padded vs original allclose:", torch.allclose(x_padded, x))
if not torch.allclose(x_padded, x_danger):
    print("x_padded[:20]:", x_padded[:20])
    print("x_danger[:20]:", x_danger[:20])
assert torch.allclose(x_padded, x)
# No not caused by padding. x_padded is the same as x
