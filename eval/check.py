import torch
import torch.nn.functional as F

import grouping_trainer as gt

model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")
# mkdir v3
# gcloud storage cp -r gs://grouping-data/runs/2026-04-07-11-56-28-large-con/inference/* v3
base_model = "v3"
prompt_prefix = "clustering: "

encoder = gt.utils.SentenceTransformer(
    base_model,
    trust_remote_code=True,
    model_kwargs=model_kwargs,
    prompt_prefix=prompt_prefix,
)
encoder_compiled = gt.compiled.SentenceTransformer(
    base_model,
    trust_remote_code=True,
    model_kwargs=model_kwargs,
    prompt_prefix=prompt_prefix,
)

for n, p in encoder.named_parameters():
    p_compiled = encoder_compiled.get_parameter(n)
    assert torch.allclose(p, p_compiled)

test_string = "test string here"
x = encoder.encode(test_string, convert_to_numpy=False, convert_to_tensor=True)
x_compiled = encoder_compiled.encode(test_string, convert_to_numpy=False, convert_to_tensor=True)
if not torch.allclose(x, x_compiled):
    print("x_compiled is different")
    print(x[:20])
    print("-" * 100)
    print(x_compiled[:20])

e = encoder.tokenize([test_string])
e_compiled = encoder_compiled.tokenize([test_string])

assert torch.all(e_compiled["input_ids"][:, :3] == e["input_ids"])
assert torch.all(e_compiled["attention_mask"][:, :3] == e["attention_mask"])
assert torch.all(e_compiled["input_ids"][:, 3:] == encoder.tokenizer.pad_token_id)
assert torch.all(e_compiled["attention_mask"][:, 3:] == 0)

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

print("padded vs danger allclose:", torch.allclose(x_padded, x_compiled))
print("padded vs original allclose:", torch.allclose(x_padded, x))
if not torch.allclose(x_padded, x_compiled):
    print("x_padded[:20]:", x_padded[:20])
    print("x_compiled[:20]:", x_compiled[:20])
assert torch.allclose(x_padded, x)
# No not caused by padding. x_padded is the same as x


def cos_sim(model: gt.utils.SentenceTransformer, text1: str, text2: str) -> torch.Tensor:
    x1 = model.encode(text1)
    x2 = model.encode(text2)
    return x1 @ x2


s1 = "test string here"
s2 = "something else here"

x_s1 = encoder.encode(s1)
x_s2 = encoder.encode(s2)
x_s1_compiled = encoder_compiled.encode(s1)
x_s2_compiled = encoder_compiled.encode(s2)

print("x_s1:", x_s1[:20])
print("x_s2:", x_s2[:20])
print("x_s1_compiled:", x_s1_compiled[:20])
print("x_s2_compiled:", x_s2_compiled[:20])

print("cos_sim_original:", torch.nn.functional.cosine_similarity(x_s1, x_s2, dim=-1))
print("cos_sim_compiled:", torch.nn.functional.cosine_similarity(x_s1_compiled, x_s2_compiled, dim=-1))

print("cos_sim_original:", cos_sim(encoder, s1, s2))
print("cos_sim_compiled:", cos_sim(encoder_compiled, s1, s2))
