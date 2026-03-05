import os
import json
import subprocess
from datetime import datetime
import time

import numpy as np
import polars as pl
from pydantic import BaseModel, field_serializer
from sentence_transformers.util import pairwise_cos_sim
import torch
from tqdm.auto import tqdm

import grouping_trainer as gt


torch.set_float32_matmul_precision("high")  # no impact it seems?


class ModelConfig(BaseModel):
    name: str
    path: str
    truncate_dim: int | None = None
    batch_size: int = 1
    model_kwargs: dict | None = None

    @field_serializer("model_kwargs")
    def serialize_model_kwargs(self, v: dict | None) -> dict:
        if v is None:
            return None
        return {k: str(val) if isinstance(val, torch.dtype) else val for k, val in v.items()}


class ModelConfigs(BaseModel):
    model_configs: list[ModelConfig]


class DataConfig(BaseModel):
    df_path: str
    sample_size: int | None = None


timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

RUN_SHORTNAME = "val-and-test"
DATA_CONFIG = DataConfig(
    df_path="final_csvs/val_and_test.csv",
    sample_size=None,
    # sample_size=100,
)
MODEL_CONFIGS = ModelConfigs(
    model_configs=[
        ModelConfig(
            name="gte-finetuned",
            path="gte-finetuned/training",
            truncate_dim=64,
            model_kwargs=dict(
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                # attn_implementation="flash_attention_2",  # hell
            ),
        ),
        ModelConfig(
            name="prod",
            path="issue_grouping_v1/embeddings",
            truncate_dim=None,
        ),
    ]
)
OUTPUT_DIR = f"./{timestamp}-{RUN_SHORTNAME}"


def encode_timed(
    model: gt.utils.SentenceTransformer, texts: list[str], progress_bar_desc: str | None = None
) -> tuple[np.ndarray, list[float]]:
    times: list[float] = []
    embeddings: list[np.ndarray] = []
    for text in tqdm(texts, desc=progress_bar_desc):
        start = time.monotonic()
        emb = model.encode(text, convert_to_numpy=True)
        end = time.monotonic()
        times.append(end - start)
        embeddings.append(emb)
    return np.array(embeddings), times


df = gt.data.load_val_df(path=DATA_CONFIG.df_path, sample_size=DATA_CONFIG.sample_size)
print(df.shape)
print(df.columns)

model_name_to_query_and_candidate_embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

for model_config in tqdm(MODEL_CONFIGS.model_configs, desc="Models"):
    print(model_config)

    if model_config.model_kwargs:
        st_class = gt.danger.SentenceTransformer
    else:
        st_class = gt.utils.SentenceTransformer

    print(f"Loading model {model_config.name} from {model_config.path}")
    start = time.monotonic()
    model = st_class(
        model_config.path,
        trust_remote_code=True,
        truncate_dim=model_config.truncate_dim,
        model_kwargs=model_config.model_kwargs,
    )
    end = time.monotonic()
    load_time = round(end - start, 1)
    print(f"Model loaded in {load_time} seconds.")

    if hasattr(model, "warmup_and_compile"):
        model.warmup_and_compile()
    else:
        _ = model.encode("warm up")
    end = time.monotonic()
    warm_up_time = round(end - start, 1)
    print(f"Warm up took {warm_up_time} seconds.")

    query_texts = df["query_stacktrace_string"].to_list()
    query_embeddings, query_times = encode_timed(model, query_texts, progress_bar_desc="Queries")

    candidate_texts = df["candidate_stacktrace_string"].to_list()
    candidate_embeddings, candidate_times = encode_timed(model, candidate_texts, progress_bar_desc="Candidates")

    model_name_to_query_and_candidate_embeddings[model_config.name] = (query_embeddings, candidate_embeddings)

    cos_sims = pairwise_cos_sim(query_embeddings, candidate_embeddings).detach().cpu().numpy()

    df = df.with_columns(
        [
            pl.Series(name=f"cos_sim_{model_config.name}", values=cos_sims),
            pl.Series(name=f"query_encode_time_{model_config.name}", values=query_times),
            pl.Series(name=f"candidate_encode_time_{model_config.name}", values=candidate_times),
        ]
    )
    print()

os.mkdir(OUTPUT_DIR)

with open(f"{OUTPUT_DIR}/model_configs.json", "w") as f:
    json.dump(MODEL_CONFIGS.model_dump(), f, indent=4)

with open(f"{OUTPUT_DIR}/data_config.json", "w") as f:
    json.dump(DATA_CONFIG.model_dump(), f, indent=4)

df.write_csv(f"{OUTPUT_DIR}/similarities.csv")
print(f"Saved similarities to {OUTPUT_DIR}/similarities.csv")

for model_name, (query_embs, candidate_embs) in model_name_to_query_and_candidate_embeddings.items():
    np.save(f"{OUTPUT_DIR}/{model_name}_query_embeddings.npy", query_embs)
    np.save(f"{OUTPUT_DIR}/{model_name}_candidate_embeddings.npy", candidate_embs)
    print(f"Saved embeddings for {model_name}: query {query_embs.shape}, candidate {candidate_embs.shape}")

# Upload to GCS
GCS_DIR = f"gs://grouping-data/runs/{OUTPUT_DIR.lstrip('./')}"
print(f"\nUploading to {GCS_DIR}...")
subprocess.run(["gsutil", "-m", "rsync", "-r", OUTPUT_DIR, GCS_DIR], check=True)
print(f"Uploaded to {GCS_DIR}")
