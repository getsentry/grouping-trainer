"""Export save_embeddings output into DB-ready CSVs.

Takes an output directory from save_embeddings.py and saves, inside <output_dir>/export_for_db/:
  - grouping_records_db.csv: one row per unique (project_id, candidate_hash), with embeddings
    formatted as pgvector strings. Schema matches DbGroupingRecord.
  - query_stacktraces.csv: unique query stacktrace strings with all query-side metadata.
  - candidate_stacktraces.csv: unique candidate stacktrace strings with all candidate-side metadata.

Usage:
    python export_for_db.py <output_dir>
    python export_for_db.py similarities/2026-02-26-12-00-00-val-and-test
"""

import json
import os
import sys

import numpy as np
import polars as pl

# model name (from model_configs.json) -> DB column name
MODEL_NAME_TO_DB_COLUMN = {
    "prod": "stacktrace_embedding",
    "gte-finetuned": "stacktrace_embedding_v2_short",
}

# DB embedding columns not populated by any model
EMPTY_DB_COLUMNS = ["stacktrace_embedding_v2"]


def _embedding_to_pgvector(emb: np.ndarray) -> str:
    """Format a 1D numpy array as a pgvector string: [0.123,0.456,...]"""
    return "[" + ",".join(f"{x:.6g}" for x in emb) + "]"


def main(output_dir: str) -> None:
    export_dir = f"{output_dir}/export_for_db"
    os.makedirs(export_dir, exist_ok=True)
    print(f"Loading from {output_dir}/")
    print(f"Writing to {export_dir}/")

    df = pl.read_csv(f"{output_dir}/similarities.csv")

    with open(f"{output_dir}/model_configs.json") as f:
        model_configs = json.load(f)
    model_names = [mc["name"] for mc in model_configs["model_configs"]]

    unknown_models = set(model_names) - set(MODEL_NAME_TO_DB_COLUMN)
    if unknown_models:
        raise ValueError(f"Unknown model names {unknown_models}, add them to MODEL_NAME_TO_DB_COLUMN")

    candidate_embs_by_model = {}
    for name in model_names:
        path = f"{output_dir}/{name}_candidate_embeddings.npy"
        candidate_embs_by_model[name] = np.load(path)
        print(f"  {name} candidate embeddings: {candidate_embs_by_model[name].shape}")

    print(f"  similarities.csv: {len(df)} rows")

    # --- grouping_records_db.csv ---
    # Add row index to track back into .npy files
    df = df.with_row_index("_row_idx")

    # Dedup candidates by (project_id, candidate_hash), keeping first occurrence
    candidate_deduped = df.unique(subset=["project_id", "candidate_hash"], keep="first", maintain_order=True)
    row_indices = candidate_deduped["_row_idx"].to_numpy()
    print(f"\n  Unique (project_id, candidate_hash): {len(candidate_deduped)} (from {len(df)} pairs)")

    # Extract embeddings for the kept rows and format as pgvector strings
    embedding_columns = {}
    for model_name, db_column in MODEL_NAME_TO_DB_COLUMN.items():
        embs = candidate_embs_by_model[model_name][row_indices]
        embedding_columns[db_column] = [_embedding_to_pgvector(emb) for emb in embs]
    for db_column in EMPTY_DB_COLUMNS:
        embedding_columns[db_column] = None

    db_records = pl.DataFrame(
        {
            "id": candidate_deduped["candidate_seer_gr_id"],
            "project_id": candidate_deduped["project_id"],
            "hash": candidate_deduped["candidate_hash"],
            "error_type": candidate_deduped["candidate_error_type"],
            **embedding_columns,
        }
    )
    db_records = db_records.select(
        [
            "id",
            "project_id",
            "hash",
            "error_type",
            "stacktrace_embedding",
            "stacktrace_embedding_v2",
            "stacktrace_embedding_v2_short",
        ]
    )
    db_records.write_csv(f"{export_dir}/grouping_records_db.csv")
    print(f"  Saved {export_dir}/grouping_records_db.csv ({len(db_records)} rows)")

    # --- query_stacktraces.csv ---
    other_cols = [
        "project_id",
        "platform",
        "org_id",
        "distance",
        "source",
        "path",
        "label",
        "thinking_output",
        "response_output",
        "confidence_score",
        "prompt",
    ]
    query_cols = [c for c in df.columns if c.startswith("query_")] + other_cols
    query_df = df.select(query_cols).unique(subset=["query_stacktrace_string"], keep="first", maintain_order=True)
    query_df.write_csv(f"{export_dir}/query_stacktraces.csv")
    print(f"  Saved {export_dir}/query_stacktraces.csv ({len(query_df)} unique query strings)")

    # --- candidate_stacktraces.csv ---
    candidate_cols = [c for c in df.columns if c.startswith("candidate_")] + other_cols
    candidate_df = df.select(candidate_cols).unique(
        subset=["candidate_stacktrace_string"], keep="first", maintain_order=True
    )
    candidate_df.write_csv(f"{export_dir}/candidate_stacktraces.csv")
    print(f"  Saved {export_dir}/candidate_stacktraces.csv ({len(candidate_df)} unique candidate strings)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <output_dir>")
        sys.exit(1)
    main(sys.argv[1])
