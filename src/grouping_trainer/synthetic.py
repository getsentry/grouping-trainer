"""
Boue Pour Un Bébé Robot
"""

from pathlib import Path
from typing import Any, Literal, cast
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
import torch
from tqdm.auto import tqdm

from grouping_trainer import utils


@dataclass(frozen=True, kw_only=True)
class LabelResult:
    idx: int
    label: str
    confidence_score: float | None
    "`None` means the confidence score was not able to be parsed from `response_output`"
    response_output: str
    prompt: str
    thinking_output: str


def top_combos(
    distances: np.ndarray,
    min_distance: float,
    max_distance: float,
    num_combos: int | None = None,
) -> list[tuple[int, ...]]:
    """
    Return at most `num_combos` indices with distance at least `min_distance`, sorted by distance descending.
    `distances` can have any shape.

    The output of this function isn't too useful w/ a symmetric distance matrix.
    """
    if num_combos < 0:
        raise ValueError("num_combos must be positive")
    if num_combos == 0:
        return []

    flat = distances.ravel()
    mask = (min_distance <= distances) & (distances <= max_distance)

    if not np.any(mask):
        return []

    indices = np.where(mask)[0]  # in raveled space
    indices_desc_by_distance = np.argsort(flat[indices])[::-1]  # in sub-raveled space
    indices_selected = indices[indices_desc_by_distance]  # back to raveled space
    top_indices = indices_selected[:num_combos]  # still in raveled space
    unraveled = np.unravel_index(top_indices, distances.shape)  # finally unravel
    return list(zip(*unraveled))  # :-]


def mine_semi_easy_negatives_from_distance_matrix(
    query_candidate_distances: np.ndarray,
    min_distance: float = 0.3,  # not too hard
    max_distance: float = 0.5,  # not too easy
    num_candidates_per_query: int = 5,  # TODO(kddubey): make this 20. Can always sample down after writing it.
):
    """
    Returns the furthest `num_candidates_per_query` candidate indices per query matching the distance filters.
    The selection is stratified across queries to avoid overrepresenting universally distant candidates.

    Note
    ----

    `min_distance`
    - Too low => label noise / precision
    - Too high => low recall

    `max_distance`
    - Too low => low recall
    - Too high => too many, too easy negatives
    """
    if query_candidate_distances.ndim != 2:
        raise ValueError("query_candidate_distances must be a 2-D array")

    farthest_candidate_indices_per_query = [  # stratified
        top_combos(
            query_distances,
            min_distance=min_distance,
            max_distance=max_distance,
            num_combos=num_candidates_per_query,
        )
        for query_distances in query_candidate_distances
    ]

    # Check that we can slice the first dimension of the candidate axis
    for candidate_indices in farthest_candidate_indices_per_query:
        for candidate_index in candidate_indices:
            assert len(candidate_index) == 1

    # Polars doesn't support np.integer-slicing. Convenient to return plain ints
    return [
        [int(candidate_index[0]) for candidate_index in candidate_indices]
        for candidate_indices in farthest_candidate_indices_per_query
    ]


def record_from_pair(
    record_query: dict[str, Any],
    record_candidate: dict[str, Any],
    distance: float,
    source: str,
    synthetic_label: Literal["GROUP", "SEPARATE"],
    seer_threshold: float = utils.SEER_THRESHOLD,
):
    # GroupHash-specific info
    query_kv = {k: v for k, v in record_query.items() if k.startswith("query_")}
    candidate_kv = {k: v for k, v in record_candidate.items() if k.startswith("candidate_")}

    # Non-GroupHash, non-pair-specific info
    rest_kv = {k: v for k, v in record_query.items() if (k not in query_kv) and (k not in candidate_kv)}
    rest_kv["source"] = source

    # Pair-specific info we'll override
    rest_kv["distance"] = distance
    rest_kv["is_grouped"] = distance < seer_threshold

    # New label
    label_result_kv = asdict(
        LabelResult(  # should be a BaseModel instead...
            idx=0,
            label=synthetic_label,
            confidence_score=None,  # could be empirically estimated via LLM
            response_output="",
            prompt="",
            thinking_output="",
        )
    )
    _ = label_result_kv.pop("idx")

    return (
        query_kv | candidate_kv | rest_kv | label_result_kv  # last to override label result values in rest_kv
    )


def synthetic_df(
    df: pl.DataFrame,
    query_candidate_index_pairs: list[tuple[int, int]],
    distances_matrix: np.ndarray,
    source: str,
    synthetic_label: Literal["GROUP", "SEPARATE"],
):
    if not query_candidate_index_pairs:
        return df.clear()
    df = pl.DataFrame(
        [
            record_from_pair(
                df.row(query_idx, named=True),
                df.row(candidate_idx, named=True),
                distances_matrix[query_idx, candidate_idx],
                source,
                synthetic_label,
            )
            for query_idx, candidate_idx in query_candidate_index_pairs
        ],
        infer_schema_length=None,
    )
    return utils.deduplicate_pairs(df)


def encode_deduplicated(
    model: SentenceTransformer, queries: list[str], candidates: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    # Map texts to idxs
    texts_unique, inverse_indices = np.unique(queries + candidates, return_inverse=True)

    # Call model
    embeddings_unique = model.encode(
        cast(list[str], texts_unique.tolist()),
        batch_size=4,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Map embeddings back to queries and candidates
    all_embeddings = embeddings_unique[inverse_indices]
    num_queries = len(queries)
    query_embeddings = all_embeddings[:num_queries]
    candidate_embeddings = all_embeddings[num_queries:]

    return query_embeddings, candidate_embeddings


def mine_semi_easy_negatives(model: SentenceTransformer, df_project: pl.DataFrame) -> pl.DataFrame:
    df_project = df_project.sort("query_stacktrace_string")
    # Compute distances b/t all pairs
    query_embeddings, candidate_embeddings = encode_deduplicated(
        model,
        df_project["query_stacktrace_string"].to_list(),
        df_project["candidate_stacktrace_string"].to_list(),
    )
    query_candidate_cosine_distances = 1 - (query_embeddings @ candidate_embeddings.T)
    farthest_candidate_indices_per_query = mine_semi_easy_negatives_from_distance_matrix(
        query_candidate_cosine_distances
    )
    far_query_candidate_index_pairs = [
        (query_idx, candidate_idx)
        for query_idx, farthest_candidate_indices in enumerate(farthest_candidate_indices_per_query)
        for candidate_idx in farthest_candidate_indices
    ]
    return synthetic_df(
        df_project,
        far_query_candidate_index_pairs,
        query_candidate_cosine_distances,
        source="synthetic-negative-semi-easy",
        synthetic_label="SEPARATE",
    )


if __name__ == "__main__":
    model_path = Path("/Users/kdubey/projects/seer") / "models/issue_grouping_v1/embeddings"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(model_path), trust_remote_code=True)

    df = pl.read_csv("train.csv")
    df = df.sort(pl.col("query_stacktrace_string").str.len_chars().mean().over("org_id", "project_id"))
    for (org_id, project_id), df_project in tqdm(
        df.group_by("org_id", "project_id"), total=len(df["project_id"].unique())
    ):
        path_dir = Path("dataset_augmented") / f"org_{org_id}" / f"project_{project_id}" / "synthetic" / "negatives"
        output_path = path_dir / "semi-easy.csv"
        if output_path.exists():
            continue
        df_negatives = mine_semi_easy_negatives(model, df_project)
        path_dir.mkdir(parents=True, exist_ok=True)
        df_negatives.write_csv(output_path)
