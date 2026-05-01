"""
All these functions assume you ran:

```bash
# from the repo root
cd grouping/data

gcloud storage cp -r "gs://grouping-data/dataset/org_{org_id}/project_{project_id}/*" .
```

and are inside `grouping/data`.
"""

import gc
from functools import wraps
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    from IPython.display import display
except ImportError:
    display = print
import polars as pl
import torch
from polars._typing import ConcatMethod
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer as SentenceTransformerOriginal
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase

SEER_THRESHOLD = 0.01
"Current grouping threshold in prod for the unfinetuned grouping model"

_CUSTOM_ID_DELIMETER = "|"
"Custom ID used for Anthropic batch API"


def as_dataset_dir(org_id: int, project_id: int, root: str = "dataset") -> Path:
    return Path(root) / f"org_{org_id}" / f"project_{project_id}"


def as_labeled_dir(org_id: int, project_id: int) -> Path:
    """
    Directory where labeled outputs (model decisions for stacktrace pairs) are stored.
    """
    return as_dataset_dir(org_id, project_id, root="dataset") / "labeled"


def read_stacktrace_strings(org_id: int, project_id: int) -> dict[str, str]:
    """
    Returns a dict mapping event ID to stacktrace string.
    """
    event_id_to_stacktrace_string = {}
    stacktrace_strings_paths = (as_dataset_dir(org_id, project_id, root="dataset") / "stacktrace_strings").glob("*.txt")
    for stacktrace_path in tqdm(list(stacktrace_strings_paths), desc="Reading stacktrace strings", disable=True):
        with open(stacktrace_path, "r") as f:
            stacktrace_string = f.read()
        event_id = stacktrace_path.stem
        event_id_to_stacktrace_string[event_id] = stacktrace_string
    return event_id_to_stacktrace_string


def concat_vertical_unordered(
    dfs: Iterable[pl.DataFrame],
    how: ConcatMethod = "vertical",
    rechunk: bool = False,
    parallel: bool = True,
) -> pl.DataFrame:
    """
    Polars doesn't have a mode for vertical concatenation when the same columns are in different orders.
    """
    dfs_iter = iter(dfs)
    df_first = next(dfs_iter)
    columns = set(df_first.columns)
    dfs_ordered = [df_first]
    for df in dfs_iter:
        if columns != set(df.columns):
            raise ValueError(f"Columns are not the same: {sorted(columns)} != {sorted(df.columns)}")
        dfs_ordered.append(df.select(df_first.columns))
    return pl.concat(dfs_ordered, how=how, rechunk=rechunk, parallel=parallel)


def deduplicate_pairs(
    df: pl.DataFrame,
    column1: str = "query_stacktrace_string",
    column2: str = "candidate_stacktrace_string",
) -> pl.DataFrame:
    """
    Keeps the first occurrence of each pair of `column1` and `column2`, even if their values appear as `column2` and
    `column1`.

    Grouping is symmetric.
    """
    assert "_pair_first" not in df.columns and "_pair_second" not in df.columns, (
        "input df must not have columns named '_pair_first' or '_pair_second' (used as scratch)"
    )
    return (
        df.with_columns(
            _pair_first=pl.min_horizontal(column1, column2),
            _pair_second=pl.max_horizontal(column1, column2),
        )
        .unique(subset=["_pair_first", "_pair_second"], keep="first", maintain_order=True)
        .drop(["_pair_first", "_pair_second"])
        .select(df.columns)
    )


def read_event_pairs(
    org_id: int,
    project_id: int,
    maybe_missing_int_columns: Sequence[str] = (
        "query_seer_gr_id",
        "candidate_seer_gr_id",
    ),
) -> pl.DataFrame:
    return concat_vertical_unordered(
        (
            pl.read_csv(csv_path).with_columns(
                source=pl.lit(csv_path.stem),
                path=pl.lit(str(csv_path)),
            )
            for pairs_path in as_dataset_dir(org_id, project_id, root="dataset").glob("2025-*")
            for csv_path in pairs_path.glob("*.csv")
        ),
        how="vertical_relaxed",
        # The how hack is temporarily used to allow concatenation. Some columns are null => typed as str when saved
        # instead of int
    ).with_columns(*(pl.col(col).cast(pl.Int64) for col in maybe_missing_int_columns))


def clean_matches(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensures "matched" and "unmatched" pairs reflect whether or not they're grouped together in prod.
    """
    return df.filter(
        ((pl.col("source") == "matched").and_(pl.col("query_group_id") == pl.col("candidate_group_id"))).or_(
            (pl.col("source") == "unmatched").and_(pl.col("query_group_id") != pl.col("candidate_group_id"))
        )
    )


def read(org_id: int, project_id: int, clean: bool = True) -> pl.DataFrame:
    """
    If `clean=True`, then:
    - only keep pairs whose "matched" and "unmatched" reflect whether or not they're grouped together in prod.
    - remove rows with empty, whitespace-only, or null stacktrace strings
    - deduplicate rows w/ the same `query_stacktrace_string` and `candidate_stacktrace_string`.

    Returns a df w/ these columns:
    - `query_seer_event_sent`
    - `candidate_seer_event_sent`
    - `distance`
    - `query_group_id`
    - `candidate_group_id`
    - `query_hash`
    - `candidate_hash`
    - `query_grouphash_id`
    - `candidate_grouphash_id`
    - `query_grouphashmetadata_id`
    - `candidate_grouphashmetadata_id`
    - `query_seer_gr_id`
    - `candidate_seer_gr_id`
    - `query_error_type`
    - `candidate_error_type`
    - `project_id`
    - `platform`
    - `source`
    - `path`
    - `query_stacktrace_string`
    - `candidate_stacktrace_string`
    """
    df_stacktrace_strings = pl.DataFrame(
        [
            {"event_id": event_id, "stacktrace_string": stacktrace_string}
            for event_id, stacktrace_string in read_stacktrace_strings(org_id, project_id).items()
        ],
        schema={"event_id": pl.String, "stacktrace_string": pl.String},
        # Manually set so that polars makes an empty dataframe if there are no stacktraces
    )
    df = (
        read_event_pairs(org_id, project_id)
        .join(
            df_stacktrace_strings,
            left_on="query_seer_event_sent",
            right_on="event_id",
            how="inner",
        )
        .join(
            df_stacktrace_strings,
            left_on="candidate_seer_event_sent",
            right_on="event_id",
            how="inner",
        )
        .rename(
            {
                "stacktrace_string": "query_stacktrace_string",
                "stacktrace_string_right": "candidate_stacktrace_string",
            }
        )
    )
    if clean:
        df = clean_matches(df)
        df = df.filter(
            (pl.col("query_stacktrace_string").is_not_null())
            & (pl.col("query_stacktrace_string").str.strip_chars() != "")
            & (pl.col("candidate_stacktrace_string").is_not_null())
            & (pl.col("candidate_stacktrace_string").str.strip_chars() != "")
        )
        df = df.unique(
            subset=["query_stacktrace_string", "candidate_stacktrace_string"],
            keep="first",
        )

        # Sanity check: unique query-candidate stacktraces should imply unique query-candidate event pairs (not
        # necessarily the other way around b/c different events can have the same stacktraces)
        query_candidate_event_pairs: list[tuple[str, str]] = list(
            zip(df["query_seer_event_sent"], df["candidate_seer_event_sent"])
        )
        if len(query_candidate_event_pairs) != len(set(query_candidate_event_pairs)):
            raise ValueError("Duplicate query-candidate event pairs")

        for query_event_id, candidate_event_id in query_candidate_event_pairs:
            assert _CUSTOM_ID_DELIMETER not in query_event_id
            assert _CUSTOM_ID_DELIMETER not in candidate_event_id

    return df


def all_project_paths(root: str = "dataset"):
    return (project_path for org_path in Path(root).glob("org_*") for project_path in org_path.glob("project_*"))


def parse_org_id_and_project_id(project_path: Path) -> tuple[int, int]:
    org_id = int(project_path.parts[1].split("_")[1])
    project_id = int(project_path.stem.split("_")[1])
    return org_id, project_id


def read_all(clean: bool = True):
    """
    If `clean=True`, then:
    - only keep pairs whose "matched" and "unmatched" reflect whether or not they're grouped together in prod.
    - remove rows with empty, whitespace-only, or null stacktrace strings
    - deduplicate rows w/ the same `query_stacktrace_string` and `candidate_stacktrace_string`.

    Yields dfs w/ these columns:
    - `query_seer_event_sent`
    - `candidate_seer_event_sent`
    - `distance`
    - `query_group_id`
    - `candidate_group_id`
    - `query_hash`
    - `candidate_hash`
    - `query_grouphash_id`
    - `candidate_grouphash_id`
    - `query_grouphashmetadata_id`
    - `candidate_grouphashmetadata_id`
    - `query_seer_gr_id`
    - `candidate_seer_gr_id`
    - `query_error_type`
    - `candidate_error_type`
    - `project_id`
    - `platform`
    - `source`
    - `path`
    - `query_stacktrace_string`
    - `candidate_stacktrace_string`
    """
    for project_path in all_project_paths():
        org_id, project_id = parse_org_id_and_project_id(project_path)
        df = read(org_id, project_id, clean=clean)
        yield project_path, df


def big_print_df(tbl_hide_column_data_types: bool = False, tbl_hide_dataframe_shape: bool = False):
    return pl.Config(
        tbl_rows=-1,
        fmt_str_lengths=1000,
        tbl_width_chars=1000,
        tbl_hide_column_data_types=tbl_hide_column_data_types,
        tbl_hide_dataframe_shape=tbl_hide_dataframe_shape,
    )


def stratify_round_robin(df: pl.DataFrame, group_name: str, target_num_rows: int) -> pl.DataFrame:
    groups = (list(df_group.rows(named=True)) for _, df_group in df.group_by(group_name, maintain_order=True))
    records = []
    for records_across_groups in zip_longest(*groups, fillvalue=None):
        for record in records_across_groups:
            if record is not None:
                records.append(record)
                if len(records) == target_num_rows:
                    break
        else:
            continue
        break

    return pl.DataFrame(records)


def stratify(
    df_projects: pl.DataFrame,
    target_num_rows: int,
    group_name: str = "platform",
    sort_by: str | None = None,
) -> pl.DataFrame:
    print("Preview")
    with pl.Config(tbl_rows=-1):
        display(df_projects.head(10))
    print()

    print("Before stratification")
    with pl.Config(tbl_rows=-1):
        display(df_projects.group_by(group_name).len().sort("len", descending=True))
    print()

    print("After stratification")
    # Bottom-up: keep the least frequent groups, and sort by sort_by to break ties
    # w/in each group
    more_by = (sort_by,) if sort_by is not None else ()
    df_projects = (
        df_projects.join(
            other=(df_projects[group_name].value_counts(sort=True).reverse().rename({"count": "_group_count"})),
            on=group_name,
        )
        .sort("_group_count", *more_by, descending=False)
        .drop("_group_count")
    )

    df_projects_sample = stratify_round_robin(df_projects, group_name, target_num_rows)
    with pl.Config(tbl_rows=-1):
        display(df_projects_sample.group_by(group_name).len().sort("len", group_name, descending=True))
    print()

    print("Projects final sample")
    with pl.Config(tbl_rows=-1):
        display(df_projects_sample.sort(group_name))
    print()

    print(f"{len(df_projects_sample)} project IDs:")
    print(*df_projects_sample["project_id"], sep="\n")

    return df_projects_sample


class CustomId(BaseModel):
    org_id: int
    project_id: int
    query_seer_event_sent: str  # UUID. don't wanna validate and risk weird things tho
    candidate_seer_event_sent: str  # UUID

    def to_string(self) -> str:
        return _CUSTOM_ID_DELIMETER.join(
            (
                f"org_{self.org_id}",
                f"proj_{self.project_id}",
                f"query_seer_event_sent_{self.query_seer_event_sent}",
                f"candidate_seer_event_sent_{self.candidate_seer_event_sent}",
            )
        )

    @classmethod
    def from_string(cls, custom_id: str) -> "CustomId":
        org, project, query_seer_event_sent, candidate_seer_event_sent = custom_id.split(_CUSTOM_ID_DELIMETER)
        return cls(
            org_id=int(org.removeprefix("org_")),
            project_id=int(project.removeprefix("proj_")),
            query_seer_event_sent=query_seer_event_sent.removeprefix("query_seer_event_sent_"),
            candidate_seer_event_sent=candidate_seer_event_sent.removeprefix("candidate_seer_event_sent_"),
        )


def _read_csvs_from_dir(dir_path: Path, glob_pattern: str = "*.csv") -> pl.DataFrame:
    csv_paths = sorted(dir_path.rglob(glob_pattern))
    if not csv_paths:
        return pl.DataFrame()
    dfs = (pl.read_csv(path) for path in csv_paths)
    return concat_vertical_unordered(dfs, how="vertical_relaxed")


def _read_all_projects(
    read_org_projct: Callable[[int, int], pl.DataFrame],
    root: str,
    desc: str = "Reading projects",
):
    project_paths = list(all_project_paths(root=root))
    for project_path in tqdm(project_paths, desc=desc):
        org_id, project_id = parse_org_id_and_project_id(project_path)
        df = read_org_projct(org_id, project_id)
        if df.is_empty():
            continue
        yield project_path, df


def _generate_dfs(read_projects: Iterable[tuple[Path, pl.DataFrame]]):
    for project_path, df in read_projects:
        org_id, project_id = parse_org_id_and_project_id(project_path)
        yield df.with_columns(org_id=pl.lit(org_id), project_id=pl.lit(project_id))


def read_labeled(org_id: int, project_id: int) -> pl.DataFrame:
    return _read_csvs_from_dir(as_labeled_dir(org_id, project_id), "batch_*.csv")


def read_labeled_all():
    yield from _read_all_projects(read_labeled, root="dataset", desc="Reading projects")


def generate_labeled_dfs():
    yield from _generate_dfs(read_labeled_all())


def read_synthetic(org_id: int, project_id: int, root: str = "dataset_augmented") -> pl.DataFrame:
    synthetic_dir = as_dataset_dir(org_id, project_id, root=root) / "synthetic"
    return _read_csvs_from_dir(synthetic_dir)


def read_all_synthetic(root: str = "dataset_augmented"):
    yield from _read_all_projects(
        lambda org_id, project_id: read_synthetic(org_id, project_id, root),
        root=root,
        desc="Reading synthetic",
    )


def generate_dfs_synthetic(root: str = "dataset_augmented"):
    yield from _generate_dfs(read_all_synthetic(root))


def _cuda_empty_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _retry_cuda_errors_once(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "CUDA" not in str(e):
                raise
            _cuda_empty_cache()
            return func(*args, **kwargs)

    return wrapper


class SentenceTransformer(SentenceTransformerOriginal):
    """
    `SentenceTransformer` which deduplicates texts during inference and retries OOMs once.
    """

    def __init__(self, *args, text_prefix: str = "", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text_prefix = text_prefix

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return super().tokenizer

    @tokenizer.setter
    def tokenizer(self, value: PreTrainedTokenizerBase) -> None:
        self._first_module().tokenizer = value

    # The getter and setter above are just for type hints. SentenceTransformer annotates it as Any

    def tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], **kwargs) -> dict[str, torch.Tensor]:
        if self.text_prefix:
            if isinstance(texts, list) and all(isinstance(text, str) for text in texts):
                texts = [self.text_prefix + text for text in texts]
            else:
                raise ValueError(f"Not sure how to add the prefix for the input text type: {type(texts)}")
        return super().tokenize(texts, **kwargs)

    @_retry_cuda_errors_once
    def encode(self, texts: str | list[str], **kwargs):
        if isinstance(texts, str):
            return super().encode(texts, **kwargs)

        unique = list(dict.fromkeys(texts))  # preserve order
        text_to_idx = {text: idx for idx, text in enumerate(unique)}
        embeddings = super().encode(unique, **kwargs)
        return embeddings[[text_to_idx[text] for text in texts]]  # assume numpy or torch


def encoder_from_base(base_model: str, use_text_prefix: bool = True) -> SentenceTransformer:
    """
    Build a SentenceTransformer encoder with standard dtype/attention settings.

    Handles model-specific quirks (e.g. jina v5's config_kwargs and trust_remote_code) and enables bfloat16 + SDPA when
    CUDA supports it.
    """
    if base_model == "jinaai/jina-embeddings-v5-text-nano-text-matching":
        return SentenceTransformer(
            base_model,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
            config_kwargs={"_attn_implementation": "sdpa"},
        )

    model_kwargs = None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa")

    text_prefix = ""
    if base_model == "lightonai/modernbert-embed-large" and use_text_prefix:
        # https://huggingface.co/lightonai/modernbert-embed-large#usage
        text_prefix = "clustering: "

    return SentenceTransformer(
        base_model,
        model_kwargs=model_kwargs,
        text_prefix=text_prefix,
    )
