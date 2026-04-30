import itertools
import logging
import math
import os
import random
import subprocess
import threading
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Literal, TypedDict, overload

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
from accelerate import DistributedType
from datasets import Dataset, DatasetDict
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_model as safetensors_load_model
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.models import Pooling
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler, default_collate
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils.import_utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
)

import grouping_trainer as gt

logger = logging.getLogger(__name__)


class Record(TypedDict):
    query_stacktrace_string: str
    candidate_stacktrace_string: str
    label: int
    sample_weight: float
    confidence_score: float


class Batch(TypedDict):
    query_stacktrace_string: list[str]
    candidate_stacktrace_string: list[str]
    label: torch.Tensor
    sample_weight: torch.Tensor
    confidence_score: torch.Tensor


def _record_from_dict(record_dict: dict[str, Any]) -> Record:
    return Record(
        query_stacktrace_string=record_dict["query_stacktrace_string"],
        candidate_stacktrace_string=record_dict["candidate_stacktrace_string"],
        label=int(record_dict["label"] == "GROUP"),
        sample_weight=float(record_dict.get("sample_weight", 1.0)),
        confidence_score=float(record_dict.get("confidence_score", 1.0)),
        # NOTE: cast to float b/c polars could read the data as a string if there were nulls in the CSV
    )


def df_to_dataset(
    df: pl.DataFrame,
    group_by_query_stacktrace_string: bool = True,
    shuffle_groups: bool = True,
    seed: int | None = None,
    use_confidence_score: bool = False,
    confidence_score_floor: float = 0.9,
) -> Dataset:
    """
    Convert a DataFrame to a Dataset, grouping records by `query_stacktrace_string`.

    Records with the same `query_stacktrace_string` are kept together for cache hits in the forward pass. By default,
    the order of groups is randomized to avoid alphabetical ordering bias during training.
    """
    if not use_confidence_score:
        df = df.drop("confidence_score", strict=False)
    else:
        df = df.with_columns(
            pl.col("confidence_score").cast(pl.Float64).fill_null(1.0).clip(lower_bound=confidence_score_floor)
        )

    if not group_by_query_stacktrace_string:
        return Dataset.from_list([_record_from_dict(record_dict) for record_dict in df.rows(named=True)])

    query_group_dfs = [
        group_df.sort(pl.col("candidate_stacktrace_string").str.len_chars())
        for _, group_df in df.group_by("query_stacktrace_string")
    ]
    # Sort deterministically first b/c polars group_by returns groups in arbitrary order, and DDP requires all processes
    # to have the same dataset ordering.
    query_group_dfs.sort(key=lambda query_group_df: query_group_df["query_stacktrace_string"][0])
    if shuffle_groups:
        rng = random.Random(seed if seed is not None else 42)
        rng.shuffle(query_group_dfs)

    return Dataset.from_list(
        [
            _record_from_dict(record_dict)
            for query_group_df in query_group_dfs
            for record_dict in query_group_df.rows(named=True)
        ]
    )


def create_project_dataset_dict(
    df: pl.DataFrame,
    min_dataset_size: int | None = None,
    use_confidence_score: bool = False,
    confidence_score_floor: float = 0.9,
) -> DatasetDict:
    """
    Create a `DatasetDict` with one dataset per project. Projects below `min_dataset_size` are packed into a single
    dataset to avoid tiny batches. `min_dataset_size` can simply be set to the global/effective training batch size.
    """
    project_id_to_dataset: dict[str, Dataset] = {}
    small_project_dfs: list[pl.DataFrame] = []

    for (project_id,), df_project in tqdm(
        df.group_by("project_id"),
        total=len(df["project_id"].unique()),
        desc="Creating project datasets",
    ):
        project_id = str(project_id)
        # DatasetDict implements __getitem__ by accepting a mix of int and str. int is for array-like indexing so
        # that it can be used by torch dataloading, while the string is for whatever we want.

        if (min_dataset_size is not None) and (df_project.height < min_dataset_size):
            small_project_dfs.append(df_project)
        else:
            project_id_to_dataset[project_id] = df_to_dataset(
                df_project, use_confidence_score=use_confidence_score, confidence_score_floor=confidence_score_floor
            )

    if small_project_dfs:
        df_packed = pl.concat(small_project_dfs)
        project_id_to_dataset["__packed__"] = df_to_dataset(
            df_packed, use_confidence_score=use_confidence_score, confidence_score_floor=confidence_score_floor
        )

    return DatasetDict(project_id_to_dataset)


def _load_train_df(
    sample_size: int | None = None,
    stress_test_min_pair_len: int | None = None,
    paths: tuple[str, ...] = gt.data.DEFAULT_TRAIN_PATHS,
    source_to_sample_weight: dict[str, float] | None = None,
) -> tuple[pl.DataFrame, int]:
    if stress_test_min_pair_len is not None:
        df = gt.data.load_train_df(paths=paths, sample_size=None)  # bypass sampling
        df = df.filter(
            (pl.col("query_stacktrace_string").str.len_chars() + pl.col("candidate_stacktrace_string").str.len_chars())
            > stress_test_min_pair_len
        )
    else:
        df = gt.data.load_train_df(paths=paths, sample_size=sample_size)

    if source_to_sample_weight:
        df = df.with_columns(
            pl.col("source").replace_strict(source_to_sample_weight, default=1.0).alias("sample_weight")
        )
    else:
        df = df.with_columns(pl.lit(1.0).alias("sample_weight"))

    num_projects = len(df["project_id"].unique())
    return df, num_projects


def load_train_dataset(
    sample_size: int | None = None,
    stress_test_min_pair_len: int | None = None,
    paths: tuple[str, ...] = gt.data.DEFAULT_TRAIN_PATHS,
    source_to_sample_weight: dict[str, float] | None = None,
    use_confidence_score: bool = False,
    confidence_score_floor: float = 0.9,
) -> tuple[Dataset, float, int]:
    """
    Args:
        stress_test_min_pair_len: If set, bypasses sample_size and instead keeps only pairs
            where (query + candidate character length) > this threshold. Useful for OOM stress testing.
        source_to_sample_weight: Maps source column values to sample weights. Sources not in the dict get weight 1.0.
    """
    df, num_projects = _load_train_df(
        sample_size=sample_size,
        stress_test_min_pair_len=stress_test_min_pair_len,
        paths=paths,
        source_to_sample_weight=source_to_sample_weight,
    )
    dataset_train = df_to_dataset(
        df,
        use_confidence_score=use_confidence_score,
        confidence_score_floor=confidence_score_floor,
        group_by_query_stacktrace_string=False,
    )
    frac_positive = (df["label"] == "GROUP").mean()
    return dataset_train, frac_positive, num_projects


def load_train_dataset_dict(
    sample_size: int | None = None,
    stress_test_min_pair_len: int | None = None,
    paths: tuple[str, ...] = gt.data.DEFAULT_TRAIN_PATHS,
    source_to_sample_weight: dict[str, float] | None = None,
    use_confidence_score: bool = False,
    confidence_score_floor: float = 0.9,
    min_dataset_size: int | None = None,
) -> tuple[DatasetDict, float, int]:
    """
    Args:
        stress_test_min_pair_len: If set, bypasses sample_size and instead keeps only pairs
            where (query + candidate character length) > this threshold. Useful for OOM stress testing.
        source_to_sample_weight: Maps source column values to sample weights. Sources not in the dict get weight 1.0.
        min_dataset_size: If set, packs projects below this size into a single dataset to avoid tiny batches.
    """
    df, num_projects = _load_train_df(
        sample_size=sample_size,
        stress_test_min_pair_len=stress_test_min_pair_len,
        paths=paths,
        source_to_sample_weight=source_to_sample_weight,
    )
    dataset_dict_train = create_project_dataset_dict(
        df,
        min_dataset_size=min_dataset_size,
        use_confidence_score=use_confidence_score,
        confidence_score_floor=confidence_score_floor,
    )
    frac_positive = (df["label"] == "GROUP").mean()
    return dataset_dict_train, frac_positive, num_projects


@dataclass
class DefaultDataCollator(SentenceTransformerDataCollator):
    def __call__(self, records: list[Record]) -> Batch:
        batch: dict[str, Any] = default_collate(records)
        # MPS doesn't support float64, so convert to float32
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                batch[key] = value.float()
        return batch


def batch_pairs_by_token_budget(
    batch: Batch,
    *,
    token_budget: int,
    count_tokens: Callable[[str], int] = lambda text: max(1, len(text) // 4),
) -> Iterator[Batch]:
    """
    Split a collated `Batch` into smaller `Batch`es whose (estimated) token usage stays under `token_budget`.

    Preserves order across and w/in pairs. If one pair exceeds the token budget, it's still included as its own
    sub-batch (i.e., its a batch with one pair).
    """
    batch_keys = list(batch.keys())

    for key1, key2 in itertools.combinations(batch_keys, 2):
        if len(batch[key1]) != len(batch[key2]):
            raise ValueError(f"Batch fields {key1} and {key2} have inconsistent lengths")
    if len(batch[batch_keys[0]]) == 0:
        raise ValueError("Batch is empty")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    start = 0
    curr_max_num_tokens = 0
    curr_pairs = 0

    for idx in range(len(batch["query_stacktrace_string"])):
        num_tokens_query = count_tokens(batch["query_stacktrace_string"][idx])
        num_tokens_candidate = count_tokens(batch["candidate_stacktrace_string"][idx])
        new_max_num_tokens = max(curr_max_num_tokens, num_tokens_query, num_tokens_candidate)
        new_pairs = curr_pairs + 1
        est_cost = (2 * new_pairs) * new_max_num_tokens
        # Approximate padded token work as: num_texts_in_microbatch * max_tokens_in_microbatch
        # Since we encode 2 texts per pair, num_texts <= 2 * num_pairs. (Not equal b/c of caching.)

        if curr_pairs > 0 and est_cost > token_budget:
            # Flush [start, i)
            yield {key: batch[key][start:idx] for key in batch_keys}
            start = idx
            curr_max_num_tokens = max(num_tokens_query, num_tokens_candidate)
            curr_pairs = 1
            continue

        curr_max_num_tokens = new_max_num_tokens
        curr_pairs = new_pairs

    # Flush tail
    if start < len(batch["query_stacktrace_string"]):
        yield {key: batch[key][start:] for key in batch_keys}


class ModelForTraining(torch.nn.Module):
    def __init__(self, encoder: gt.utils.SentenceTransformer, loss: gt.loss.PairwiseLoss):
        super().__init__()
        self.encoder = encoder
        # TODO: torch.compile(encoder[0].auto_model, dynamic=True) b/c variable batch sizes when calling the model, and
        # very variable sequence lengths. Don't batch by sequence length b/c that seems statistically bad.
        self.loss = loss

    def encode(self, inputs: Batch) -> gt.loss.Features:
        """
        Deduplicates inputs before calling the model.
        Recall that our dataloader loads stacktraces from the same project together, sorted by query string.
        """

        # NOTE: HF accelerate's DDP shards the dataset at the batch level, not the record level. So each GPU gets a
        # batch from a different project. The cache is local to the GPU, so cache hits are still maximized w/ DDP.

        queries = inputs["query_stacktrace_string"]
        candidates = inputs["candidate_stacktrace_string"]

        device = self.encoder.device

        texts_unique, inverse_indices = np.unique(queries + candidates, return_inverse=True)
        inverse_indices = torch.as_tensor(inverse_indices, device=device)

        encodings = self.encoder.tokenize(texts_unique.tolist(), return_tensors="pt", padding=True)
        encodings = {k: v.to(device) for k, v in encodings.items()}
        embeddings_unique: torch.Tensor = self.encoder(encodings)["sentence_embedding"]

        all_embeddings = embeddings_unique[inverse_indices]
        num_queries = len(queries)
        query_embeddings = all_embeddings[:num_queries]
        candidate_embeddings = all_embeddings[num_queries:]

        return gt.loss.Features(query_embeddings=query_embeddings, candidate_embeddings=candidate_embeddings)

    def forward(
        self,
        inputs: Batch,
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        confidence_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encode(inputs)
        return self.loss(features, labels, sample_weight=sample_weight, confidence_scores=confidence_scores)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        encoder: gt.utils.SentenceTransformer,
        loss_type: Literal["sigmoid", "contrastive"] = "contrastive",
        contrastive_margin: float = 0.5,
    ) -> "ModelForTraining":
        if loss_type == "sigmoid":
            loss = gt.loss.SigmoidPairwiseLoss()
        elif loss_type == "contrastive":
            loss = gt.loss.ContrastiveLoss(margin=contrastive_margin)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        model = cls(encoder=encoder, loss=loss)
        safetensors_load_model(model, os.path.join(checkpoint_dir, "model.safetensors"))
        return model


class Trainer(SentenceTransformerTrainer):
    """
    Unlike SentenceTransformerTrainer, this class inputs a module whose forward computes the loss. This makes things
    like DDP and FSDP work out of the box (after I figure out how to make the subbatch backward stuff work w/ it...)

    Also fixes a bug where loss parameters aren't saved and aren't picked up when resuming training from a checkpoint.

    Note
    ----
    - The saved model is not what should be used directly for inference. Save `.encoder` separately.
    - Should pass `multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL` to get some interleaving of
      projects across batches, while keeping the batch size high to average each gradient over many candidates for each
      query.
    """

    model: ModelForTraining

    def __init__(
        self,
        model: ModelForTraining,
        *args,
        shuffle_within_dataset: bool = False,
        per_device_token_budget: int = 8192 * 4,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.shuffle_within_dataset = shuffle_within_dataset
        self.per_device_token_budget = per_device_token_budget
        self.loss = None
        "The loss is part of the model."

    def add_model_card_callback(self, default_args_dict):
        """
        No-op. (The superclass tokenizes the entire dataset as part of init.)
        """
        return None

    def _include_prompt_length(self) -> bool:
        for module in self.model.encoder:
            if isinstance(module, Pooling):
                return not module.include_prompt
        return False

    def call_model_init(self, trial=None):
        return super(SentenceTransformerTrainer, self).call_model_init(trial=trial)

    def prepare_loss(self, loss, model):
        """
        Pass-through. The model has the loss module. So it's already on the device.
        """
        return loss

    def compute_loss(
        self, model: ModelForTraining, inputs: Batch, return_outputs: bool = False, num_items_in_batch=None
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        loss = model(
            inputs,
            inputs["label"],
            sample_weight=inputs["sample_weight"],
            confidence_scores=inputs["confidence_score"],
        )
        if return_outputs:
            return loss, {}
        return loss

    def get_batch_sampler(
        self,
        dataset,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> BatchSampler:
        """
        Returns a sampler for a single dataset/project. By default, returns a SequentialSampler for more cache hits in
        each batch, as each batch is assumed to be sorted by query string.
        """
        sampler = (
            RandomSampler(dataset, generator=generator)
            if self.shuffle_within_dataset
            else SequentialSampler(dataset)  # project dataset is sorted by query string
        )
        return BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

    def _count_tokens(self, text: str) -> int:
        # profile_dataloading.ipynb shows this is fast enough to not be a bottleneck.
        return self.model.encoder.tokenize([text])["input_ids"].shape[1]

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Batch,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Stacktrace lengths are intentionally variant.
        Reduce the chance of OOM by splitting `inputs` into sub-batches and accumulating gradients.

        Couldn't get flash-attn installed b/c of obscure GCC errors, so can't use varlen.

        NOTE: training_step corresponds to one optimizer.step call.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        num_pairs_total = len(inputs["label"])

        def _backward_on_sub_batch(sub_batch: Batch, *, no_sync: bool) -> torch.Tensor:
            num_pairs_sub_batch = len(sub_batch["label"])
            if num_pairs_sub_batch == 0:
                raise ValueError("Sub-batch has no pairs / label is empty")

            sub_inputs = self._prepare_inputs(sub_batch)

            sync_ctx = self.accelerator.no_sync(model) if no_sync else nullcontext()
            with sync_ctx:
                with self.compute_loss_context_manager():
                    loss: torch.Tensor = self.compute_loss(model, sub_inputs, num_items_in_batch=num_items_in_batch)

                if (
                    self.args.torch_empty_cache_steps is not None
                    and self.state.global_step % self.args.torch_empty_cache_steps == 0
                ):
                    if is_torch_mps_available():
                        torch.mps.empty_cache()
                    elif is_torch_cuda_available():
                        torch.cuda.empty_cache()

                kwargs = {}

                loss = loss * (num_pairs_sub_batch / num_pairs_total)
                # Assume the loss is an average over the sub-batch. Re-scale to match averaging over the full batch.
                #
                # The rest of this is just loss.backward() w/ MP bells and whistles.

                # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
                if (
                    not self.model_accepts_loss_kwargs or num_items_in_batch is None
                ) and self.compute_loss_func is None:
                    # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient
                    # accumulation steps
                    loss = loss / self.current_gradient_accumulation_steps

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)

                return loss.detach()

        sub_batches = list(
            batch_pairs_by_token_budget(
                inputs, token_budget=self.per_device_token_budget, count_tokens=self._count_tokens
            )
        )
        num_sub_batches = len(sub_batches)

        is_distributed = self.accelerator.num_processes > 1
        losses = []
        for sub_batch_idx in range(num_sub_batches):
            loss = _backward_on_sub_batch(sub_batches[sub_batch_idx], no_sync=True)
            losses.append(loss)

        if is_distributed:
            # Hardcode for DDP. The dummy gather to account for variable # sub-batches across GPUs didn't work for some
            # reason. TODO: figure out why to overlap all-reduce w/ final / no_sync=False backward
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        return sum(losses)  # we already re-scaled each loss

    def get_default_decay_parameter_names(self, model: ModelForTraining) -> list[str]:
        # Rename the method `get_decay_parameter_names` for clarity. The `forbidden_name_patterns` list is hardcoded.
        return super().get_decay_parameter_names(model)

    def get_decay_parameter_names(self, model: ModelForTraining) -> list[str]:
        """
        Additionally excludes `loss.log_scale`.
        """
        default_decay_parameters = set(self.get_default_decay_parameter_names(model))
        default_decay_parameters.discard("loss.log_scale")
        return list(default_decay_parameters)

    def get_optimizer_cls_and_kwargs(
        self, args: SentenceTransformerTrainingArguments, model: ModelForTraining | None = None
    ):
        try:  # there are good tests for this hack
            self.loss = self.model
            # Override optimizer param groups to be based on the model, not the loss.
            return super().get_optimizer_cls_and_kwargs(args, model)
            # SentenceTransformerTrainer has the learning_rate_mapping feature, so use super()
        finally:
            self.loss = None

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        super(SentenceTransformerTrainer, self)._save(output_dir, state_dict=state_dict)

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        super(SentenceTransformerTrainer, self)._load_from_checkpoint(checkpoint_path)


class GCSCheckpointUploadCallback(TrainerCallback):
    """
    Uploads checkpoints to GCS in a background thread after each save.

    Writes a sentinel file after each checkpoint upload so that a polling evaluator knows the upload is complete.

    Writes a sentinel file when training is complete.
    """

    def __init__(self, run_gcs_dir: str):
        self.run_gcs_dir = run_gcs_dir.rstrip("/")
        self._prev_thread: threading.Thread | None = None

    def _join_prev_thread(self):
        if self._prev_thread is not None:
            self._prev_thread.join()
            self._prev_thread = None

    def _upload_checkpoint(self, checkpoint_path: str, gcs_dest: str):
        subprocess.run(
            ["gcloud", "storage", "rsync", "-r", checkpoint_path, gcs_dest],
            check=True,
        )
        subprocess.run(
            ["gcloud", "storage", "cp", "-", f"{gcs_dest}/{gt.sentinels.CHECKPOINT_DONE}"],
            # eval poller triggers eval on this sentinel
            input=b"",
            check=True,
        )
        logger.info(f"Uploaded checkpoint to {gcs_dest}")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Avoid racing with save_total_limit cleanup. Useful for mini CPU runs
        self._join_prev_thread()

        checkpoint_path = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        gcs_dest = f"{self.run_gcs_dir}/{PREFIX_CHECKPOINT_DIR}-{state.global_step}"

        thread = threading.Thread(target=self._upload_checkpoint, args=(checkpoint_path, gcs_dest))
        thread.start()
        self._prev_thread = thread

    def on_train_end(self, args, state, control, **kwargs):
        # Join to ensure the last CHECKPOINT_DONE is visible before TRAINING_DONE.
        # Otherwise the eval poller's final backfill could miss it.
        self._join_prev_thread()
        subprocess.run(
            ["gcloud", "storage", "cp", "-", f"{self.run_gcs_dir}/{gt.sentinels.TRAINING_DONE}"],
            # eval poller stops on this sentinel
            input=b"",
            check=True,
        )
        logger.info("Wrote TRAINING_DONE sentinel")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_shortname: str

    # Training args
    per_device_train_batch_size: int
    per_device_token_budget: int
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    # The gradient is an average over per_device_train_batch_size pairs from gradient_accumulation_steps projects.
    training_csvs: tuple[str, ...] = gt.data.DEFAULT_TRAIN_PATHS
    sample_size_train: int | None = None  # downsample for CPU sanity check runs
    log_of_scale_init: float = math.log(10)
    learning_rate: float = 1e-4  # effective batch size should be large b/c more deduplication and more project mixing
    learning_rate_mapping: dict[str, float] = {
        r"^loss\.log_scale$": 2e-4,
        r"^loss\.bias$": 2e-4,
    }
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    resume_from_checkpoint: str | bool | None = None
    source_to_sample_weight: dict[str, float] = {
        "synthetic-negative-semi-easy": 1.0,
        "unmatched": 1.0,
        "matched": 1.0,
        "synthetic-hard-negative-llm": 2.0,
    }  # TODO: Literal. source values aren't documented anywhere yet.

    # Default for cache hits. 2x overall training speedup w/o increasing gradient var
    group_by_query_stacktrace_string: bool = True
    shuffle_within_dataset: bool = False
    # group_by_query_stacktrace_string=True, shuffle_within_dataset=True is a middleground: don't include too many of
    # the same query stacktrace strings in a batch, while still generating pairs from 1 project per batch.

    # TODO: poor accuracy. Needs to be fixed
    use_confidence_score: bool = False
    confidence_score_floor: float = 0.9

    # Loss
    loss_type: Literal["sigmoid", "contrastive"] = "contrastive"
    contrastive_margin: float = 0.5  # did the best among 0.25, 0.5, 0.75

    # MRL
    mrl_dim_to_weight: dict[int, float] = {768: 2.0, 512: 1.0, 256: 1.0, 128: 0.5, 64: 0.25}
    # Equal weights did slightly worse overall, no better at dim 64
    n_dims_per_step: int = 2  # TODO: tune

    # Logging
    wandb_project: str = "grouping-trainer"
    num_logs: int = 100
    num_checkpoints: int = 10  # also the number of eval runs


def init_bias(frac_positive: float) -> float:
    bias_init = math.log(frac_positive / (1 - frac_positive))
    logger.info(f"Bias init: {bias_init:.4f}")
    return bias_init


def make_trainer(model: gt.utils.SentenceTransformer, training_config: TrainingConfig) -> Trainer:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"{timestamp}-{training_config.run_shortname}"

    # Load data
    load_kwargs = dict(
        sample_size=training_config.sample_size_train,
        paths=training_config.training_csvs,
        source_to_sample_weight=training_config.source_to_sample_weight or None,
        use_confidence_score=training_config.use_confidence_score,
        confidence_score_floor=training_config.confidence_score_floor,
    )
    if training_config.group_by_query_stacktrace_string:
        train_dataset, frac_positive, num_projects = load_train_dataset_dict(
            **load_kwargs, min_dataset_size=training_config.per_device_train_batch_size
        )
        if "__packed__" in train_dataset:
            logger.info(
                f"Packed {len(train_dataset['__packed__'])} pairs from projects w/ fewer than "
                f"{training_config.per_device_train_batch_size} rows into a single dataset."
            )
        num_rows = sum(train_dataset.num_rows.values())
    else:
        train_dataset, frac_positive, num_projects = load_train_dataset(**load_kwargs)
        num_rows = train_dataset.num_rows

    logger.info(f"Training dataset: {num_projects:,} projects, {num_rows:,} pairs")

    # Turn num_logs and num_checkpoints into log_steps and save_steps
    num_devices = max(1, torch.cuda.device_count())
    rows_per_device = math.ceil(num_rows / num_devices)  # DistributedSampler pads
    num_batches = math.ceil(rows_per_device / training_config.per_device_train_batch_size)
    steps_total = math.ceil(num_batches / training_config.gradient_accumulation_steps)
    logging_steps = max(1, steps_total // training_config.num_logs)
    save_steps = max(1, steps_total // training_config.num_checkpoints)
    logger.info(f"Estimated {steps_total:,} optimizer steps, logging every {logging_steps}, saving every {save_steps}")

    # Set up model
    gt.utils._cuda_empty_cache()
    assert "batch" not in repr(model[0].auto_model).lower(), (
        "Batch transformations like batch norm mess up deduplication"
    )
    kwargs_mrl = dict(
        mrl_dim_to_weight=training_config.mrl_dim_to_weight,
        n_dims_per_step=training_config.n_dims_per_step,
    )
    if training_config.loss_type == "sigmoid":
        loss = gt.loss.SigmoidPairwiseLoss(
            bias_init=init_bias(frac_positive),
            log_of_scale_init=torch.tensor(training_config.log_of_scale_init),
            **kwargs_mrl,
        )
    elif training_config.loss_type == "contrastive":
        loss = gt.loss.ContrastiveLoss(
            margin=training_config.contrastive_margin,
            **kwargs_mrl,
        )
    else:
        raise ValueError(f"Unknown loss_type: {training_config.loss_type}")

    # Sigmoid loss has learnable params (log_scale, bias) with custom LRs; contrastive has none.
    learning_rate_mapping = training_config.learning_rate_mapping if training_config.loss_type == "sigmoid" else {}

    model_for_training = ModelForTraining(encoder=model, loss=loss)

    return gt.train.Trainer(
        model=model_for_training,
        args=SentenceTransformerTrainingArguments(
            output_dir=f"./{run_name}",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=False,
            dataloader_pin_memory=torch.cuda.is_available(),
            num_train_epochs=1,  # data is very large. empirically 1 epoch is plenty
            gradient_checkpointing=training_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            #
            # Datalaoder
            # When group_by_query_stacktrace_string is False (train_dataset is a Dataset):
            batch_sampler=BatchSamplers.BATCH_SAMPLER,
            # When group_by_query_stacktrace_string is True (train_dataset is a DatasetDict):
            multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
            # Each iter, pick a project randomly, sample from it.
            # Next iter, pick another project randomly, sample from it, etc.
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            seed=42,  # passed to batch sampler
            #
            # Optimizer
            learning_rate=training_config.learning_rate,
            learning_rate_mapping=learning_rate_mapping,
            weight_decay=training_config.weight_decay,
            warmup_ratio=training_config.warmup_ratio,
            #
            # Logging
            logging_strategy="steps",
            logging_steps=logging_steps,
            run_name=run_name,
            report_to="wandb",
            #
            # Checkpointing
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
        ),
        #
        # Training
        data_collator=gt.train.DefaultDataCollator(tokenize_fn=model_for_training.encoder.tokenize),
        train_dataset=train_dataset,
        shuffle_within_dataset=(
            (not training_config.group_by_query_stacktrace_string) or training_config.shuffle_within_dataset
        ),
        per_device_token_budget=training_config.per_device_token_budget,
        #
        # Eval is async on a separate machine. See eval/eval_poller.py
    )
