from dataclasses import dataclass
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Callable, TypedDict

from accelerate import DistributedType
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sentence_transformers.data_collator import SentenceTransformerDataCollator
import torch
from datasets import Dataset, DatasetDict
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.util import pairwise_cos_sim
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    default_collate,
)
from tqdm.auto import tqdm
from transformers.utils.import_utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
)


def df_to_dataset(df: pl.DataFrame) -> Dataset:
    return Dataset.from_list(
        [
            {
                "query_stacktrace_string": record["query_stacktrace_string"],
                "candidate_stacktrace_string": record["candidate_stacktrace_string"],
                "label": int(record["label"] == "GROUP"),
            }
            for record in df.sort("query_stacktrace_string").rows(named=True)
            # Sort for cache hits in the forward pass w/in each batch.
        ]
    )


def create_project_dataset_dict(
    df: pl.DataFrame,
    min_dataset_size: int | None = None,
) -> DatasetDict:
    """
    Create a DatasetDict with one dataset per project. Projects below `min_dataset_size` are packed into a single
    dataset to avoid tiny batches.

    Each dataset is sorted by query_stacktrace_string for cache hits in the forward pass.
    """
    project_id_to_dataset: dict[str, Dataset] = {}
    small_project_dfs: list[pl.DataFrame] = []

    for (project_id,), df_project in tqdm(df.group_by("project_id"), total=len(df["project_id"].unique())):
        project_id = str(project_id)
        # DatasetDict implements __getitem__ by accepting a mix of int and str. int is for array-like indexing so
        # that it can be used by torch dataloading, while the string is for whatever we want.

        if (min_dataset_size is not None) and (df_project.height < min_dataset_size):
            small_project_dfs.append(df_project)
        else:
            project_id_to_dataset[project_id] = df_to_dataset(df_project)

    if small_project_dfs:
        df_packed = pl.concat(small_project_dfs)
        project_id_to_dataset["__packed__"] = df_to_dataset(df_packed)

    return DatasetDict(project_id_to_dataset)


class Record(TypedDict):
    query_stacktrace_string: str
    candidate_stacktrace_string: str
    label: int


class Batch(TypedDict):
    query_stacktrace_string: list[str]
    candidate_stacktrace_string: list[str]
    label: torch.Tensor
    # NOTE: "label" is hardcoded in SentenceTransformerTrainer.collect_features, but we overrode it


@dataclass
class DefaulDataCollator(SentenceTransformerDataCollator):
    """
    We'll let the forward pass do the tokenization + device transfers b/c we'll have custom deduplication logic in the
    forward pass to save compute. It's easier to deduplicate via strings than via tensors.

    If we need to, we can deduplicate via tensors—

    Pros:
    - `SentenceTransformerDataCollator` has some niceties like handling the task type.

    Cons:
    - Requires custom handling for whatever the niceties are lol.
      Having `forward_deduplicated` dedupe input IDs is easy (re-pad them, call `torch.unique`). It's likely safe to
      update the attention mask by hand by setting all pad tokens to 0.
      I'm not familiar w/ the more custom encoding info for, e.g., sentence transformers that accept prompts.
      Finally, get rid of the `.collect_features` override.
    - Tokenization is repeated. I doubt it'd be a bottleneck, not sure.
    """

    def __call__(self, records: list[Record]) -> Batch:
        return default_collate(records)


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
    queries = batch["query_stacktrace_string"]
    candidates = batch["candidate_stacktrace_string"]
    labels = batch["label"]

    if len(queries) != len(candidates) or len(queries) != len(labels):
        raise ValueError("Batch fields have inconsistent lengths")
    if len(queries) == 0:
        raise ValueError("Batch has no pairs / label is empty")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    start = 0
    curr_max_num_tokens = 0
    curr_pairs = 0

    for i in range(len(queries)):
        num_tokens_query = count_tokens(queries[i])
        num_tokens_candidate = count_tokens(candidates[i])
        new_max_num_tokens = max(curr_max_num_tokens, num_tokens_query, num_tokens_candidate)
        new_pairs = curr_pairs + 1
        est_cost = (2 * new_pairs) * new_max_num_tokens
        # Approximate padded token work as: (num_texts_in_microbatch * max_tokens_in_microbatch).
        # Since we encode 2 texts per pair, num_texts <= 2 * num_pairs. (Not equal b/c of caching.)

        if curr_pairs > 0 and est_cost > token_budget:
            # Flush [start, i)
            yield {
                "query_stacktrace_string": queries[start:i],
                "candidate_stacktrace_string": candidates[start:i],
                "label": labels[start:i],
            }
            start = i
            curr_max_num_tokens = max(num_tokens_query, num_tokens_candidate)
            curr_pairs = 1
            continue

        curr_max_num_tokens = new_max_num_tokens
        curr_pairs = new_pairs

    # Flush tail
    if start < len(queries):
        yield {
            "query_stacktrace_string": queries[start:],
            "candidate_stacktrace_string": candidates[start:],
            "label": labels[start:],
        }


class Trainer(SentenceTransformerTrainer):
    """
    Should pass `multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL` to get some interleaving of
    projects across batches, while keeping the batch size high to average each gradient over many candidates for each
    query.
    """

    def __init__(
        self,
        *args,
        shuffle_within_dataset: bool = False,
        per_device_token_budget: int = 8192 * 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shuffle_within_dataset = shuffle_within_dataset
        self.per_device_token_budget = per_device_token_budget

    def add_model_card_callback(self, default_args_dict):
        """
        Skip this. The superclass tokenizes the entire dataset as part of init.
        """
        return None

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

    def collect_features(self, inputs: Batch):
        """
        Pass through the collated batch as is. We tokenize inside forward.
        """
        return inputs, inputs["label"]

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Batch,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Stacktrace lengths are intentionally variant.
        Reduce the chance of OOM by splitting `inputs` into sub-batches and accumulating gradients.

        NOTE: training_step corresponds to one optimizer.step call.
        """
        # No context parallelism here. I highly doubt we need that. Rather DDP if we have multiple GPUs.
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

        # Whether we should suppress gradient synchronization for intermediate microbatches.
        # If we're not syncing gradients anyway (e.g. during outer gradient accumulation), no_sync is unnecessary.
        should_no_sync = (
            self.accelerator.distributed_type == DistributedType.MULTI_GPU and self.accelerator.sync_gradients
        )

        sub_batches = batch_pairs_by_token_budget(inputs, token_budget=self.per_device_token_budget)
        sub_iter = iter(sub_batches)
        prev_sub_batch = next(sub_iter)
        losses = []

        # Process all but the last with no_sync
        for next_sub_batch in sub_iter:
            loss = _backward_on_sub_batch(prev_sub_batch, no_sync=should_no_sync)
            prev_sub_batch = next_sub_batch
            losses.append(loss)

        # Finally sync gradients for the last sub-batch.
        loss = _backward_on_sub_batch(prev_sub_batch, no_sync=False)
        losses.append(loss)

        return sum(losses)  # we already re-scaled each loss


class SigmoidPairwiseLoss(torch.nn.Module):
    """
    Simple pairwise loss. Does not collate in-batch negatives. For our dataset of pairs around the decision boundary,
    that's noisy and wrong.

    Note
    ----
    `bias_init` is required b/c it's highly dependent on your dataset. Can do log(1 / 1-p) where p is the fraction of
    positives.

    `log_of_scale_init`/temperature is on a log scale so that the learning rate is more reasonable. (Trick copied from
    SigLIP.) 10 is reasonable for our dataset. Anything higher could risk training just not working.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        *,
        bias_init: float,
        device: torch.device | None = None,
        log_of_scale_init: torch.Tensor = torch.tensor(10).log(),
        matryoshka_dims: list[int] | None = None,
        matryoshka_weights: list[float] | None = None,
        n_dims_per_step: int = -1,
    ):
        super().__init__()
        self.model = model
        self.device = device if device is not None else model.device

        self.log_scale = torch.nn.Parameter(log_of_scale_init.to(device=self.device))
        self.bias = torch.nn.Parameter(torch.tensor(bias_init, device=self.device))
        self.bce_with_logits_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

        self.n_dims_per_step = n_dims_per_step
        if matryoshka_dims is None:
            self.matryoshka_dims = None
            self.matryoshka_weights = None
        else:
            if len(matryoshka_dims) == 0:
                raise ValueError("matryoshka_dims must be non-empty (or None)")
            if matryoshka_weights is None:
                matryoshka_weights = [1] * len(matryoshka_dims)
            if len(matryoshka_weights) != len(matryoshka_dims):
                raise ValueError("matryoshka_weights must have the same length as matryoshka_dims")

            self.matryoshka_dims = matryoshka_dims
            self.matryoshka_weights = matryoshka_weights

    def forward_deduplicated(self, queries: list[str], candidates: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        # Map texts to idxs
        texts_unique, inverse_indices = np.unique(queries + candidates, return_inverse=True)
        inverse_indices = torch.as_tensor(inverse_indices, device=self.device)

        # Call model
        encodings = self.model.tokenize(texts_unique.tolist(), return_tensors="pt", padding=True)
        # No truncation needed b/c we sampled from already-encoded stacktraces in the grouping DB.
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        embeddings_unique: torch.Tensor = self.model(encodings)["sentence_embedding"]

        # Map embeddings back to queries and candidates
        all_embeddings = embeddings_unique[inverse_indices]
        # Copies, which is fine. Just want gradients to flow back correctly.
        num_queries = len(queries)
        query_embeddings = all_embeddings[:num_queries]
        candidate_embeddings = all_embeddings[num_queries:]

        return query_embeddings, candidate_embeddings

    def compute_loss_from_embeddings(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        exp_scale = torch.exp(self.log_scale)
        similarities = pairwise_cos_sim(query_embeddings, candidate_embeddings)
        logits = (similarities * exp_scale) + self.bias
        return self.bce_with_logits_loss(logits, labels)

    def compute_loss_mrl(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.matryoshka_dims is None:
            return self.compute_loss_from_embeddings(query_embeddings, candidate_embeddings, labels)

        embedding_dim = query_embeddings.shape[-1]
        if any(d > embedding_dim for d in self.matryoshka_dims):
            raise ValueError(f"matryoshka_dims cannot exceed embedding dim {embedding_dim}: {self.matryoshka_dims}")

        dim_indices = list(range(len(self.matryoshka_dims)))
        if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
            dim_indices = torch.randperm(len(self.matryoshka_dims), device=query_embeddings.device)[
                : self.n_dims_per_step
            ].tolist()

        # Prolly fine to append to computation graph as in SentenceTransformer's MatryoshkaLoss.
        # Our batch size and matryoshka_dims are small enough that it's not a big deal.
        loss_total = 0.0
        for idx in dim_indices:
            dim = self.matryoshka_dims[idx]
            weight = self.matryoshka_weights[idx]
            loss_dim = self.compute_loss_from_embeddings(
                query_embeddings[..., :dim], candidate_embeddings[..., :dim], labels
            )
            loss_total += weight * loss_dim
        return loss_total / len(dim_indices)

    def forward(self, batch: Batch, labels: torch.Tensor):
        query_embeddings, candidate_embeddings = self.forward_deduplicated(
            batch["query_stacktrace_string"],
            batch["candidate_stacktrace_string"],
        )
        return self.compute_loss_mrl(query_embeddings, candidate_embeddings, labels.float())
