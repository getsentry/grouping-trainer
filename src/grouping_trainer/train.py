import gc
import logging
import math
import os
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Self, cast

import numpy as np
import torch
from accelerate import DistributedType
from pydantic import BaseModel, ConfigDict
from safetensors.torch import load_model as safetensors_load_model
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.models import Pooling
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler, default_collate
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils.import_utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
)

import grouping_trainer as gt

logger = logging.getLogger(__name__)


@dataclass
class DefaultDataCollator(SentenceTransformerDataCollator):
    def __call__(self, records: list[gt.data.Record]) -> gt.data.Batch:  # type: ignore[override]
        batch: dict[str, Any] = default_collate(records)
        # MPS doesn't support float64, so convert to float32
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                batch[key] = value.float()
        return batch  # type: ignore[return-value]


def batch_pairs_by_token_budget(
    batch: gt.data.Batch,
    *,
    token_budget: int,
    count_tokens: Callable[[str], int] = lambda text: max(1, len(text) // 4),
) -> Iterator[gt.data.Batch]:
    """
    Split a collated `Batch` into smaller `Batch`es whose (estimated) token usage stays under `token_budget`.

    Preserves order across and w/in pairs. If one pair exceeds the token budget, it's still included as its own
    sub-batch (i.e., its a batch with one pair).
    """
    n = len(batch["query_stacktrace_string"])
    if not (n == len(batch["candidate_stacktrace_string"]) == len(batch["label"]) == len(batch["sample_weight"])):
        raise ValueError("Batch fields have inconsistent lengths")
    if n == 0:
        raise ValueError("Batch is empty")

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    def _slice_batch(lo: int, hi: int | None) -> gt.data.Batch:
        sliced: gt.data.Batch = {key: batch[key][lo:hi] for key in batch.keys()}  # type: ignore[assignment,literal-required]
        return sliced

    start = 0
    curr_max_num_tokens = 0
    curr_pairs = 0

    for idx in range(n):
        num_tokens_query = count_tokens(batch["query_stacktrace_string"][idx])
        num_tokens_candidate = count_tokens(batch["candidate_stacktrace_string"][idx])
        new_max_num_tokens = max(curr_max_num_tokens, num_tokens_query, num_tokens_candidate)
        new_pairs = curr_pairs + 1
        est_cost = (2 * new_pairs) * new_max_num_tokens
        # Approximate padded token work as: num_texts_in_microbatch * max_tokens_in_microbatch
        # Since we encode 2 texts per pair, num_texts <= 2 * num_pairs. (Not equal b/c of caching.)

        if curr_pairs > 0 and est_cost > token_budget:
            yield _slice_batch(start, idx)
            start = idx
            curr_max_num_tokens = max(num_tokens_query, num_tokens_candidate)
            curr_pairs = 1
            continue

        curr_max_num_tokens = new_max_num_tokens
        curr_pairs = new_pairs

    if start < n:
        yield _slice_batch(start, None)  # flush tail


class ModelForTraining(torch.nn.Module):
    def __init__(self, encoder: gt.utils.SentenceTransformer, loss: gt.loss.PairwiseLoss):
        super().__init__()
        self.encoder = encoder
        # TODO: torch.compile(encoder[0].auto_model, dynamic=True) b/c variable batch sizes when calling the model, and
        # very variable sequence lengths. Don't globally batch by sequence length b/c that seems statistically bad.
        self.loss = loss

    def encode(self, inputs: gt.data.Batch) -> gt.loss.Features:
        """
        Deduplicates inputs before calling the model.
        Recall that our dataloader loads stacktraces from the same project together, sorted by query string.
        """

        # NOTE: HF accelerate's DDP shards the dataset at the batch level, not the record level. So each GPU gets a
        # batch from a different project, and this batch is sequentially sampled / contiguous as usual. So cache hits
        # are still maximized w/ DDP.

        queries = inputs["query_stacktrace_string"]
        candidates = inputs["candidate_stacktrace_string"]

        device = self.encoder.device

        texts_unique, inverse_indices_np = np.unique(queries + candidates, return_inverse=True)
        inverse_indices = torch.as_tensor(inverse_indices_np, device=device)

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
        inputs: gt.data.Batch,
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self.encode(inputs)
        return self.loss(features, labels, sample_weight=sample_weight)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        encoder: gt.utils.SentenceTransformer,
        loss_type: Literal["sigmoid", "contrastive"] = "contrastive",
        contrastive_margin: float = 0.5,
    ) -> Self:
        loss: gt.loss.PairwiseLoss
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
    Unlike `SentenceTransformerTrainer`, this class inputs a module whose forward computes the loss. Makes things like
    DDP and FSDP work out of the box. This choice also fixes a bug where loss parameters aren't saved and aren't picked
    up when resuming training from a checkpoint. The saved model is `ModelForTraining`, not a `SentenceTransformer`.

    Subclassing `SentenceTransformerTrainer` b/c it comes w/ very useful samplers, has optimizer param groups, and it
    handles custom tokenization that we could hook into in the future.
    """

    model: ModelForTraining  # type: ignore[assignment]

    def __init__(
        self,
        model: ModelForTraining,
        *args,
        shuffle_within_dataset: bool = False,
        per_device_token_budget: int = 8192 * 4,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)  # type: ignore[arg-type]
        self.shuffle_within_dataset = shuffle_within_dataset
        self.per_device_token_budget = per_device_token_budget
        self.loss = None  # type: ignore[assignment]
        "The loss is part of the model."

    def add_model_card_callback(self, default_args_dict: dict[str, Any]):
        """
        No-op. (The superclass tokenizes the entire dataset as part of init.)
        """
        return

    def _include_prompt_length(self) -> bool:
        for module in self.model.encoder:
            if isinstance(module, Pooling):
                return not module.include_prompt
        return False

    def call_model_init(self, trial: Any | None = None):  # type: ignore[bad-override]
        return super(SentenceTransformerTrainer, self).call_model_init(trial=trial)

    def prepare_loss(self, loss, model):
        """
        Pass-through. The model has the loss module. So it's already on the device.
        """
        return loss

    def compute_loss(  # type: ignore[override]
        self,
        model: ModelForTraining,
        inputs: gt.data.Batch,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        loss = model(inputs, inputs["label"], sample_weight=inputs["sample_weight"])
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
        Returns a sampler for a single dataset/project. By default, returns a `SequentialSampler` for more cache hits in
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

    def _backward_on_sub_batch(
        self,
        model: ModelForTraining,
        sub_batch: gt.data.Batch,
        global_total_pairs: int,
        world_size: int,
        *,
        no_sync: bool,
        is_dummy: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_pairs_sub_batch = len(sub_batch["label"])
        if num_pairs_sub_batch == 0:
            raise ValueError("Sub-batch has no pairs / label is empty")

        sub_inputs = cast(gt.data.Batch, self._prepare_inputs(sub_batch))  # type: ignore[arg-type]

        sync_ctx = self.accelerator.no_sync(model) if no_sync else nullcontext()
        with sync_ctx:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, sub_inputs, num_items_in_batch=num_items_in_batch)
                assert isinstance(loss, torch.Tensor)  # return_outputs defaults to False
                if is_dummy:
                    loss = loss * 0.0

            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_mps_available():
                    torch.mps.empty_cache()
                elif is_torch_cuda_available():
                    torch.cuda.empty_cache()

            kwargs = {}

            scale_factor = (num_pairs_sub_batch / global_total_pairs) * world_size
            loss = loss * scale_factor
            # Assume the loss is an average over the sub-batch. Re-scale to match averaging over the full batch.

            # The rest of this is loss.backward() w/ MP bells and whistles. Comments were copied from the base class.

            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient
                # accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()

    def training_step(  # type: ignore[override]
        self,
        model: ModelForTraining,
        inputs: gt.data.Batch,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Stacktrace lengths are intentionally variant, ranging from 10 tokens to 8192 tokens.
        Reduce the chance of OOM by splitting `inputs` into sub-batches and accumulating gradients.

        Couldn't get flash-attn installed, so can't use varlen. Was getting an obscure chain of GCC errors.
        """
        # NOTE: training_step corresponds to one optimizer.step call and is wrapped in a no_sync context by accelerate.
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):  # type: ignore[union-attr]
            self.optimizer.train()  # type: ignore[union-attr]

        num_pairs_total = len(inputs["label"])
        sub_batches = list(
            batch_pairs_by_token_budget(
                inputs, token_budget=self.per_device_token_budget, count_tokens=self._count_tokens
            )
        )
        num_sub_batches = len(sub_batches)

        if self.accelerator.num_processes > 1:
            local_stats = torch.tensor(
                [num_pairs_total, num_sub_batches],
                dtype=torch.long,
                device=self.accelerator.device,
            )
            gathered_stats = self.accelerator.gather(local_stats)
            global_total_pairs = gathered_stats[0::2].sum().item()
            max_sub_batches = gathered_stats[1::2].max().item()
            world_size = self.accelerator.num_processes
        else:
            global_total_pairs = num_pairs_total
            max_sub_batches = num_sub_batches
            world_size = 1

        losses: list[torch.Tensor] = []
        dummy_batch = gt.data.make_dummy_batch()  # unit-tested to work w/ ModelForTraining.forward()

        # Each GPU can have a different number of sub-batches. Need all to call backward() the same number of times w/
        # the same no_sync pattern so that all GPUs always agree on the communication op. Otherwise, e.g., GPU 1 w/ too
        # few subbatches will want to AllReduce while GPU 2 wants to broadcast buffers (or something) -> deadlock.
        # To fix, pad GPU 1's training step to the max sub-batch count with dummy backward passes.
        # An alternate approach is to manually call all-reduce after the last sub-batch, but that removes overlap b/t
        # all-reduce and backward and hardcodes this method to DDP.
        for sub_batch_idx in range(max_sub_batches):
            if self.accelerator.sync_gradients:
                # accelerate syncs on the final batch in the grad acc loop. Override to not sync until the last
                # sub-batch of the last batch
                is_last_sub_batch = sub_batch_idx == max_sub_batches - 1
                no_sync = not is_last_sub_batch
            else:
                # accelerate already wrapped this entire training_step in a no_sync context. Nesting another no_sync
                # would break FSDP2, as its __exit__ unconditionally re-enables syncing.
                no_sync = False

            is_dummy = sub_batch_idx >= num_sub_batches
            sub_batch = sub_batches[sub_batch_idx] if not is_dummy else dummy_batch
            loss = self._backward_on_sub_batch(
                model,
                sub_batch,
                global_total_pairs,
                world_size,
                no_sync=no_sync,
                is_dummy=is_dummy,
                num_items_in_batch=num_items_in_batch,
            )
            losses.append(loss)

        return torch.stack(losses).sum()

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

    def get_optimizer_cls_and_kwargs(  # type: ignore[override]
        self, args: SentenceTransformerTrainingArguments, model: ModelForTraining | None = None
    ):
        try:  # there are good tests for this hack
            self.loss = self.model  # type: ignore[assignment]
            # Override optimizer param groups to be based on the model, not the loss.
            return super().get_optimizer_cls_and_kwargs(args, model)  # type: ignore[arg-type]
            # SentenceTransformerTrainer has the learning_rate_mapping feature, so use super()
        finally:
            self.loss = None  # type: ignore[assignment]

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        super(SentenceTransformerTrainer, self)._save(output_dir, state_dict=state_dict)

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:  # type: ignore[override]
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

        assert args.output_dir is not None
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
    base_model: str

    # Training args
    global_train_batch_size: int
    per_device_token_budget: int
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    sample_size_train: int | None = None  # downsample for CPU sanity check runs
    log_of_scale_init: float = math.log(10)
    learning_rate: float = 1e-4  # effective batch size should be large b/c more deduplication and more project mixing
    learning_rate_mapping: dict[str, float] = {
        r"^loss\.log_scale$": 2e-4,
        r"^loss\.bias$": 2e-4,
    }
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Loss
    loss_type: Literal["sigmoid", "contrastive"] = "contrastive"
    contrastive_margin: float = 0.5  # did the best among 0.25, 0.5, 0.75

    # MRL
    mrl_dim_to_weight: dict[int, float] = {768: 2.0, 512: 1.0, 256: 1.0, 128: 0.5, 64: 0.25}
    # Equal weights did slightly worse overall, no better at dim 64
    n_dims_per_step: int = 2  # doesn't have much of an impact on accuracy or throughput

    # Training data and loader args
    training_csvs: tuple[str, ...] = gt.data.DEFAULT_TRAIN_PATHS
    # Out-of-platform generalization experiment. `platforms_holdout` names the platforms the experiment is built around
    # (used by both arms). `holdout_mode="drop_platforms"` is the treatment (those platforms never seen in training);
    # "drop_random_match" is the volume-matched control (same row count dropped at random, platforms still present).
    # `holdout_seed` only matters for the control. Empty `platforms_holdout` is a no-op (normal training).
    platforms_holdout: tuple[str, ...] = ()
    holdout_mode: gt.data.HoldoutMode = "drop_platforms"
    holdout_seed: int = 42
    source_to_sample_weight: dict[str, float] = {  # TODO: Literal. source values aren't documented anywhere yet.
        "synthetic-negative-semi-easy": 1.0,
        "unmatched": 1.0,
        "matched": 1.0,
        "synthetic-hard-negative-llm": 2.0,
    }
    # Defaults for cache hits. 2-3x overall training speedup. Costs some grad var and training stability b/c pairs are
    # positive correlated when grouped, but doesn't appear to affect out-of-sample accuracy.
    # The gradient is an average over global_train_batch_size pairs from gradient_accumulation_steps projects.
    group_by_query_stacktrace_string: bool = True
    shuffle_within_dataset: bool = False
    # group_by_query_stacktrace_string=True, shuffle_within_dataset=True is a middleground: don't include too many of
    # the same query stacktrace strings in a batch, while still generating pairs from 1 project per batch.

    # Logging
    num_logs: int = 100
    num_checkpoints: int = 10  # also the number of eval runs


def init_bias(frac_positive: float) -> float:
    bias_init = math.log(frac_positive / (1 - frac_positive))
    logger.info(f"Bias init: {bias_init:.4f}")
    return bias_init


def make_trainer(
    model: gt.utils.SentenceTransformer,
    training_config: TrainingConfig,
    run_name: str | None = None,
) -> Trainer:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_name = f"{timestamp}-{training_config.run_shortname}"

    num_devices = max(1, torch.cuda.device_count())
    per_device_train_batch_size = max(1, training_config.global_train_batch_size // num_devices)

    # Load data
    load_kwargs: dict[str, Any] = dict(
        sample_size=training_config.sample_size_train,
        paths=training_config.training_csvs,
        source_to_sample_weight=training_config.source_to_sample_weight or None,
        platforms_holdout=training_config.platforms_holdout,
        holdout_mode=training_config.holdout_mode,
        holdout_seed=training_config.holdout_seed,
    )
    if training_config.group_by_query_stacktrace_string:
        train_dataset, frac_positive, num_projects = gt.data.load_train_dataset_dict(
            **load_kwargs, min_dataset_size=per_device_train_batch_size
        )
        if "__packed__" in train_dataset:
            logger.info(
                f"Packed {len(train_dataset['__packed__'])} pairs from projects w/ fewer than "
                f"{per_device_train_batch_size} rows into a single dataset."
            )
        num_rows = sum(train_dataset.num_rows.values())
    else:
        train_dataset, frac_positive, num_projects = gt.data.load_train_dataset(**load_kwargs)
        num_rows = train_dataset.num_rows

    logger.info(f"Training dataset: {num_projects:,} projects, {num_rows:,} pairs")

    steps_total, logging_steps, save_steps = gt.utils.compute_train_steps(
        num_rows=num_rows,
        num_devices=num_devices,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        num_logs=training_config.num_logs,
        num_checkpoints=training_config.num_checkpoints,
    )
    logger.info(f"Estimated {steps_total:,} optimizer steps, logging every {logging_steps}, saving every {save_steps}")

    # Set up model
    gc.collect()
    torch.cuda.empty_cache()
    assert "batch" not in repr(model[0].auto_model).lower(), (
        "Batch transformations like batch norm mess up deduplication"
    )
    loss: gt.loss.PairwiseLoss
    if training_config.loss_type == "sigmoid":
        loss = gt.loss.SigmoidPairwiseLoss(
            bias_init=init_bias(frac_positive),
            log_of_scale_init=torch.tensor(training_config.log_of_scale_init),
            mrl_dim_to_weight=training_config.mrl_dim_to_weight,
            n_dims_per_step=training_config.n_dims_per_step,
        )
    elif training_config.loss_type == "contrastive":
        loss = gt.loss.ContrastiveLoss(
            margin=training_config.contrastive_margin,
            mrl_dim_to_weight=training_config.mrl_dim_to_weight,
            n_dims_per_step=training_config.n_dims_per_step,
        )
    else:
        raise ValueError(f"Unknown loss_type: {training_config.loss_type}")

    # Sigmoid loss has learnable params (log_scale, bias) with custom learning rates
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
            per_device_train_batch_size=per_device_train_batch_size,
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
        # Eval runs in parallel to training on a separate machine. See eval/eval_poller.py
    )
