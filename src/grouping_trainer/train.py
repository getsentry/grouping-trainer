from datetime import datetime
import logging
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from collections.abc import Iterator
from contextlib import nullcontext
from typing import Callable, Literal, cast, overload
import warnings

from accelerate import DistributedType
from safetensors.torch import load_model as safetensors_load_model
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.models import Pooling
from sentence_transformers.training_args import MultiDatasetBatchSamplers
import torch
from datasets import Dataset, DatasetDict
from sentence_transformers.trainer import SentenceTransformerTrainer
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    default_collate,
)
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, TrainOutput
from transformers.utils.import_utils import (
    is_torch_cuda_available,
    is_torch_mps_available,
)
import wandb

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def df_to_dataset(df: pl.DataFrame, shuffle_groups: bool = True, seed: int | None = None) -> Dataset:
    """
    Convert a DataFrame to a Dataset, grouping records by `query_stacktrace_string`.

    Records with the same `query_stacktrace_string` are kept together for cache hits in the forward pass. By default, the
    order of groups is randomized to avoid alphabetical ordering bias during training.
    """
    query_group_dfs = [group_df for _, group_df in df.group_by("query_stacktrace_string")]
    if shuffle_groups:
        rng = random.Random(seed)
        rng.shuffle(query_group_dfs)
    else:
        query_group_dfs.sort(key=lambda query_group_df: query_group_df["query_stacktrace_string"][0])

    return Dataset.from_list(
        [
            {
                "query_stacktrace_string": record["query_stacktrace_string"],
                "candidate_stacktrace_string": record["candidate_stacktrace_string"],
                "label": int(record["label"] == "GROUP"),
            }
            for query_group_df in query_group_dfs
            for record in query_group_df.rows(named=True)
        ]
    )


def create_project_dataset_dict(
    df: pl.DataFrame,
    min_dataset_size: int | None = None,
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
            project_id_to_dataset[project_id] = df_to_dataset(df_project)

    if small_project_dfs:
        df_packed = pl.concat(small_project_dfs)
        project_id_to_dataset["__packed__"] = df_to_dataset(df_packed)

    return DatasetDict(project_id_to_dataset)


@dataclass
class DefaultDataCollator(SentenceTransformerDataCollator):
    def __call__(self, records: list[gt.data.Record]) -> gt.data.Batch:
        return default_collate(records)


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


class ModelForTraining(torch.nn.Module):
    def __init__(self, encoder: SentenceTransformer, loss: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.loss = loss

    def encode(self, inputs: gt.data.Batch) -> gt.data.Features:
        """
        Deduplicates inputs before calling the model.
        Recall that our dataloader loads stacktraces from the same project together, sorted by query string.
        """
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

        return gt.data.Features(query_embeddings=query_embeddings, candidate_embeddings=candidate_embeddings)

    def forward(self, inputs: gt.data.Batch, labels: torch.Tensor) -> torch.Tensor:
        features = self.encode(inputs)
        return self.loss(features, labels)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, encoder: SentenceTransformer) -> "ModelForTraining":
        loss = gt.loss.SigmoidPairwiseLoss()
        model = cls(encoder=encoder, loss=loss)
        safetensors_load_model(model, os.path.join(checkpoint_dir, "model.safetensors"))
        return model


class Trainer(SentenceTransformerTrainer):
    """
    Inputs a module whose forward method computes the loss. This change makes things like DDP and FSDP work out of the
    box. The saved model is not what should be used for inference. Saved `.encoder` separately.

    Note
    ----
    Should pass `multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL` to get some interleaving of
    projects across batches, while keeping the batch size high to average each gradient over many candidates for each
    query.
    """

    model: ModelForTraining

    def __init__(
        self,
        model: ModelForTraining,
        *args,
        shuffle_within_dataset: bool = False,
        per_device_token_budget: int = 8192 * 4,  # works for A100 80GB w/o sdpa (like jina-ai)
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.shuffle_within_dataset = shuffle_within_dataset
        self.per_device_token_budget = per_device_token_budget
        self.loss = None
        "The loss is part of the model."

    def add_model_card_callback(self, default_args_dict):
        """
        no-op. (The superclass tokenizes the entire dataset as part of init.)
        """
        return None

    def _include_prompt_length(self) -> bool:
        for module in self.model.encoder:
            if isinstance(module, Pooling):
                return not module.include_prompt
        return False

    def call_model_init(self, trial=None):
        return super(SentenceTransformerTrainer, self).call_model_init(trial=trial)

    def collect_features(self, inputs: gt.data.Batch) -> tuple[gt.data.Batch, torch.Tensor]:
        """
        Pass-through. The model encodes and calculates the loss.
        """
        return inputs, inputs["label"]

    def prepare_loss(self, loss, model):
        """
        Pass-through. The model has the loss module. So it's on the device.
        """
        return loss

    def compute_loss(
        self, model: ModelForTraining, inputs: gt.data.Batch, return_outputs: bool = False, num_items_in_batch=None
    ):
        loss = model(inputs, inputs["label"])
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
        return self.model.encoder.tokenize([text])["input_ids"].shape[1]

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: gt.data.Batch,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Stacktrace lengths are intentionally variant.
        Reduce the chance of OOM by splitting `inputs` into sub-batches and accumulating gradients.

        NOTE: training_step corresponds to one optimizer.step call.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        num_pairs_total = len(inputs["label"])

        def _backward_on_sub_batch(sub_batch: gt.data.Batch, *, no_sync: bool) -> torch.Tensor:
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

        sub_batches = batch_pairs_by_token_budget(
            inputs, token_budget=self.per_device_token_budget, count_tokens=self._count_tokens
        )
        sub_iter = iter(sub_batches)
        prev_sub_batch = next(sub_iter)
        losses = []

        # Backward all but the last with no_sync
        # Nesting another no_sync would break FSDP2. Its no_sync unconditionally re-enables sync on exit
        should_no_sync = self.accelerator.sync_gradients
        for next_sub_batch in sub_iter:
            loss = _backward_on_sub_batch(prev_sub_batch, no_sync=should_no_sync)
            prev_sub_batch = next_sub_batch
            losses.append(loss)

        loss = _backward_on_sub_batch(prev_sub_batch, no_sync=False)
        losses.append(loss)

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
        """
        Overrides optimizer param groups to be based on the model, not the loss.
        """
        # NOTE: ideally this logic is in a superclass method where the model is passed.

        if model is None:  # NOTE: this model is only wrapped when using sagemaker MP
            model = self.model

        optimizer_cls, optimizer_kwargs = super(SentenceTransformerTrainer, self).get_optimizer_cls_and_kwargs(
            args, model
        )

        decay_parameters = self.get_decay_parameter_names(model)

        # If the superclass did not already provide optimizer groups, create them from model params.
        if not {"params", "model", "optimizer_dict"} & set(optimizer_kwargs.keys()):
            # NOTE: optimizer_dict is what Trainer uses to store param groups to avoid argument conflicts.
            optimizer_kwargs["optimizer_dict"] = [
                {
                    "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
        # TODO: type optimizer_kwargs

        # One of "params", "model", or "optimizer_dict" should be present.
        optimizer_param_keys = set(optimizer_kwargs.keys()) & {"params", "model", "optimizer_dict"}
        optimizer_param_key = optimizer_param_keys.pop() if optimizer_param_keys else "optimizer_dict"

        for parameter_pattern, learning_rate in args.learning_rate_mapping.items():
            matching_params = {n: p for n, p in model.named_parameters() if re.search(parameter_pattern, n)}

            if not matching_params:
                raise ValueError(
                    f"No parameters found matching the pattern '{parameter_pattern}' in the model. "
                    "Please check the pattern and ensure it matches some of the model's parameters."
                )

            # Remove matching params from any existing optimizer groups so they can be re-added
            # with their custom learning rate.
            for group in optimizer_kwargs[optimizer_param_key]:
                if "params" in group:
                    group["params"] = [
                        p for p in group["params"] if all(p is not param for param in matching_params.values())
                    ]

            matching_params_with_decay = {n: p for n, p in matching_params.items() if n in decay_parameters}
            matching_params_without_decay = {n: p for n, p in matching_params.items() if n not in decay_parameters}

            if matching_params_with_decay:
                optimizer_kwargs[optimizer_param_key].append(
                    {
                        "params": list(matching_params_with_decay.values()),
                        "lr": learning_rate,
                        "weight_decay": self.args.weight_decay,
                    }
                )

            if matching_params_without_decay:
                optimizer_kwargs[optimizer_param_key].append(
                    {
                        "params": list(matching_params_without_decay.values()),
                        "lr": learning_rate,
                        "weight_decay": 0.0,
                    }
                )

        return optimizer_cls, optimizer_kwargs

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        super(SentenceTransformerTrainer, self)._save(output_dir, state_dict=state_dict)

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        super(SentenceTransformerTrainer, self)._load_from_checkpoint(checkpoint_path)


class GCSCheckpointUploadCallback(TrainerCallback):
    """
    Uploads checkpoints to GCS in a background thread after each save.

    Writes a `.checkpoint_done` sentinel after each checkpoint upload so that a polling evaluator knows the upload is complete.
    Writes a `.training_done` sentinel when training ends.
    """

    def __init__(self, gcs_dir: str):
        self.gcs_dir = gcs_dir.rstrip("/")
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
            [
                "gcloud",
                "storage",
                "cp",
                "-",
                f"{gcs_dest}/{gt.sentinels.CHECKPOINT_DONE}",
            ],  # eval poller triggers eval on this sentinel
            input=b"",
            check=True,
        )
        logger.info(f"Uploaded checkpoint to {gcs_dest}")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Join previous upload to avoid racing with save_total_limit cleanup
        self._join_prev_thread()

        checkpoint_path = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        gcs_dest = f"{self.gcs_dir}/{PREFIX_CHECKPOINT_DIR}-{state.global_step}"

        thread = threading.Thread(target=self._upload_checkpoint, args=(checkpoint_path, gcs_dest))
        thread.start()
        self._prev_thread = thread

    def on_train_end(self, args, state, control, **kwargs):
        self._join_prev_thread()
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                "-",
                f"{self.gcs_dir}/{gt.sentinels.TRAINING_DONE}",
            ],  # eval poller stops on this sentinel
            input=b"",
            check=True,
        )
        logger.info("Wrote .training_done sentinel")


def _launch_l4_eval(eval_cmd: str):
    """Copy the startup script to a tempfile with the eval command appended, then create the L4 instance."""
    startup_path = os.path.join(os.path.dirname(__file__), "../../bin/_startup.sh")
    # Append the eval command to run after setup, activating the conda env.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as tmp:
        shutil.copyfileobj(open(startup_path), tmp)
        tmp.write(f"\n{eval_cmd}\n")
        tmp_path = tmp.name

    try:
        create_l4_path = os.path.join(os.path.dirname(__file__), "../../bin/create_l4.sh")
        with open(create_l4_path) as f:
            create_script = f.read()
        create_script = create_script.replace("startup-script=bin/_startup.sh", f"startup-script={tmp_path}")
        subprocess.run(["bash", "-c", create_script], check=True)
    finally:
        os.unlink(tmp_path)
    logger.info("Created l4-eval instance with eval poller in startup script")


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_shortname: str

    # Training args
    per_device_train_batch_size: int
    per_device_token_budget: int
    training_csvs: tuple[str, ...] = (
        "final_csvs/train.csv",
        "final_csvs/synthetic-semi-easy-negatives.csv",
    )  # TODO: Literal
    sample_size_train: int | None = None  # downsample for CPU sanity check runs
    gradient_checkpointing: bool = False
    log_of_scale_init: float = math.log(5)
    learning_rate: float = 1e-4
    learning_rate_mapping: dict[str, float] = {
        r"^loss\.log_scale$": 2e-4,
        r"^loss\.bias$": 2e-4,
    }
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    resume_from_checkpoint: str | bool | None = None

    # MRL
    matryoshka_dims: tuple[int, ...] = (768, 512, 256, 128, 64)
    matryoshka_weights: tuple[float, ...] = (2.0, 1.0, 1.0, 0.5, 0.25)
    n_dims_per_step: int = 2

    # Logging
    wandb_project: str = "grouping-trainer"
    logging_steps: int = 30
    save_steps: int = 150


def init_bias(frac_positive: float) -> float:
    bias_init = math.log(frac_positive / (1 - frac_positive))
    print(f"Bias init: {bias_init:.4f}")
    return bias_init


@overload
def run(model: SentenceTransformer, training_config: TrainingConfig, just_make_trainer: Literal[True]) -> Trainer: ...


@overload
def run(
    model: SentenceTransformer, training_config: TrainingConfig, just_make_trainer: Literal[False] = ...
) -> TrainOutput: ...


def run(
    model: SentenceTransformer, training_config: TrainingConfig, just_make_trainer: bool = False
) -> Trainer | TrainOutput:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = f"{timestamp}-{training_config.run_shortname}"

    # Load data
    dataset_dict_train, frac_positive = gt.data.load_train_dataset_dict(
        sample_size=training_config.sample_size_train,
        min_dataset_size=training_config.per_device_train_batch_size,
        paths=training_config.training_csvs,
    )
    print(
        f"Packed {len(dataset_dict_train['__packed__'])} pairs from projects w/ fewer than "
        f"{training_config.per_device_train_batch_size} rows into a single dataset."
    )
    print(f"Training dataset: {len(dataset_dict_train):,} projects, {sum(dataset_dict_train.num_rows.values()):,} rows")

    # Set up model
    gt.utils._cuda_empty_cache()
    assert "batch" not in repr(model[0].auto_model).lower(), (
        "Batch transformations like batch norm mess up deduplication"
    )
    model_for_training = ModelForTraining(
        encoder=model,
        loss=gt.loss.SigmoidPairwiseLoss(
            bias_init=init_bias(frac_positive),
            log_of_scale_init=torch.tensor(training_config.log_of_scale_init),
            matryoshka_dims=list(training_config.matryoshka_dims),
            matryoshka_weights=list(training_config.matryoshka_weights),
            n_dims_per_step=training_config.n_dims_per_step,
        ),
    )

    # Build trainer
    trainer = gt.train.Trainer(
        model=model_for_training,
        args=SentenceTransformerTrainingArguments(
            output_dir=f"./{run_name}-output",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=False,
            dataloader_pin_memory=torch.cuda.is_available(),
            num_train_epochs=1,
            gradient_checkpointing=training_config.gradient_checkpointing,
            #
            # Datalaoder
            multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
            # Each iter, pick a project randomly, sample from it.
            # Next iter, pick another project randomly, sample from it, etc.
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            seed=42,  # passed to batch sampler
            #
            # Optimizer
            learning_rate=training_config.learning_rate,
            learning_rate_mapping=training_config.learning_rate_mapping,
            weight_decay=training_config.weight_decay,
            warmup_ratio=training_config.warmup_ratio,
            #
            # Logging
            logging_strategy="steps",
            logging_steps=training_config.logging_steps,
            run_name=run_name,
            report_to="wandb" if not just_make_trainer else "none",
            #
            # Checkpointing
            save_strategy="steps",
            save_steps=training_config.save_steps,
            save_total_limit=2,
        ),
        #
        # Training
        data_collator=gt.train.DefaultDataCollator(tokenize_fn=model_for_training.encoder.tokenize),
        train_dataset=dataset_dict_train,
        shuffle_within_dataset=False,  # more cache hits in each forward
        per_device_token_budget=training_config.per_device_token_budget,
        # Eval runs async on a separate machine (eval_poller.py)
    )
    assert trainer.args.output_dir is not None  # for typing

    if just_make_trainer:
        return trainer

    # Set up wandb + GCS for training run
    wandb.login()
    gcs_dir = f"gs://grouping-data/runs/{run_name}"
    wandb.init(project=training_config.wandb_project, name=run_name)

    # Create an instance which polls for checkpoints and evaluates them
    base_model = model_for_training.encoder.model_card_data.base_model
    eval_cmd = (
        f"python eval/eval_poller.py --run_gcs_dir {gcs_dir} --wandb_run_id {wandb.run.id} --base_model {base_model}"
    )
    print(f"\nEval command: {eval_cmd}\n")
    _launch_l4_eval(eval_cmd)

    # Set up trainer
    trainer.add_callback(gt.train.GCSCheckpointUploadCallback(gcs_dir=gcs_dir))

    # Train
    warnings.filterwarnings(
        "ignore",
        message=".*torch.utils.checkpoint: the use_reentrant parameter.*",
        category=UserWarning,
    )
    train_output = cast(TrainOutput, trainer.train(resume_from_checkpoint=training_config.resume_from_checkpoint))
    trainer.save_model()
    trainer.model.encoder.save_pretrained(os.path.join(trainer.args.output_dir, "inference"))

    # Upload wandb artifacts and final model to GCS
    subprocess.run(
        ["gcloud", "storage", "cp", "-r", "wandb", f"{gcs_dir}/wandb"],
        check=True,
    )
    subprocess.run(
        ["gcloud", "storage", "rsync", "-r", trainer.args.output_dir, f"{gcs_dir}/training"],
        check=True,
    )
    logger.info(f"Uploaded wandb artifacts and model to {gcs_dir}")

    return train_output
