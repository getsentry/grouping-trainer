"""
Trains a model, logs to wandb, and saves it to local and GCS.
Evaluation runs async on a separate machine.
"""

import torch

import grouping_trainer as gt

IS_CUDA_AVAILABLE = torch.cuda.is_available()

model = gt.utils.SentenceTransformer(
    "Alibaba-NLP/gte-modernbert-base",
    model_kwargs=(
        dict(
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        if IS_CUDA_AVAILABLE
        else None
    ),
)

training_config_full = gt.train.TrainingConfig(
    run_shortname="gte",
    per_device_train_batch_size=256,
    per_device_token_budget=8192 * 4,
    sample_size_val=8000,
)

training_config_mini = gt.train.TrainingConfig(
    run_shortname="cpu-sanity-check",
    per_device_train_batch_size=2,
    per_device_token_budget=512,
    gradient_checkpointing=True,
    sample_size_train=30,
    sample_size_val=20,
)

if __name__ == "__main__":
    assert IS_CUDA_AVAILABLE  # comment out for local sanity check runs
    assert torch.cuda.is_bf16_supported(), "Get a GPU that supports bfloat16"

    gt.train.run(model, training_config_full if IS_CUDA_AVAILABLE else training_config_mini)
