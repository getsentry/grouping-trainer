"""
Trains a model, logs to wandb, and saves it to local and GCS.
Evaluation runs async on a separate machine.
"""

import torch
from tap import tapify

import grouping_trainer as gt


def main(mini_cpu_test: bool = False):
    """Train a grouping model.

    :param mini_cpu_test: Run a mini training run on CPU to sanity check the code.
    """
    is_cuda = torch.cuda.is_available()

    if not mini_cpu_test:
        assert is_cuda, "CUDA is required for full training"
        assert torch.cuda.is_bf16_supported(), "Get a GPU that supports bfloat16"

    model = gt.utils.SentenceTransformer(
        "Alibaba-NLP/gte-modernbert-base",
        model_kwargs=(
            dict(
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            if is_cuda
            else None
        ),
    )

    if mini_cpu_test:
        config = gt.train.TrainingConfig(
            run_shortname="cpu-sanity-check",
            per_device_train_batch_size=2,
            per_device_token_budget=64,
            gradient_checkpointing=True,
            sample_size_train=30,
            sample_size_val=20,
            logging_steps=1,
            save_steps=10,
        )
    else:
        config = gt.train.TrainingConfig(
            run_shortname="gte",
            per_device_train_batch_size=256,
            per_device_token_budget=8192 * 4,
            sample_size_val=8000,
        )

    gt.train.run(model, config)


if __name__ == "__main__":
    tapify(main)
