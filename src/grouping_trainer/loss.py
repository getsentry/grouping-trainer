"""
Pairwise loss w/ deduplication.
"""

import os
from typing import TypedDict, cast
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim
import torch


class CalibrationHead(torch.nn.Module):
    """
    `bias_init` is highly dependent on the dataset. Start w/ log-odds if unsure.
    """

    def __init__(self, *, bias_init: float = 0.0, log_of_scale_init: torch.Tensor = torch.tensor(5.0).log()):
        super().__init__()
        self.log_scale = torch.nn.Parameter(log_of_scale_init.clone().detach())
        self.bias = torch.nn.Parameter(torch.tensor(bias_init))

    def forward(self, similarities: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_scale)
        return (similarities * scale) + self.bias


class TrainableSentenceTransformer(SentenceTransformer):  # just for typing
    calibration_head: CalibrationHead


def add_head_to_model(model: SentenceTransformer, calibration_head: CalibrationHead) -> TrainableSentenceTransformer:
    """
    Add a module w/ parameters to the model so that things like DDP work out of the box.
    """
    model.calibration_head = calibration_head
    return cast(TrainableSentenceTransformer, model)


def add_head_to_model_from_checkpoint(model: SentenceTransformer, checkpoint_dir: str) -> None:
    model.calibration_head = CalibrationHead()
    state = torch.load(os.path.join(checkpoint_dir, "calibration_head.pt"), map_location="cpu")
    model.calibration_head.load_state_dict(state)
    model.calibration_head.to(model.device)


class FeaturesWithHead(TypedDict):
    query_embeddings: torch.Tensor
    candidate_embeddings: torch.Tensor
    calibration_head: CalibrationHead
    # TODO: rename to head_for_loss, type as nn.Module, pass type for add_head_to_model_from_checkpoint


class SigmoidPairwiseLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        matryoshka_dims: list[int] | None = None,
        matryoshka_weights: list[float] | None = None,
        n_dims_per_step: int = -1,
    ):
        super().__init__()
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

    def compute_loss_from_similarities(
        self, similarities: torch.Tensor, labels: torch.Tensor, calibration_head: CalibrationHead
    ) -> torch.Tensor:
        logits = calibration_head(similarities)
        return self.bce_with_logits_loss(logits, labels)

    def compute_loss_from_embeddings(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        labels: torch.Tensor,
        calibration_head: CalibrationHead,
    ) -> torch.Tensor:
        similarities = pairwise_cos_sim(query_embeddings, candidate_embeddings)
        return self.compute_loss_from_similarities(similarities, labels, calibration_head)

    def compute_loss_mrl(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        labels: torch.Tensor,
        calibration_head: CalibrationHead,
    ) -> torch.Tensor:
        if self.matryoshka_dims is None:
            return self.compute_loss_from_embeddings(query_embeddings, candidate_embeddings, labels, calibration_head)

        embedding_dim = query_embeddings.shape[-1]
        if any(d > embedding_dim for d in self.matryoshka_dims):
            raise ValueError(f"matryoshka_dims cannot exceed embedding dim {embedding_dim}: {self.matryoshka_dims}")

        dim_indices = list(range(len(self.matryoshka_dims)))
        if self.n_dims_per_step > 0 and self.n_dims_per_step < len(dim_indices):
            dim_indices = torch.randperm(len(self.matryoshka_dims), device=query_embeddings.device)[
                : self.n_dims_per_step
            ].tolist()

        loss_total = 0.0
        for idx in dim_indices:
            dim = self.matryoshka_dims[idx]
            weight = self.matryoshka_weights[idx]
            loss_dim = self.compute_loss_from_embeddings(
                query_embeddings[..., :dim], candidate_embeddings[..., :dim], labels, calibration_head
            )
            loss_total += weight * loss_dim
        return loss_total / len(dim_indices)

    def forward(self, features_with_head: FeaturesWithHead, labels: torch.Tensor) -> torch.Tensor:
        return self.compute_loss_mrl(
            query_embeddings=features_with_head["query_embeddings"],
            candidate_embeddings=features_with_head["candidate_embeddings"],
            labels=labels.float(),
            calibration_head=features_with_head["calibration_head"],
        )
