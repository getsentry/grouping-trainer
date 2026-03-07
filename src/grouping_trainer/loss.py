"""
Pairwise loss w/ deduplication.
"""

from sentence_transformers.util import pairwise_cos_sim
import torch

import grouping_trainer as gt


class SigmoidPairwiseLoss(torch.nn.Module):
    def __init__(
        self,
        *,
        bias_init: float = 0.0,
        log_of_scale_init: torch.Tensor = torch.tensor(5.0).log(),
        matryoshka_dims: list[int] | None = None,
        matryoshka_weights: list[float] | None = None,
        n_dims_per_step: int = -1,
    ):
        super().__init__()
        self.log_scale = torch.nn.Parameter(log_of_scale_init.clone().detach())
        self.bias = torch.nn.Parameter(torch.tensor(bias_init))
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

    def compute_loss_from_similarities(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.log_scale)
        logits = (similarities * scale) + self.bias
        return self.bce_with_logits_loss(logits, labels)

    def compute_loss_from_embeddings(
        self, query_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        similarities = pairwise_cos_sim(query_embeddings, candidate_embeddings)
        return self.compute_loss_from_similarities(similarities, labels)

    def compute_loss_mrl(
        self, query_embeddings: torch.Tensor, candidate_embeddings: torch.Tensor, labels: torch.Tensor
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

        loss_total = torch.zeros((), device=query_embeddings.device)
        for idx in dim_indices:
            dim = self.matryoshka_dims[idx]
            weight = self.matryoshka_weights[idx]
            loss_dim = self.compute_loss_from_embeddings(
                query_embeddings[..., :dim], candidate_embeddings[..., :dim], labels
            )
            loss_total += weight * loss_dim
        return loss_total / len(dim_indices)

    def forward(self, features: gt.data.Features, labels: torch.Tensor) -> torch.Tensor:
        return self.compute_loss_mrl(
            query_embeddings=features["query_embeddings"],
            candidate_embeddings=features["candidate_embeddings"],
            labels=labels.float(),
        )
