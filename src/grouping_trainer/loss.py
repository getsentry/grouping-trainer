"""
Pairwise losses. See https://github.com/getsentry/grouping-trainer/blob/main/decisions.md
"""

from abc import ABC, abstractmethod
from typing import Protocol, TypedDict

import torch
from sentence_transformers.util import pairwise_cos_sim


class Features(TypedDict):
    query_embeddings: torch.Tensor
    candidate_embeddings: torch.Tensor


class _ComputeLossFromEmbeddings(Protocol):
    def __call__(self, all_embeddings: tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor: ...


def _mrl_loss(
    mrl_dim_to_weight: dict[int, float],
    n_dims_per_step: int,
    compute_loss_from_embeddings: _ComputeLossFromEmbeddings,
    all_embeddings: tuple[torch.Tensor, ...],
    *args,
    **kwargs,
) -> torch.Tensor:
    embedding_dim = all_embeddings[0].shape[-1]
    device = all_embeddings[0].device
    if any(mrl_dim > embedding_dim for mrl_dim in mrl_dim_to_weight.keys()):
        raise ValueError(f"mrl_dim_to_weight cannot exceed embedding dim {embedding_dim}: {mrl_dim_to_weight}")

    mrl_dims = list(mrl_dim_to_weight.keys())
    dim_indices = list(range(len(mrl_dims)))
    if n_dims_per_step > 0 and n_dims_per_step < len(dim_indices):
        # Randomizing over dims across subbatches, batches, and ranks is fine
        dim_indices = torch.randperm(len(mrl_dims))[:n_dims_per_step].tolist()

    loss_total = torch.zeros((), device=device)
    for idx in dim_indices:
        dim = mrl_dims[idx]
        embeddings_for_dim = tuple(embedding[..., :dim] for embedding in all_embeddings)
        loss_for_dim = compute_loss_from_embeddings(embeddings_for_dim, *args, **kwargs)
        loss_total += mrl_dim_to_weight[dim] * loss_for_dim
        # Appends a computation graph, but that's fine since our models don't have super high embedding dimensions.
        # Can address by:
        # 1. Detach embeddings from graph
        # 2. Loop over dims, compute loss, backward in loop
        # 3. Backprop detached embeddings' gradients to model
    weight_total = sum(mrl_dim_to_weight[mrl_dims[idx]] for idx in dim_indices)
    # NOTE: the MRL paper and sentence-transformers don't normalize. Idk why. Choosing to normalize so learning rates
    # don't lie when changing mrl_dim_to_weight or n_dims_per_step.
    return loss_total / weight_total


class PairwiseLoss(torch.nn.Module, ABC):
    @abstractmethod
    def compute_loss_from_similarities(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def compute_loss(
        self,
        all_embeddings: tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def forward(
        self,
        features: Features,
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.compute_loss(
            all_embeddings=(features["query_embeddings"], features["candidate_embeddings"]),
            labels=labels.float(),
            sample_weight=sample_weight,
        )


class ContrastiveLoss(PairwiseLoss):
    def __init__(
        self,
        *,
        margin: float = 0.5,
        mrl_dim_to_weight: dict[int, float] | None = None,
        n_dims_per_step: int = -1,
    ):
        super().__init__()
        self.margin = margin
        self.mrl_dim_to_weight = mrl_dim_to_weight if mrl_dim_to_weight else None
        self.n_dims_per_step = n_dims_per_step

    def compute_loss_from_similarities(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        distances: torch.Tensor = 1 - similarities
        loss_pos = labels * distances.pow(2)
        loss_neg = (1 - labels) * torch.relu(self.margin - distances).pow(2)
        loss_unreduced: torch.Tensor = 0.5 * (loss_pos + loss_neg)
        if sample_weight is None:
            return loss_unreduced.mean()
        return (loss_unreduced * sample_weight).sum() / sample_weight.sum()

    def compute_loss_from_embeddings(
        self,
        all_embeddings: tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_embeddings, candidate_embeddings = all_embeddings
        similarities = pairwise_cos_sim(query_embeddings, candidate_embeddings)
        return self.compute_loss_from_similarities(similarities, labels, sample_weight=sample_weight)

    def compute_loss(
        self,
        all_embeddings: tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.mrl_dim_to_weight is None:
            return self.compute_loss_from_embeddings(all_embeddings, labels, sample_weight=sample_weight)
        return _mrl_loss(
            mrl_dim_to_weight=self.mrl_dim_to_weight,
            n_dims_per_step=self.n_dims_per_step,
            compute_loss_from_embeddings=self.compute_loss_from_embeddings,
            all_embeddings=all_embeddings,
            labels=labels,
            sample_weight=sample_weight,
        )


class SigmoidPairwiseLoss(PairwiseLoss):
    def __init__(
        self,
        *,
        bias_init: float = 0.0,
        log_of_scale_init: torch.Tensor | None = None,
        mrl_dim_to_weight: dict[int, float] | None = None,
        n_dims_per_step: int = -1,
    ):
        super().__init__()
        if log_of_scale_init is None:
            log_of_scale_init = torch.tensor(5.0).log()
        self.log_scale = torch.nn.Parameter(log_of_scale_init.clone().detach())
        self.bias = torch.nn.Parameter(torch.tensor(bias_init))

        self.n_dims_per_step = n_dims_per_step
        self.mrl_dim_to_weight = mrl_dim_to_weight if mrl_dim_to_weight else None

    def compute_loss_from_similarities(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        scale = torch.exp(self.log_scale)
        logits = (similarities * scale) + self.bias
        loss_unreduced = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if sample_weight is None:
            return loss_unreduced.mean()
        return (loss_unreduced * sample_weight).sum() / sample_weight.sum()

    def compute_loss_from_embeddings(
        self,
        all_embeddings: tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_embeddings, candidate_embeddings = all_embeddings
        similarities = pairwise_cos_sim(query_embeddings, candidate_embeddings)
        return self.compute_loss_from_similarities(similarities, labels, sample_weight=sample_weight)

    def compute_loss(
        self,
        all_embeddings: tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.mrl_dim_to_weight is None:
            return self.compute_loss_from_embeddings(all_embeddings, labels, sample_weight=sample_weight)
        return _mrl_loss(
            mrl_dim_to_weight=self.mrl_dim_to_weight,
            n_dims_per_step=self.n_dims_per_step,
            compute_loss_from_embeddings=self.compute_loss_from_embeddings,
            all_embeddings=all_embeddings,
            labels=labels,
            sample_weight=sample_weight,
        )
