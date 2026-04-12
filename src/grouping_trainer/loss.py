"""
Pairwise losses.

In-batch negatives are not used b/c that would result in lots of false negatives for our dataset. Across pairs and w/in
a project, there are plenty of similar stacktraces. Could change the sampler to pick a platform and sample a single
pair across many projects, but then there wouldn't be cache hits. And these negatives would likely be too easy b/c
they're cross-project. The module synthetic.py mines easy to semi-easy negative pairs w/in each project offline.
"""

from abc import ABC, abstractmethod
from typing import Protocol
from sentence_transformers.util import pairwise_cos_sim
import torch

import grouping_trainer as gt


class _ComputeLossFromEmbeddings(Protocol):
    def __call__(self, all_embeddings: tuple[torch.Tensor, ...], *args, **kwargs) -> torch.Tensor: ...


_mrl_call_counter = 0


def _mrl_loss(
    mrl_dim_to_weight: dict[int, float],
    n_dims_per_step: int,
    compute_loss_from_embeddings: _ComputeLossFromEmbeddings,
    all_embeddings: tuple[torch.Tensor, ...],
    *args,
    **kwargs,
) -> torch.Tensor:
    global _mrl_call_counter
    embedding_dim = all_embeddings[0].shape[-1]
    device = all_embeddings[0].device
    if any(mrl_dim > embedding_dim for mrl_dim in mrl_dim_to_weight.keys()):
        raise ValueError(f"mrl_dim_to_weight cannot exceed embedding dim {embedding_dim}: {mrl_dim_to_weight}")

    mrl_dims = list(mrl_dim_to_weight.keys())
    dim_indices = list(range(len(mrl_dims)))
    if n_dims_per_step > 0 and n_dims_per_step < len(dim_indices):
        # Use a call counter as seed so all DDP processes select the same dims
        gen = torch.Generator(device="cpu").manual_seed(_mrl_call_counter)
        _mrl_call_counter += 1
        dim_indices = torch.randperm(len(mrl_dims), generator=gen)[:n_dims_per_step].tolist()

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
    return loss_total / len(dim_indices)


class PairwiseLoss(torch.nn.Module, ABC):
    @abstractmethod
    def compute_loss_from_similarities(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        confidence_scores: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    def compute_loss(
        self,
        all_embeddings: tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        confidence_scores: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def forward(
        self,
        features: gt.data.Features,
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        confidence_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.compute_loss(
            all_embeddings=(features["query_embeddings"], features["candidate_embeddings"]),
            labels=labels.float(),
            sample_weight=sample_weight,
            confidence_scores=confidence_scores,
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
        log_of_scale_init: torch.Tensor = torch.tensor(5.0).log(),
        mrl_dim_to_weight: dict[int, float] | None = None,
        n_dims_per_step: int = -1,
    ):
        super().__init__()
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
        confidence_scores: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if confidence_scores is not None:
            labels = torch.where(labels == 0, 1 - confidence_scores, confidence_scores)
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
        confidence_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_embeddings, candidate_embeddings = all_embeddings
        similarities = pairwise_cos_sim(query_embeddings, candidate_embeddings)
        return self.compute_loss_from_similarities(
            similarities, labels, sample_weight=sample_weight, confidence_scores=confidence_scores
        )

    def compute_loss(
        self,
        all_embeddings: tuple[torch.Tensor, ...],
        labels: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
        confidence_scores: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.mrl_dim_to_weight is None:
            return self.compute_loss_from_embeddings(
                all_embeddings, labels, sample_weight=sample_weight, confidence_scores=confidence_scores
            )
        return _mrl_loss(
            mrl_dim_to_weight=self.mrl_dim_to_weight,
            n_dims_per_step=self.n_dims_per_step,
            compute_loss_from_embeddings=self.compute_loss_from_embeddings,
            all_embeddings=all_embeddings,
            labels=labels,
            sample_weight=sample_weight,
            confidence_scores=confidence_scores,
        )
