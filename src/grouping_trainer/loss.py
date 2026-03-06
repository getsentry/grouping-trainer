"""
Pairwise loss w/ deduplication.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_cos_sim
import torch

import grouping_trainer as gt


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

    These parameters are not registered on the model, but that's fine. We're monotonically transforming similarity.
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
        scale = torch.exp(self.log_scale)
        similarities = pairwise_cos_sim(query_embeddings, candidate_embeddings)
        logits = (similarities * scale) + self.bias
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

    def forward(self, batch: gt.data.Batch, labels: torch.Tensor):
        query_embeddings, candidate_embeddings = self.forward_deduplicated(
            batch["query_stacktrace_string"],
            batch["candidate_stacktrace_string"],
        )
        return self.compute_loss_mrl(query_embeddings, candidate_embeddings, labels.float())
