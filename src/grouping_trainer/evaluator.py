import logging
import os
from typing import Any

import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.util import pairwise_cos_sim

import grouping_trainer as gt

logger = logging.getLogger(__name__)


def _metric_name(*, dim: int, target_precision: float, metric: str) -> str:
    """Generate metric name like 'dim768_pr95_recall'."""
    return f"dim{dim}_pr{int(target_precision * 100)}_{metric}"


def _metric_name_loss(*, dim: int) -> str:
    return f"dim{dim}_loss"


class MinPrecisionEvaluator(SentenceEvaluator):
    """
    Evaluate a model by finding the minimum similarity threshold needed to achieve target precision levels, then
    reporting the recall at those thresholds.

    We want to control precision (e.g., avoid overgrouping errors) and maximize recall (reduce undergrouping/noise).

    One problem w/ this method is that precision doesn't monotonically relate to the threshold. In practice this isn't
    much of a problem b/c we have 80k pairs in the validation dataset.
    """

    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        labels: list[int],
        name: str = "",
        batch_size: int = 1,
        show_progress_bar: bool | None = False,
        write_csv: bool = True,
        truncate_dims: tuple[int, ...] | None = None,
        target_precisions: list[float] | None = None,
        min_predictions: int = 100,
    ):
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.truncate_dims = truncate_dims  # None means use model's full dimension
        self.target_precisions = target_precisions or [0.85, 0.90, 0.95, 0.99]
        self.min_predictions = min_predictions

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert label == 0 or label == 1

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = "min_precision_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.records: list[dict[str, float | int]] = []

    def __call__(
        self,
        model: SentenceTransformer | gt.train.ModelForTraining,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        """
        Compute the evaluation metrics for the given model.

        Args:
            model: The model to evaluate.
            output_path: Path to save the evaluation results CSV file.
            epoch: The epoch number. Defaults to -1.
            steps: The number of steps. Defaults to -1.

        Returns:
            A dictionary containing the evaluation metrics.
        """
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"Min Precision Evaluation of the model on the {self.name} dataset{out_txt}:")

        dims, raw_metrics = self.compute_metrics(model)

        # Log results
        for dim in dims:
            logger.info(f"  dim={dim}:")
            for p in self.target_precisions:
                precision = raw_metrics[_metric_name(dim=dim, target_precision=p, metric="precision")]
                recall = raw_metrics[_metric_name(dim=dim, target_precision=p, metric="recall")]

                if np.isnan(precision):
                    logger.info(f"    Precision >= {p:.0%}: Not achievable with >= {self.min_predictions} predictions")
                else:
                    logger.info(f"    Precision >= {p:.0%}: precision={precision:.2%}, recall={recall:.2%}")

        self.records.append({"epoch": epoch, "steps": steps, **raw_metrics})

        if output_path is not None and self.write_csv:
            pl.DataFrame(self.records).write_csv(os.path.join(output_path, self.csv_file))

        # Set primary metric (recall at precision 0.95, using largest dimension)
        primary_dim = max(dims)
        self.primary_metric = _metric_name(dim=primary_dim, target_precision=0.95, metric="recall")

        # Prefix with name
        metrics = self.prefix_name_to_metrics(raw_metrics, self.name)
        encoder = model.encoder if isinstance(model, gt.train.ModelForTraining) else model
        self.store_metrics_in_model_card_data(encoder, metrics, epoch, steps)

        return metrics

    def embed_inputs(  # type: ignore[override]
        self,
        model: SentenceTransformer | gt.train.ModelForTraining,
        sentences: str | list[str],
        **encode_kwargs,
    ) -> torch.Tensor:
        encoder = model.encoder if isinstance(model, gt.train.ModelForTraining) else model
        return encoder.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            convert_to_numpy=False,
            **encode_kwargs,
        )

    def get_config_dict(self) -> dict:
        config_dict: dict[str, Any] = {}
        if self.truncate_dims is not None:
            config_dict["truncate_dims"] = self.truncate_dims
        config_dict["target_precisions"] = self.target_precisions
        config_dict["min_predictions"] = self.min_predictions
        return config_dict

    def compute_metrics(
        self,
        model: SentenceTransformer | gt.train.ModelForTraining,
    ) -> tuple[tuple[int, ...], dict[str, float]]:
        """
        Compute embeddings, similarities, and find thresholds for each dimension and target precision.

        Returns:
            A tuple of (dims, metrics) where dims is the tuple of dimensions evaluated.
        """
        model.eval()
        sentences = list(set(self.sentences1 + self.sentences2))
        embeddings = self.embed_inputs(model, sentences)
        encoder = model.encoder if isinstance(model, gt.train.ModelForTraining) else model
        assert embeddings.device == encoder.device
        sentence_to_embedding = {sentence: embedding for sentence, embedding in zip(sentences, embeddings, strict=True)}
        embeddings1 = torch.stack([sentence_to_embedding[sentence] for sentence in self.sentences1])
        embeddings2 = torch.stack([sentence_to_embedding[sentence] for sentence in self.sentences2])

        dims = (embeddings1.shape[-1],) if self.truncate_dims is None else self.truncate_dims
        metric_name_to_value: dict[str, float] = {}
        for dim in dims:
            embeddings1_truncated = embeddings1[..., :dim]
            embeddings2_truncated = embeddings2[..., :dim]

            similarities = pairwise_cos_sim(embeddings1_truncated, embeddings2_truncated)

            # Compute loss if model includes the loss head
            if isinstance(model, gt.train.ModelForTraining):
                with torch.inference_mode():
                    loss = model.loss.compute_loss_from_similarities(
                        similarities,
                        labels=torch.as_tensor(self.labels, device=embeddings1.device, dtype=torch.float),
                    )
                metric_name_to_value[_metric_name_loss(dim=dim)] = loss.detach().cpu().item()

            metric_name_to_value.update(
                self.find_recall_at_precision_thresholds(
                    scores=similarities.detach().float().cpu().numpy(),
                    labels=np.asarray(self.labels),
                    target_precisions=self.target_precisions,
                    min_predictions=self.min_predictions,
                    dim=dim,
                )
            )

        return dims, metric_name_to_value

    @staticmethod
    def find_recall_at_precision_thresholds(
        *,
        scores: np.ndarray,
        labels: np.ndarray,
        target_precisions: list[float],
        min_predictions: int,
        dim: int,
    ) -> dict[str, float]:
        """
        For each target precision P, find the smallest threshold where:
        - precision >= P
        - n_predictions >= min_predictions

        Returns a dict with keys like dim768_pr95_threshold, dim768_pr95_recall, etc.
        """
        assert len(scores) == len(labels)

        # Sort by score descending (highest similarity first)
        rows = list(zip(scores, labels, strict=True))
        rows = sorted(rows, key=lambda x: x[0], reverse=True)

        total_positives = sum(labels)
        n_pairs = len(rows)

        # Track the last valid result for each target precision
        # "Last valid" means the lowest threshold we've seen that still meets the target
        target_precision_to_metric_name_to_value: dict[float, dict[str, float] | None] = {
            target_precision: None for target_precision in target_precisions
        }

        tp = 0
        n_predictions = 0

        for i in range(n_pairs):
            score, label = rows[i]
            n_predictions += 1
            if label == 1:
                tp += 1

            if n_predictions < min_predictions:
                continue

            precision = tp / n_predictions
            recall = tp / total_positives if total_positives > 0 else 0.0

            # The threshold for this point is the current score
            # (all pairs with score >= this score are predicted positive)
            threshold = score

            # For each target, if we meet it, record this as the new "lowest threshold"
            for target_precision in target_precisions:
                if precision >= target_precision:
                    target_precision_to_metric_name_to_value[target_precision] = {
                        "threshold": threshold,
                        "precision": precision,
                        "recall": recall,
                        "n_predictions": n_predictions,
                    }

        # Convert to flat dict with metric names
        metric_names = (
            # "threshold",
            # "n_predictions",
            "precision",
            "recall",
        )
        output = {}
        for target_precision in target_precisions:
            metric_name_to_value = target_precision_to_metric_name_to_value[target_precision]
            if metric_name_to_value is not None:
                for metric in metric_names:
                    output[_metric_name(dim=dim, target_precision=target_precision, metric=metric)] = (
                        metric_name_to_value[metric]
                    )
            else:
                for metric in metric_names:
                    output[_metric_name(dim=dim, target_precision=target_precision, metric=metric)] = float("nan")

        return output

    def parse_metric_name(self, metric_name: str) -> tuple[int, float, str]:
        """
        Parse a metric name like 'val_dim768_pr90_n_predictions' into (dim, precision, metric).

        Inverse of _metric_name.

        Returns:
            A tuple of (dim, precision, metric_name).
        """
        # metric_name example: val_dim768_pr90_n_predictions
        prefix = f"{self.name}_dim" if self.name else "dim"
        metric_name = metric_name.removeprefix(prefix)
        # Now: 768_pr90_n_predictions
        parts = metric_name.split("_")
        dim = int(parts[0])
        # parts[1] is like "pr90"
        precision = float(parts[1].removeprefix("pr")) / 100
        metric = "_".join(parts[2:])
        return dim, precision, metric
