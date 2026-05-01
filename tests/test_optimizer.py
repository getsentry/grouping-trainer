from copy import copy

import pytest
import torch
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss

import grouping_trainer as gt


@pytest.fixture()
def model() -> gt.utils.SentenceTransformer:
    return gt.utils.SentenceTransformer("Alibaba-NLP/gte-modernbert-base")


@pytest.fixture()
def config() -> gt.train.TrainingConfig:
    return gt.train.TrainingConfig(
        run_shortname="cpu-sanity-check",
        per_device_train_batch_size=2,
        per_device_token_budget=64,
        gradient_checkpointing=True,
        sample_size_train=30,
        num_logs=30,
        num_checkpoints=3,
        loss_type="sigmoid",  # only sigmoid loss has log_scale and bias params for the optimizer to group
    )


@pytest.fixture()
def trainer(model: gt.utils.SentenceTransformer, config: gt.train.TrainingConfig) -> gt.train.Trainer:
    return gt.train.make_trainer(model, config)


@pytest.fixture()
def baseline_trainer(model: gt.utils.SentenceTransformer, trainer: gt.train.Trainer) -> SentenceTransformerTrainer:
    args = copy(trainer.args)
    args.learning_rate_mapping = {}
    return SentenceTransformerTrainer(model, args=args, loss=MultipleNegativesRankingLoss(model))


def test_last_two_params_are_loss_scale_and_bias(trainer: gt.train.Trainer) -> None:
    names = [name for name, _ in trainer.model.named_parameters()]
    assert names[-2:] == ["loss.log_scale", "loss.bias"]


def test_optimizer_cls_matches_baseline(
    trainer: gt.train.Trainer, baseline_trainer: SentenceTransformerTrainer
) -> None:
    cls_custom, _ = trainer.get_optimizer_cls_and_kwargs(trainer.args)
    cls_baseline, _ = baseline_trainer.get_optimizer_cls_and_kwargs(baseline_trainer.args)
    assert cls_custom == cls_baseline


def test_has_extra_param_groups_for_loss(
    trainer: gt.train.Trainer, baseline_trainer: SentenceTransformerTrainer
) -> None:
    _, kwargs_custom = trainer.get_optimizer_cls_and_kwargs(trainer.args)
    _, kwargs_baseline = baseline_trainer.get_optimizer_cls_and_kwargs(baseline_trainer.args)
    groups_custom = kwargs_custom["optimizer_dict"]
    groups_baseline = kwargs_baseline["optimizer_dict"]

    assert len(groups_custom) >= 4
    assert len(groups_custom[:-2]) == len(groups_baseline)


def test_shared_param_groups_match_baseline(
    trainer: gt.train.Trainer, baseline_trainer: SentenceTransformerTrainer
) -> None:
    _, kwargs_custom = trainer.get_optimizer_cls_and_kwargs(trainer.args)
    _, kwargs_baseline = baseline_trainer.get_optimizer_cls_and_kwargs(baseline_trainer.args)
    groups_custom = kwargs_custom["optimizer_dict"]
    groups_baseline = kwargs_baseline["optimizer_dict"]

    for group_custom, group_baseline in zip(groups_custom[:-2], groups_baseline):
        assert len(group_custom["params"]) == len(group_baseline["params"])
        assert group_custom.keys() == group_baseline.keys()

        for key in group_custom:
            if key == "params":
                continue
            assert torch.allclose(torch.tensor(group_custom[key]), torch.tensor(group_baseline[key])), (
                f"mismatch on {key!r}"
            )

        for param_custom, param_baseline in zip(group_custom["params"], group_baseline["params"]):
            assert isinstance(param_custom, torch.nn.Parameter)
            assert isinstance(param_baseline, torch.nn.Parameter)
            assert torch.allclose(param_custom, param_baseline)
            assert param_custom.device == param_baseline.device
            assert param_custom.requires_grad == param_baseline.requires_grad


def test_extra_groups_have_expected_shape(trainer: gt.train.Trainer) -> None:
    _, kwargs = trainer.get_optimizer_cls_and_kwargs(trainer.args)
    extra = kwargs["optimizer_dict"][-2:]

    for group in extra:
        assert len(group["params"]) == 1
        assert group["lr"] == 0.0002
        assert group["weight_decay"] == 0.0
        assert group["params"][0].requires_grad
