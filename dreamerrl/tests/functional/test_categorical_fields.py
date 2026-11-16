"""Entity-field loss budgets, masked labels, independent vocabularies, and schema identity."""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerrl.env.cage2.categorical_schema import encode_host_fields, make_cage2_decoder, salient_host_fields
from dreamerrl.models.categorical_fields import (
    CategoricalField,
    CategoricalFieldDecoder,
    CategoricalFieldHead,
    CategoricalLossConfig,
    categorical_field_loss,
)


def test_group_budget_survives_background_replication_and_controls_gradients():
    for count in (2, 100):
        ce = torch.tensor([9.0] + [1.0] * (count - 1), requires_grad=True)
        salient = torch.arange(count) == 0
        reduced = categorical_field_loss(
            ce,
            torch.zeros(count),
            salient=salient,
            config=CategoricalLossConfig(balance="groups", salient_fraction=0.5),
        )
        assert reduced.loss == 5
        assert reduced.metrics["salient_loss_mass"] == 0.5
        reduced.loss.backward()
        assert ce.grad is not None
        torch.testing.assert_close(ce.grad[0], torch.tensor(0.5))
        torch.testing.assert_close(ce.grad[1:].sum(), torch.tensor(0.5))
    uniform = categorical_field_loss(torch.tensor([9.0] + [1.0] * 99), torch.zeros(100))
    torch.testing.assert_close(uniform.loss, torch.tensor(1.08))


@pytest.mark.parametrize("salient", [False, True])
def test_single_present_group_keeps_full_loss_scale(salient):
    ce = torch.tensor([2.0, 4.0], requires_grad=True)
    reduced = categorical_field_loss(
        ce,
        torch.zeros(2),
        salient=torch.full((2,), salient),
        config=CategoricalLossConfig(balance="groups", salient_fraction=0.2),
    )
    assert reduced.loss == 3
    assert reduced.metrics["salient_loss_mass"] == int(salient)


def test_masked_missing_labels_are_not_classes_and_have_no_gradient():
    head = CategoricalFieldHead(3, CategoricalField("test", 2))
    features = torch.randn(2, 2, 3, requires_grad=True)
    target = torch.tensor([[0.0, float("nan")], [1.0, -1.0]])
    mask = torch.tensor([[True, False], [True, False]])
    ce = head.per_entity_loss(features, target, mask)
    loss = categorical_field_loss(ce, target, observed=mask).loss
    expected = F.cross_entropy(head(features[mask]), target[mask].long())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert features.grad is not None
    assert features.grad[~mask].count_nonzero() == 0
    assert features.grad[mask].abs().sum() > 0
    empty = torch.zeros_like(mask)
    zero = head.per_entity_loss(features.detach(), target, empty)
    head.zero_grad()
    categorical_field_loss(zero, target, observed=empty).loss.backward()
    assert head.weight.grad is not None and head.weight.grad.count_nonzero() == 0


def test_fields_have_independent_vocabularies_weights_and_class_normalization():
    fields = (CategoricalField("activity", 3), CategoricalField("compromised", 4))
    decoder = CategoricalFieldDecoder(5, fields)
    features = torch.randn(1, 2, 5, requires_grad=True)
    targets = {"activity": torch.tensor([[0, -1]]), "compromised": torch.tensor([[1, 3]])}
    observed = {"activity": torch.tensor([[True, False]]), "compromised": torch.ones(1, 2, dtype=torch.bool)}
    configs = {
        "activity": CategoricalLossConfig(field_weight=2),
        "compromised": CategoricalLossConfig(field_weight=0.25, class_weights=(1, 2, 3, 4)),
    }
    loss, metrics = decoder.loss(features, targets, observed=observed, configs=configs)
    logits = decoder(features)
    activity = F.cross_entropy(logits["activity"][:, 0], torch.tensor([0]))
    compromised = F.cross_entropy(
        logits["compromised"].flatten(0, 1), targets["compromised"].flatten(), reduction="none"
    )
    expected = 2 * activity + 0.25 * (compromised * torch.tensor([2.0, 4.0])).sum() / 6
    torch.testing.assert_close(loss, expected)
    assert metrics["activity"]["observed_entities"] == 1
    assert metrics["compromised"]["observed_entities"] == 2
    loss.backward()
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_cage2_unknown_is_valid_and_host_order_is_explicit():
    targets, masks = encode_host_fields(
        {
            "host1": {"Activity": "None", "Compromised": "Unknown"},
            "host2": {"Activity": "Exploit", "Compromised": "Privileged"},
        },
        ["host2", "host1", "host3"],
    )
    assert targets["Activity"].tolist() == [2, 0, -1]
    assert targets["Compromised"].tolist() == [3, 1, -1]
    assert masks["Compromised"].tolist() == [True, True, False]
    salient = salient_host_fields(targets, masks)
    assert salient["Activity"].tolist() == [True, False, False]
    assert salient["Compromised"].tolist() == [True, True, False]
    decoder = make_cage2_decoder(4)
    features = torch.randn(3, 4, requires_grad=True)
    logits = decoder(features)
    assert logits["Activity"].shape == (3, 3)
    assert logits["Compromised"].shape == (3, 4)
    loss, _ = decoder.loss(features, targets, observed=masks)
    loss.backward()
    assert features.grad is not None
    assert features.grad[2].count_nonzero() == 0
    assert features.grad[1].abs().sum() > 0


def test_field_head_keeps_linear_checkpoint_layout():
    legacy = nn.Linear(4, 7)
    head = CategoricalFieldHead(4, CategoricalField("glyph", 7))
    head.load_state_dict(legacy.state_dict(), strict=True)
    assert set(head.state_dict()) == {"weight", "bias"}
    features = torch.randn(5, 4)
    torch.testing.assert_close(head(features), legacy(features))


@pytest.mark.parametrize(
    "config",
    [
        dict(balance="bad"),
        dict(field_weight=-1),
        dict(salient_fraction=0),
        dict(salient_fraction=1),
        dict(balance="groups", salient_weight=4),
        dict(class_weights=(0, 0)),
        dict(class_weights=(1, math.nan)),
    ],
)
def test_invalid_loss_configuration_fails(config):
    with pytest.raises(ValueError):
        CategoricalLossConfig(**config)


def test_cage2_schema_rejects_dropped_or_duplicate_host_identity():
    with pytest.raises(ValueError, match="host_order"):
        encode_host_fields({"host1": {"Activity": "Scan"}}, ["host2"])
    with pytest.raises(ValueError, match="unique"):
        encode_host_fields({}, ["host1", "host1"])
