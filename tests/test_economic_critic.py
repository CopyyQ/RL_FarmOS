from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import torch

from kaggrl.constants import PRODUCTS
from kaggrl.economic_critic import (
    ECON_CRITIC_FEATURE_DIM,
    EconomicCritic,
    build_critic_examples,
    economic_critic_loss,
    economic_feature_names,
    encode_economic_snapshot,
    examples_to_tensors,
)


def make_snapshot(step, own_cash, rival_cash):
    empty_int = {product: 0 for product in PRODUCTS}
    inv = {product: 10000 for product in PRODUCTS}
    ratios = {product: 1.0 for product in PRODUCTS}
    output = {product: 2.0 for product in PRODUCTS}
    return {
        "step": step,
        "own_cash": own_cash,
        "rival_cash": rival_cash,
        "cash_margin": own_cash - rival_cash,
        "own_shed": dict(empty_int),
        "market_inventory": inv,
        "scarcity_ratio": ratios,
        "shop_sink": dict(empty_int),
        "own_visible_output": output,
        "rival_visible_output": output,
        "own_liquidation_now": 1000,
        "own_current_net_worth": own_cash + 1000,
        "own_horizon_liquidation": 4000,
        "rival_visible_horizon_liquidation": 5000,
        "own_horizon_value": own_cash + 4000,
        "rival_visible_horizon_value": rival_cash + 5000,
        "visible_horizon_advantage": (
            own_cash + 4000 - rival_cash - 5000
        ),
    }


def make_trace():
    trace = []
    economic_steps = {0, 24, 72, 144}
    for step in range(201):
        own = 10000 + 20 * step
        rival = 12000 + 35 * step
        trace.append(
            {
                "step": step,
                "own_money": own,
                "rival_money": rival,
                "margin": own - rival,
                "economic": (
                    make_snapshot(step, own, rival)
                    if step in economic_steps
                    else None
                ),
            }
        )
    return trace


def test_feature_shape_is_stable():
    snapshot = make_snapshot(24, 11000, 13000)
    features = encode_economic_snapshot(snapshot)
    assert len(features) == ECON_CRITIC_FEATURE_DIM
    assert len(features) == len(economic_feature_names())
    assert ECON_CRITIC_FEATURE_DIM == 66


def test_targets_and_horizon_masks():
    examples = build_critic_examples(make_trace())
    assert [row["step"] for row in examples] == [0, 24, 72, 144]

    first = examples[0]
    # own +24: +480, rival +24: +840
    assert abs(first["regression"][0] - 0.0048) < 1e-7
    assert abs(first["regression"][1] - 0.0084) < 1e-7
    assert first["regression_mask"] == [1.0] * 7

    last = examples[-1]
    # At step 144, +72 and +144 exceed final step 200.
    assert last["regression_mask"] == [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0
    ]
    # Synthetic final margin is below -20k? No.
    assert last["risk"] == [0.0, 0.0, 0.0]


def test_critic_forward_and_loss_backward():
    examples = build_critic_examples(make_trace())
    batch = examples_to_tensors(examples)
    model = EconomicCritic(hidden_dim=32)
    out = model(batch["features"])
    assert out.regression.shape == (4, 7)
    assert out.risk_logits.shape == (4, 3)

    losses = economic_critic_loss(model, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grad_norm = sum(
        float(parameter.grad.abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert grad_norm > 0.0


if __name__ == "__main__":
    test_feature_shape_is_stable()
    test_targets_and_horizon_masks()
    test_critic_forward_and_loss_backward()
    print("ECONOMIC_CRITIC_TESTS_OK")
