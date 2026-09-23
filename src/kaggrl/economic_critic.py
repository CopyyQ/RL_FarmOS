from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .constants import PRODUCTS
from .v4_market_race import MARKET_PARAMS

MONEY_SCALE = 100000.0
OUTPUT_SCALE = 100.0
SHED_SCALE = 100.0
SINK_SCALE = 72.0

REGRESSION_TARGETS = (
    "own_delta24",
    "rival_delta24",
    "own_delta72",
    "rival_delta72",
    "own_delta144",
    "rival_delta144",
    "terminal_margin",
)
RISK_TARGETS = (
    "risk_below_20k",
    "risk_below_30k",
    "risk_below_50k",
)


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, float(value))))


def _money(value: Any) -> float:
    try:
        return _clip(float(value) / MONEY_SCALE, -5.0, 5.0)
    except Exception:
        return 0.0


def economic_feature_names():
    names = [
        "progress",
        "remaining",
        "own_cash",
        "rival_cash",
        "cash_margin",
        "own_liquidation_now",
        "own_current_net_worth",
        "own_horizon_liquidation",
        "rival_visible_horizon_liquidation",
        "own_horizon_value",
        "rival_visible_horizon_value",
        "visible_horizon_advantage",
    ]
    for product in PRODUCTS:
        key = product.lower()
        names.extend(
            (
                f"{key}_price_ratio",
                f"{key}_inventory_offset",
                f"{key}_shop_sink",
                f"{key}_own_output",
                f"{key}_rival_output",
                f"{key}_own_shed",
            )
        )
    return tuple(names)


ECON_CRITIC_FEATURE_DIM = len(economic_feature_names())


def encode_economic_snapshot(snapshot: dict[str, Any]) -> list[float]:
    step = int(snapshot.get("step", 0) or 0)
    progress = _clip(step / 719.0, 0.0, 1.0)
    features = [
        progress,
        1.0 - progress,
        _money(snapshot.get("own_cash", 0)),
        _money(snapshot.get("rival_cash", 0)),
        _money(snapshot.get("cash_margin", 0)),
        _money(snapshot.get("own_liquidation_now", 0)),
        _money(snapshot.get("own_current_net_worth", 0)),
        _money(snapshot.get("own_horizon_liquidation", 0)),
        _money(snapshot.get("rival_visible_horizon_liquidation", 0)),
        _money(snapshot.get("own_horizon_value", 0)),
        _money(snapshot.get("rival_visible_horizon_value", 0)),
        _money(snapshot.get("visible_horizon_advantage", 0)),
    ]

    ratios = dict(snapshot.get("scarcity_ratio") or {})
    inventory = dict(snapshot.get("market_inventory") or {})
    sink = dict(snapshot.get("shop_sink") or {})
    own_output = dict(snapshot.get("own_visible_output") or {})
    rival_output = dict(snapshot.get("rival_visible_output") or {})
    own_shed = dict(snapshot.get("own_shed") or {})

    for product in PRODUCTS:
        params = MARKET_PARAMS[product]
        i0 = float(params["I0"])
        threshold = max(1.0, float(params["T"]))
        inv = float(inventory.get(product, i0) or 0.0)
        features.extend(
            (
                _clip(float(ratios.get(product, 1.0) or 0.0), 0.0, 10.0),
                _clip((inv - i0) / threshold, -20.0, 20.0),
                _clip(
                    float(sink.get(product, 0.0) or 0.0) / SINK_SCALE,
                    0.0,
                    5.0,
                ),
                _clip(
                    float(own_output.get(product, 0.0) or 0.0)
                    / OUTPUT_SCALE,
                    0.0,
                    10.0,
                ),
                _clip(
                    float(rival_output.get(product, 0.0) or 0.0)
                    / OUTPUT_SCALE,
                    0.0,
                    10.0,
                ),
                _clip(
                    float(own_shed.get(product, 0.0) or 0.0)
                    / SHED_SCALE,
                    0.0,
                    10.0,
                ),
            )
        )
    if len(features) != ECON_CRITIC_FEATURE_DIM:
        raise RuntimeError(
            f"economic critic feature size mismatch: "
            f"{len(features)} != {ECON_CRITIC_FEATURE_DIM}"
        )
    return features


def _future_row(trace, target_step):
    for row in trace:
        if int(row.get("step", -1)) >= int(target_step):
            return row
    return trace[-1]


def build_critic_examples(trace):
    trace = list(trace or [])
    if not trace:
        return []
    trace = sorted(trace, key=lambda row: int(row.get("step", 0)))
    final = trace[-1]
    final_margin = int(final.get("margin", 0) or 0)
    final_step = int(final.get("step", 0) or 0)
    examples = []

    for row in trace:
        snapshot = row.get("economic")
        if not snapshot:
            continue
        step = int(row.get("step", 0) or 0)
        own_now = int(row.get("own_money", 0) or 0)
        rival_now = int(row.get("rival_money", 0) or 0)

        regression = []
        regression_mask = []
        for horizon in (24, 72, 144):
            valid = step + horizon <= final_step
            future = _future_row(trace, step + horizon)
            regression.extend(
                (
                    _money(
                        int(future.get("own_money", 0) or 0) - own_now
                    ),
                    _money(
                        int(future.get("rival_money", 0) or 0)
                        - rival_now
                    ),
                )
            )
            regression_mask.extend((float(valid), float(valid)))

        regression.append(_money(final_margin))
        regression_mask.append(1.0)
        risk = [
            float(final_margin <= -20000),
            float(final_margin <= -30000),
            float(final_margin <= -50000),
        ]

        examples.append(
            {
                "step": step,
                "features": encode_economic_snapshot(snapshot),
                "regression": regression,
                "regression_mask": regression_mask,
                "risk": risk,
                "terminal_margin": final_margin,
                "cash_margin_baseline": int(
                    snapshot.get("cash_margin", row.get("margin", 0)) or 0
                ),
                "visible_horizon_baseline": int(
                    snapshot.get("visible_horizon_advantage", 0) or 0
                ),
            }
        )
    return examples


@dataclass
class EconomicCriticOutput:
    regression: torch.Tensor
    risk_logits: torch.Tensor


class EconomicCritic(nn.Module):
    def __init__(
        self,
        feature_dim=ECON_CRITIC_FEATURE_DIM,
        hidden_dim=96,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.regression_head = nn.Linear(
            int(hidden_dim), len(REGRESSION_TARGETS)
        )
        self.risk_head = nn.Linear(
            int(hidden_dim), len(RISK_TARGETS)
        )

    def forward(self, features):
        hidden = self.trunk(features)
        return EconomicCriticOutput(
            regression=self.regression_head(hidden),
            risk_logits=self.risk_head(hidden),
        )


def examples_to_tensors(examples, device="cpu"):
    if not examples:
        raise ValueError("economic critic examples are empty")
    return {
        "features": torch.tensor(
            [row["features"] for row in examples],
            dtype=torch.float32,
            device=device,
        ),
        "regression": torch.tensor(
            [row["regression"] for row in examples],
            dtype=torch.float32,
            device=device,
        ),
        "regression_mask": torch.tensor(
            [row["regression_mask"] for row in examples],
            dtype=torch.float32,
            device=device,
        ),
        "risk": torch.tensor(
            [row["risk"] for row in examples],
            dtype=torch.float32,
            device=device,
        ),
    }


def economic_critic_loss(
    model,
    batch,
    *,
    regression_weight=1.0,
    risk_weight=0.5,
):
    output = model(batch["features"])
    per_regression = F.smooth_l1_loss(
        output.regression,
        batch["regression"],
        reduction="none",
        beta=0.10,
    )
    mask = batch["regression_mask"]
    regression_loss = (
        (per_regression * mask).sum() / mask.sum().clamp_min(1.0)
    )
    risk_loss = F.binary_cross_entropy_with_logits(
        output.risk_logits,
        batch["risk"],
    )
    loss = (
        float(regression_weight) * regression_loss
        + float(risk_weight) * risk_loss
    )
    return {
        "loss": loss,
        "regression_loss": regression_loss,
        "risk_loss": risk_loss,
        "output": output,
    }


def decode_regression(regression):
    """Convert normalized money predictions back to integer-ish currency."""
    values = regression.detach().cpu()
    return values * MONEY_SCALE
