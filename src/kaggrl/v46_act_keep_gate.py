from __future__ import annotations

import torch
from torch import nn


class ActKeepGate(nn.Module):
    """Tiny binary KEEP-vs-ACT gate for the V4.6 micro skill policy."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def init_act_keep_gate(gate: ActKeepGate) -> None:
    """Deterministic neutral init for checkpoint-compatible migration."""
    with torch.no_grad():
        norm = gate.net[0]
        norm.weight.fill_(1.0)
        norm.bias.zero_()

        first = gate.net[1]
        values = torch.arange(
            first.weight.numel(),
            device=first.weight.device,
            dtype=torch.float32,
        )
        values = torch.sin(values * 0.017 + 3.71) * 0.01
        first.weight.copy_(values.reshape(first.weight.shape).to(first.weight.dtype))
        first.bias.zero_()

        # Neutral at migration: runtime must stay on the legacy flat gate until
        # held-out ACT/KEEP metrics explicitly enable this branch.
        final = gate.net[3]
        final.weight.zero_()
        final.bias.zero_()


def binary_gate_targets(task_target: torch.Tensor) -> torch.Tensor:
    """0=KEEP, 1=ACT for existing MICRO_TASKS labels."""
    return (task_target != 0).long()


def balanced_gate_loss(
    logits: torch.Tensor,
    task_target: torch.Tensor,
    *,
    keep_weight: float = 3.0,
) -> torch.Tensor:
    target = binary_gate_targets(task_target)
    weights = torch.tensor(
        [float(keep_weight), 1.0],
        dtype=logits.dtype,
        device=logits.device,
    )
    return nn.functional.cross_entropy(logits, target, weight=weights)
