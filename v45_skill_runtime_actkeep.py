import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from kaggrl.v4_farm_supervisor import (
    MICRO_TASKS,
    micro_local_features,
    micro_plant_context_features,
)
from v45_skill_runtime import V45SkillRuntime


def _build_gate(kind, dim):
    if kind == "linear":
        return nn.Linear(dim, 2)
    if kind == "mlp":
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )
    raise ValueError(f"unsupported ACT/KEEP gate kind: {kind}")


class V45SkillActKeepRuntime(V45SkillRuntime):
    """Two-stage skill policy: ACT/KEEP gate, then conditional skill head."""

    def __init__(
        self,
        *args,
        act_keep_gate_path=None,
        act_keep_threshold=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if act_keep_gate_path is None:
            act_keep_gate_path = os.environ.get("FARMOS_ACT_KEEP_GATE_PATH", "")
        if act_keep_threshold is None:
            act_keep_threshold = float(
                os.environ.get("FARMOS_ACT_KEEP_THRESHOLD", "0.50")
            )
        self.act_keep_gate = None
        self.act_keep_threshold = float(act_keep_threshold)
        self.act_keep_gate_meta = None
        if act_keep_gate_path:
            payload = torch.load(
                Path(act_keep_gate_path), map_location="cpu", weights_only=False
            )
            dim = int(payload["input_dim"])
            self.act_keep_gate = _build_gate(str(payload["kind"]), dim)
            self.act_keep_gate.load_state_dict(payload["state_dict"])
            self.act_keep_gate.eval()
            self.act_keep_gate_meta = {
                "kind": str(payload["kind"]),
                "keep_weight": float(payload.get("keep_weight", 1.0)),
                "epoch": int(payload.get("epoch", 0)),
            }

    def _sample_task(self, observation, unit_idx, h, c):
        if self.act_keep_gate is None:
            return super()._sample_task(observation, unit_idx, h, c)
        local_np = np.asarray(
            micro_local_features(observation, unit_idx), dtype=np.float32
        )
        plant_np = np.asarray(
            micro_plant_context_features(observation, unit_idx), dtype=np.float32
        )
        local_t = torch.from_numpy(local_np).view(1, -1)
        plant_t = torch.from_numpy(plant_np).view(1, -1)
        valid = self._valid_task_mask(observation, unit_idx)

        with torch.inference_mode():
            x = torch.cat(
                (
                    h,
                    c,
                    local_t,
                    self.base.micro_cache["economic"],
                    self.base.micro_cache["opponent"],
                ),
                dim=-1,
            )
            z = self.base.actor.micro_trunk(x)
            z = z + torch.tanh(
                self.base.actor.micro_plant_context(plant_t)
            )
            gate_logits = self.act_keep_gate(z)
            gate_probs = torch.softmax(gate_logits, dim=-1)
            act_prob = float(gate_probs[0, 1].item())

        # No valid operational task means KEEP regardless of gate output.
        choose_act = bool(any(valid[1:])) and act_prob >= self.act_keep_threshold
        if not choose_act:
            return {
                "confidence": float(1.0 - act_prob),
                "task": "KEEP",
                "task_idx": 0,
                "until": int(self.base.micro_cache["step"]) + 1,
                "mask_bits": 1,
                "local_np": local_np,
                "plant_np": plant_np,
                "old_logp": 0.0,
                "old_value": 0.0,
                "entropy": 0.0,
                "gate_act_prob": act_prob,
                "gate_decision": "KEEP",
            }

        # Conditional ACT distribution: KEEP is removed entirely so PPO sees
        # the same conditional skill policy that generated the action.
        valid = list(valid)
        valid[0] = False
        task_mask_bits = sum((1 << i) for i, flag in enumerate(valid) if flag)
        with torch.inference_mode():
            mout = self.base.actor.micro_forward(
                h,
                c,
                local_t,
                economic=self.base.micro_cache["economic"],
                opponent=self.base.micro_cache["opponent"],
                plant_context=plant_t,
            )
            logits = mout["task"] / self.base.temperature
            mask = torch.tensor(valid, dtype=torch.bool).view(1, -1)
            logits = logits.masked_fill(
                ~mask, torch.finfo(logits.dtype).min
            )
            task_action, logp, entropy = self.base._sample(logits)
            probs = torch.softmax(logits, dim=-1)

        task_idx = int(task_action.item())
        return {
            "confidence": float(probs[0, task_idx].item()),
            "task": str(MICRO_TASKS[task_idx]),
            "task_idx": task_idx,
            "until": int(self.base.micro_cache["step"])
            + self.base.micro_decision_every,
            "mask_bits": int(task_mask_bits),
            "local_np": local_np,
            "plant_np": plant_np,
            "old_logp": float(logp.item()),
            "old_value": float(mout["value"][0].item()),
            "entropy": float(entropy.item()),
            "gate_act_prob": act_prob,
            "gate_decision": "ACT",
        }
