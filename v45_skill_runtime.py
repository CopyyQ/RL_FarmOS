import numpy as np
import torch

from continuous_runtime import ContinuousRuntime
from kaggrl.clock import resolve_clock
from kaggrl.v4_farm_supervisor import (
    MICRO_TASKS,
    compile_micro_task,
    compile_v46_skill_task,
    micro_local_features,
    micro_plant_context_features,
    state_aware_market_orders,
    apply_feasible_investment_overlay,
    apply_portfolio_switch_overlay,
)


class V45SkillRuntime:
    """One-way V4.5 cutover from macro opening to learned state-aware skills."""

    def __init__(
        self,
        parent_path,
        snapshot_path,
        *,
        skill_cutover_step=720,
        skill_keep_penalty=4.50,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
        enable_investment_overlay=False,
        investment_crop_improvement_ratio=1.20,
        investment_filter_animals=True,
        enable_portfolio_switch=False,
        portfolio_crop_improvement_ratio=1.30,
        portfolio_feed_reserve_days=2.0,
        portfolio_activation_step=144,
        portfolio_horizon_extra_days=2.0,
        portfolio_min_undersupply_ratio=0.0,
        portfolio_min_shop_demand=1.0,
        portfolio_source_mode="any",
        portfolio_objective="best_crop_roi",
        portfolio_forced_animal=None,
        portfolio_animal_min_roi=1.0,
        portfolio_animal_payback_margin=0.0,
        portfolio_diversity_penalty=0.20,
        **kwargs,
    ):
        self.base = ContinuousRuntime(parent_path, snapshot_path, **kwargs)
        self.skill_cutover_step = max(0, int(skill_cutover_step))
        self.skill_keep_penalty = max(0.0, float(skill_keep_penalty))
        self.skill_shadow_start_step = max(0, int(skill_shadow_start_step))
        self.skill_confidence_threshold = float(
            max(0.0, min(0.999, skill_confidence_threshold))
        )
        self.enable_investment_overlay = bool(enable_investment_overlay)
        self.investment_crop_improvement_ratio = max(1.0, float(investment_crop_improvement_ratio))
        self.investment_filter_animals = bool(investment_filter_animals)
        self.enable_portfolio_switch = bool(enable_portfolio_switch)
        self.portfolio_crop_improvement_ratio = max(
            1.0, float(portfolio_crop_improvement_ratio)
        )
        self.portfolio_feed_reserve_days = max(
            0.0, float(portfolio_feed_reserve_days)
        )
        self.portfolio_activation_step = max(
            0, int(portfolio_activation_step)
        )
        self.portfolio_horizon_extra_days = max(
            0.0, float(portfolio_horizon_extra_days)
        )
        self.portfolio_min_undersupply_ratio = max(
            0.0, float(portfolio_min_undersupply_ratio)
        )
        self.portfolio_min_shop_demand = max(
            0.0, float(portfolio_min_shop_demand)
        )
        self.portfolio_source_mode = str(portfolio_source_mode)
        self.portfolio_objective = str(portfolio_objective)
        self.portfolio_forced_animal = (
            None if portfolio_forced_animal is None
            else str(portfolio_forced_animal)
        )
        self.portfolio_animal_min_roi = max(
            0.0, float(portfolio_animal_min_roi)
        )
        self.portfolio_animal_payback_margin = float(
            portfolio_animal_payback_margin
        )
        self.portfolio_diversity_penalty = max(
            0.0, float(portfolio_diversity_penalty)
        )
        self.skill_records = []
        self.skill_teacher_records = []
        self.skill_exec_trace = []
        self.skill_active = {}
        self.skill_mode_started = False
        self.skill_stats = {
            "samples": 0,
            "keep_samples": 0,
            "nonkeep_proposals": 0,
            "confident_nonkeep": 0,
            "semantic_safe_nonkeep": 0,
            "executed_sampled": 0,
            "executed_actions": 0,
        }
        self.investment_overlay_stats = {
            "applied_steps": 0, "plant_switches": 0, "plant_drops": 0,
            "seed_switches": 0, "seed_drops": 0, "animal_buy_drops": 0,
        }
        self.portfolio_switch_stats = {
            "applied_steps": 0,
            "plant_switches": 0,
            "seed_switches": 0,
            "animal_switches": 0,
        }
        self.portfolio_switch_events = []

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def records(self):
        # Route PPO owns only the macro opening. After cutover all worker
        # movement is controlled by the skill policy and route credit stops.
        return [
            r for r in self.base.records
            if int(r.get("step", 0)) < self.skill_cutover_step
        ]

    @property
    def micro_records(self):
        return list(self.base.micro_records) + list(self.skill_records)

    @property
    def teacher_records(self):
        return list(self.skill_teacher_records)

    def reset(self):
        self.base.reset()
        self.skill_records = []
        self.skill_teacher_records = []
        self.skill_exec_trace = []
        self.skill_active = {}
        self.skill_mode_started = False
        self.skill_stats = {
            "samples": 0,
            "keep_samples": 0,
            "nonkeep_proposals": 0,
            "confident_nonkeep": 0,
            "semantic_safe_nonkeep": 0,
            "executed_sampled": 0,
            "executed_actions": 0,
        }
        self.investment_overlay_stats = {
            "applied_steps": 0, "plant_switches": 0, "plant_drops": 0,
            "seed_switches": 0, "seed_drops": 0, "animal_buy_drops": 0,
        }
        self.portfolio_switch_stats = {
            "applied_steps": 0,
            "plant_switches": 0,
            "seed_switches": 0,
            "animal_switches": 0,
        }
        self.portfolio_switch_events = []

    def __call__(self, observation, configuration, context):
        # Keep the V4.4 strategic encoder alive so hidden/economic/opponent
        # context is refreshed every frame. Route records are filtered above
        # after cutover, so stale macro choices receive no post-cutover PPO.
        return self.base(observation, configuration, context)

    @staticmethod
    def _unit_action(action, unit_idx):
        if int(unit_idx) == -1:
            raw = (action or {}).get("farmer") or ["PASS"]
        else:
            hands = list((action or {}).get("hands") or [])
            raw = hands[int(unit_idx)] if int(unit_idx) < len(hands) else ["PASS"]
        return list(raw) if isinstance(raw, (list, tuple)) and raw else ["PASS"]

    @staticmethod
    def _semantic_task_from_action(macro_action):
        action = list(macro_action or ["PASS"])
        op = str(action[0])
        if op == "PASS":
            return "KEEP"
        direct = {
            "WATER": "WATER",
            "FEED": "FEED",
            "CARE": "CARE",
            "COLLECT_FERTILIZER": "COLLECT_FERT",
            "FERTILIZE": "FERTILIZE",
            "HARVEST": "HARVEST",
            "DIG": "DIG_WEED",
            "PLANT": "PLANT_BEST",
            "BUILD_COOP": "DEPLOY_ANIMAL",
            "BUILD_PASTURE": "DEPLOY_ANIMAL",
        }
        if op in direct:
            return direct[op]
        if op == "PICKUP" and len(action) >= 2:
            item = str(action[1])
            if item == "WHEAT":
                return "FETCH_WHEAT"
            if item in {"GOOSE", "COW", "SHEEP"}:
                return "DEPLOY_ANIMAL"
            if item == "FERTILIZER":
                return "COLLECT_FERT"
        if op == "PLACE" and len(action) >= 2:
            item = str(action[1])
            if item in {"GOOSE", "COW", "SHEEP"}:
                return "DEPLOY_ANIMAL"
            return "STORE_SHED"
        return None

    @staticmethod
    def _route_unit_action(row, unit_idx):
        if int(unit_idx) == -1:
            raw = (row or {}).get("farmer") or ["PASS"]
        else:
            hands = list((row or {}).get("hands") or [])
            raw = hands[int(unit_idx)] if int(unit_idx) < len(hands) else ["PASS"]
        return list(raw) if isinstance(raw, (list, tuple)) and raw else ["PASS"]

    def _future_teacher_task(self, observation, selected_route, step, unit_idx):
        route = self.base.market_macro.routes.get(int(selected_route))
        if not route:
            return None
        limit = min(len(route), int(step) + 49)
        for future_step in range(int(step) + 1, limit):
            future_action = self._route_unit_action(route[future_step], unit_idx)
            op = str(future_action[0])
            if op in {"PASS", "NORTH", "SOUTH", "EAST", "WEST"}:
                continue
            task_name = self._semantic_task_from_action(future_action)
            # Stop at the first real operation. If it is unsupported (e.g.
            # BUILD_*), do not mislabel the path as a later unrelated skill.
            if task_name is None:
                return None
            _, meta = compile_v46_skill_task(observation, unit_idx, task_name)
            return task_name if bool(meta.get("valid", False)) else None
        return None

    def _teacher_task(
        self, observation, unit_idx, macro_action, *, selected_route=None, step=None
    ):
        macro_action = list(macro_action or ["PASS"])
        op = str(macro_action[0])
        if op == "PASS":
            return 0
        if op in {"NORTH", "SOUTH", "EAST", "WEST"}:
            if selected_route is None or step is None:
                return None
            task_name = self._future_teacher_task(
                observation, selected_route, step, unit_idx
            )
            return (
                MICRO_TASKS.index(task_name)
                if task_name in MICRO_TASKS
                else None
            )
        task_name = self._semantic_task_from_action(macro_action)
        if task_name not in MICRO_TASKS:
            return None
        _, meta = compile_v46_skill_task(observation, unit_idx, task_name)
        return MICRO_TASKS.index(task_name) if bool(meta.get("valid", False)) else None

    def _collect_teacher_records(
        self, observation, units, action, step, h, c, selected_route
    ):
        if step < self.skill_shadow_start_step:
            return 0
        # Sample every micro interval, but always keep direct operational
        # actions because they are the highest-value demonstrations.
        added = 0
        for unit_idx in units:
            macro_action = self._unit_action(action, unit_idx)
            direct_op = str(macro_action[0]) not in {
                "PASS", "NORTH", "SOUTH", "EAST", "WEST"
            }
            if step % self.base.micro_decision_every != 0 and not direct_op:
                continue
            teacher_idx = self._teacher_task(
                observation,
                unit_idx,
                macro_action,
                selected_route=selected_route,
                step=step,
            )
            if teacher_idx is None:
                continue
            local_np = np.asarray(
                micro_local_features(observation, unit_idx), dtype=np.float32
            )
            plant_np = np.asarray(
                micro_plant_context_features(observation, unit_idx), dtype=np.float32
            )
            valid = self._valid_task_mask(observation, unit_idx)
            valid[teacher_idx] = True
            task_mask_bits = sum(
                (1 << i) for i, flag in enumerate(valid) if flag
            )
            self.skill_teacher_records.append(
                {
                    "step": int(step),
                    "hand_idx": int(unit_idx),
                    "hidden_f16": h[0].numpy().astype(np.float16).tobytes(),
                    "clock_f16": c[0].numpy().astype(np.float16).tobytes(),
                    "local_f16": local_np.astype(np.float16).tobytes(),
                    "plant_ctx_f16": plant_np.astype(np.float16).tobytes(),
                    "economic_f16": self.base.micro_cache["economic"][0]
                    .numpy().astype(np.float16).tobytes(),
                    "opponent_f16": self.base.micro_cache["opponent"][0]
                    .numpy().astype(np.float16).tobytes(),
                    "teacher_task_action": int(teacher_idx),
                    "task_mask_bits": int(task_mask_bits),
                }
            )
            added += 1
        return added

    def _valid_task_mask(self, observation, unit_idx):
        valid = []
        for task_name in MICRO_TASKS:
            _, meta = compile_v46_skill_task(observation, unit_idx, task_name)
            valid.append(bool(meta.get("valid", False)))
        # KEEP is always available as a safe fallback.
        valid[0] = True
        return valid

    def _sample_task(self, observation, unit_idx, h, c):
        local_np = np.asarray(
            micro_local_features(observation, unit_idx), dtype=np.float32
        )
        plant_np = np.asarray(
            micro_plant_context_features(observation, unit_idx), dtype=np.float32
        )
        local_t = torch.from_numpy(local_np).view(1, -1)
        plant_t = torch.from_numpy(plant_np).view(1, -1)
        valid = self._valid_task_mask(observation, unit_idx)
        task_mask_bits = sum(
            (1 << i) for i, flag in enumerate(valid) if flag
        )
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
            # In full skill mode KEEP should be a fallback, not the default.
            # The penalty is applied only when at least one non-KEEP task is valid.
            if any(valid[1:]):
                logits[:, 0] -= self.skill_keep_penalty
            task_action, logp, entropy = self.base._sample(logits)
            probs = torch.softmax(logits, dim=-1)
        task_idx = int(task_action.item())
        return {
            "confidence": float(probs[0, task_idx].item()),
            "task": str(MICRO_TASKS[task_idx]),
            "task_idx": task_idx,
            "until": int(self.base.micro_cache["step"]) + self.base.micro_decision_every,
            "mask_bits": int(task_mask_bits),
            "local_np": local_np,
            "plant_np": plant_np,
            "old_logp": float(logp.item()),
            "old_value": float(mout["value"][0].item()),
            "entropy": float(entropy.item()),
        }

    def micro_action(self, observation, configuration, selected_route, action):
        step = int(resolve_clock(observation, configuration).step)
        if not self.base.enable_micro_policy or self.base.micro_cache is None:
            return action, {"applied": False, "records": 0, "skill_mode": False}
        if int(self.base.micro_cache.get("step", -1)) != step:
            return action, {"applied": False, "records": 0, "skill_mode": False}

        farms = list(observation.get("farms") or [])
        player = int(observation.get("player", 0) or 0)
        if not (0 <= player < len(farms)):
            return action, {"applied": False, "records": 0, "skill_mode": False}

        hand_count = len(list(farms[player].get("hands") or []))
        units = [-1] + list(range(hand_count))
        h = self.base.micro_cache["hidden"]
        c = self.base.micro_cache["clock"]

        portfolio_switch = {"applied": False}
        if self.enable_portfolio_switch:
            action, portfolio_switch = apply_portfolio_switch_overlay(
                observation,
                action,
                crop_improvement_ratio=self.portfolio_crop_improvement_ratio,
                feed_reserve_days=self.portfolio_feed_reserve_days,
                activation_step=self.portfolio_activation_step,
                projected_horizon_extra_days=self.portfolio_horizon_extra_days,
                min_undersupply_ratio=self.portfolio_min_undersupply_ratio,
                min_shop_demand=self.portfolio_min_shop_demand,
                source_mode=self.portfolio_source_mode,
                objective=self.portfolio_objective,
                forced_animal=self.portfolio_forced_animal,
                animal_min_roi=self.portfolio_animal_min_roi,
                animal_payback_margin=self.portfolio_animal_payback_margin,
                diversity_penalty=self.portfolio_diversity_penalty,
            )
            if bool(portfolio_switch.get("applied", False)):
                self.portfolio_switch_stats["applied_steps"] += 1
            for key in ("plant_switches", "seed_switches", "animal_switches"):
                self.portfolio_switch_stats[key] += int(
                    portfolio_switch.get(key, 0) or 0
                )
            for event in portfolio_switch.get("events", []) or []:
                if len(self.portfolio_switch_events) >= 128:
                    break
                row = dict(event)
                row["step"] = int(step)
                self.portfolio_switch_events.append(row)

        investment_overlay = {"applied": False}
        if self.enable_investment_overlay:
            action, investment_overlay = apply_feasible_investment_overlay(
                observation,
                action,
                crop_improvement_ratio=self.investment_crop_improvement_ratio,
                filter_animal_buys=self.investment_filter_animals,
            )
            if bool(investment_overlay.get("applied", False)):
                self.investment_overlay_stats["applied_steps"] += 1
            for key in (
                "plant_switches", "plant_drops", "seed_switches",
                "seed_drops", "animal_buy_drops",
            ):
                self.investment_overlay_stats[key] += int(
                    investment_overlay.get(key, 0) or 0
                )

        teacher_added = self._collect_teacher_records(
            observation,
            units,
            action,
            step,
            h,
            c,
            selected_route,
        )

        # Shadow phase: macro remains authoritative. Existing V4.4 retired-hand
        # micro policy may still use otherwise-idle labor, but full-worker skills
        # only learn by imitation and cannot break choreography yet.
        if step < self.skill_cutover_step:
            base_action, meta = self.base.micro_action(
                observation, configuration, selected_route, action
            )
            meta = dict(meta or {})
            meta.update(
                {
                    "skill_mode": False,
                    "shadow_teacher_records": int(teacher_added),
                    "cutover_step": int(self.skill_cutover_step),
                    "scheduler": "shadow_macro_teacher",
                    "investment_overlay": dict(investment_overlay),
                    "portfolio_switch": dict(portfolio_switch),
                }
            )
            return base_action, meta

        if not self.skill_mode_started:
            self.skill_active = {}
            self.skill_mode_started = True

        # Confidence-gated takeover keeps the macro action as a per-unit fallback.
        # Market orders also remain on the proven macro/market head until worker
        # skills have earned control; this avoids changing two abstractions at once.
        out = {
            "farmer": list((action or {}).get("farmer") or ["PASS"]),
            "hands": [
                list(x) if isinstance(x, (list, tuple)) and x else ["PASS"]
                for x in ((action or {}).get("hands") or [])
            ],
            "market": [
                list(x) for x in ((action or {}).get("market") or [])
                if isinstance(x, (list, tuple))
            ],
        }
        while len(out["hands"]) < hand_count:
            out["hands"].append(["PASS"])

        day = step // 24
        new_records = 0
        applied = 0
        task_counts = {}
        confident = 0

        for unit_idx in units:
            key = (int(day), int(unit_idx))
            active = self.skill_active.get(key)
            if active is not None:
                _, active_meta = compile_v46_skill_task(
                    observation, unit_idx, str(active["task"])
                )
                if (
                    str(active["task"]) != "KEEP"
                    and not bool(active_meta.get("valid", False))
                ):
                    active = None

            sampled_now = False
            if active is None or step >= int(active.get("until", -1)):
                active = self._sample_task(observation, unit_idx, h, c)
                self.skill_active[key] = active
                sampled_now = True

            task_name = str(active["task"])
            compiled, meta = compile_v46_skill_task(
                observation, unit_idx, task_name
            )
            confidence = float(active.get("confidence", 0.0))
            macro_unit_action = self._unit_action(action, unit_idx)
            macro_op = str(macro_unit_action[0])
            skill_op = str(compiled[0])
            move_ops = {"NORTH", "SOUTH", "EAST", "WEST"}
            teacher_idx = self._teacher_task(
                observation,
                unit_idx,
                macro_unit_action,
                selected_route=selected_route,
                step=step,
            )
            # Strict choreography handoff. A macro route is position-sensitive:
            # taking a different move (even at high confidence) desynchronizes all
            # later tape actions. Therefore learned skills may only reproduce the
            # exact macro movement, or fill a genuine PASS with a same-tile op.
            # Direct macro operations remain authoritative.
            if macro_op == "PASS":
                semantic_safe = skill_op not in move_ops and skill_op != "PASS"
            elif macro_op in move_ops:
                semantic_safe = list(compiled) == list(macro_unit_action)
            else:
                semantic_safe = False
            is_move = skill_op in move_ops
            movement_safe = (
                not is_move
                or list(compiled) == list(macro_unit_action)
            )
            execute_skill = (
                task_name != "KEEP"
                and skill_op != "PASS"
                and bool(meta.get("valid", False))
                and confidence >= self.skill_confidence_threshold
                and semantic_safe
                and movement_safe
            )

            # Decision-level takeover telemetry. Unlike skill_records, these
            # counters include rejected/KEEP proposals, so promotion can measure
            # real policy takeover instead of records/records == 1.0.
            if sampled_now:
                self.skill_stats["samples"] += 1
                if task_name == "KEEP":
                    self.skill_stats["keep_samples"] += 1
                else:
                    self.skill_stats["nonkeep_proposals"] += 1
                    if confidence >= self.skill_confidence_threshold:
                        self.skill_stats["confident_nonkeep"] += 1
                    if (
                        skill_op != "PASS"
                        and bool(meta.get("valid", False))
                        and semantic_safe
                        and movement_safe
                    ):
                        self.skill_stats["semantic_safe_nonkeep"] += 1

            if execute_skill:
                self.skill_stats["executed_actions"] += 1
                if sampled_now:
                    self.skill_stats["executed_sampled"] += 1
                self.skill_exec_trace.append(
                    {
                        "step": int(step),
                        "unit_idx": int(unit_idx),
                        "task": str(task_name),
                        "confidence": float(confidence),
                        "macro_action": list(macro_unit_action),
                        "skill_action": list(compiled),
                    }
                )
                if int(unit_idx) == -1:
                    out["farmer"] = list(compiled)
                else:
                    out["hands"][unit_idx] = list(compiled)
                applied += int(str(compiled[0]) != "PASS")
                confident += 1
                if sampled_now:
                    local_np = active.pop("local_np")
                    plant_np = active.pop("plant_np")
                    self.skill_records.append(
                        {
                            "step": step,
                            "day": int(day),
                            "hand_idx": int(unit_idx),
                            "hidden_f16": h[0].numpy().astype(np.float16).tobytes(),
                            "clock_f16": c[0].numpy().astype(np.float16).tobytes(),
                            "local_f16": local_np.astype(np.float16).tobytes(),
                            "plant_ctx_f16": plant_np.astype(np.float16).tobytes(),
                            "economic_f16": self.base.micro_cache["economic"][0]
                            .numpy().astype(np.float16).tobytes(),
                            "opponent_f16": self.base.micro_cache["opponent"][0]
                            .numpy().astype(np.float16).tobytes(),
                            "task_action": int(active["task_idx"]),
                            "task_mask_bits": int(active["mask_bits"]),
                            "old_logp": float(active["old_logp"]),
                            "old_value": float(active["old_value"]),
                            "entropy": float(active["entropy"]),
                            "skill_mode": True,
                            "confidence": confidence,
                        }
                    )
                    new_records += 1
            elif sampled_now:
                # Re-sample low-confidence choices next frame; do not lock a weak
                # skill for the full micro horizon.
                active["until"] = step + 1

            task_counts[task_name] = task_counts.get(task_name, 0) + 1

        return out, {
            "applied": bool(applied),
            "idle_hands": len(units),
            "overrides": int(applied),
            "records": int(new_records),
            "tasks": task_counts,
            "skill_mode": True,
            "cutover_step": int(self.skill_cutover_step),
            "shadow_teacher_records": int(teacher_added),
            "confident_units": int(confident),
            "market_ops": len(out.get("market") or []),
            "scheduler": "shadow_bc_confidence_gated_skill_ppo_v2",
            "investment_overlay": dict(investment_overlay),
            "portfolio_switch": dict(portfolio_switch),
        }
