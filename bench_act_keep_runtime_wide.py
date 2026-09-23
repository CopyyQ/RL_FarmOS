from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT / "src"), str(ROOT)]

from kaggle_environments import make
from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from v45_skill_runtime import V45SkillRuntime
from v45_skill_runtime_actkeep import V45SkillActKeepRuntime
from kaggrl.v4_farm_supervisor import farm_transition_reward

PARENT = str(ROOT / "assets" / "parent_promoted_v2.pt")
SNAP = "/tmp/farmos_actkeep_wide_snapshot.pt"
GATE = str(ROOT / "runs" / "v46_econ_shadow" / "act_keep_gate_sweep_best.pt")
OPP = str(ROOT / "assets" / "v51_main.py")


def one(args):
    mode, seed, seat = args
    cls = V45SkillRuntime if mode == "flat" else V45SkillActKeepRuntime
    kwargs = {}
    if mode != "flat":
        kwargs["act_keep_gate_path"] = GATE
        kwargs["act_keep_threshold"] = float(mode.replace("gate", "")) / 100.0
    rt = cls(
        PARENT,
        SNAP,
        stochastic=False,
        temperature=1.0,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        allowed_market_modes=None,
        seed=seed + seat,
        skill_cutover_step=672,
        skill_confidence_threshold=0.70,
        **kwargs,
    )
    agent = V4HybridRolloutAgent(
        option_policy=rt,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
    )
    env = make(
        "kaggriculture",
        configuration={"seed": seed, "episodeSteps": 720},
        debug=False,
    )
    agents = [agent, OPP] if seat == 0 else [OPP, agent]
    env.run(agents)
    farms = env.steps[-1][0].observation.farms
    own = int(farms[seat].money)
    rival = int(farms[1 - seat].money)
    deaths = 0
    escapes = 0
    for i in range(len(env.steps) - 1):
        _, st = farm_transition_reward(
            env.steps[i][seat].observation,
            env.steps[i + 1][seat].observation,
            env.steps[i + 1][seat].action or {},
            seat,
        )
        deaths += int(st.get("plant_deaths", 0))
        escapes += int(st.get("animal_escapes", 0))
    skill_records = [
        r for r in rt.micro_records if r.get("skill_mode", False)
    ]
    stats = dict(getattr(rt, "skill_stats", {}) or {})
    return {
        "mode": mode,
        "seed": seed,
        "seat": seat,
        "margin": own - rival,
        "deaths": deaths,
        "escapes": escapes,
        "records": len(skill_records),
        "samples": int(stats.get("samples", 0)),
        "proposals": int(stats.get("nonkeep_proposals", 0)),
        "executed": int(stats.get("executed_sampled", 0)),
    }


def main():
    modes = ("flat", "gate55")
    fresh_seeds = list(range(24510101, 24510109))
    cases = []
    for mode in modes:
        cases.append((mode, 23205684, 1))
        for seed in fresh_seeds:
            for seat in (0, 1):
                cases.append((mode, seed, seat))
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        rows = pool.map(one, cases)

    for mode in modes:
        xs = [r for r in rows if r["mode"] == mode]
        winner = next(r for r in xs if r["seed"] == 23205684)
        fresh = [r for r in xs if r["seed"] != 23205684]
        mean = sum(r["margin"] for r in fresh) / len(fresh)
        deaths = sum(r["deaths"] for r in fresh) / len(fresh)
        escapes = sum(r["escapes"] for r in fresh) / len(fresh)
        records = sum(r["records"] for r in fresh) / len(fresh)
        samples = sum(r["samples"] for r in fresh)
        proposals = sum(r["proposals"] for r in fresh)
        executed = sum(r["executed"] for r in fresh)
        takeover = executed / max(1, proposals)
        print(
            "MODE", mode,
            "WINNER", winner["margin"],
            "FRESH_MEAN", round(mean, 1),
            "PLANT_DEATH", round(deaths, 1),
            "ESCAPE", round(escapes, 1),
            "SKILL_RECORDS", round(records, 1),
            "SAMPLES", samples,
            "PROPOSALS", proposals,
            "EXECUTED", executed,
            "TAKEOVER", round(takeover, 4),
            flush=True,
        )


if __name__ == "__main__":
    main()
