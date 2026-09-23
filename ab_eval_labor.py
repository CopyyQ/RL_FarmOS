import json, multiprocessing as mp, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"vendor"),str(ROOT),str(ROOT/"src")]
from kaggle_environments import make
from continuous_runtime import ContinuousRuntime
from rollout.v4_hybrid_agent import V4HybridRolloutAgent

def run(task):
    seed,seat,weed,late_hire,snapshot=task
    rt=ContinuousRuntime(
        ROOT/"assets"/"parent_promoted_v2.pt",
        snapshot,
        stochastic=False,
        temperature=0.90,
        residual_scale=1.0,
        base_keep_bias=2.3,
        decision_every=24,
        seed=seed*13+seat,
    )
    learner=V4HybridRolloutAgent(
        option_policy=rt,
        min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=weed,
        enable_late_hire_pruning=late_hire,
    )
    opp=str(ROOT/"assets"/"v51_main.py")
    agents=[learner,opp] if seat==0 else [opp,learner]
    env=make("kaggriculture",configuration={"seed":int(seed),"episodeSteps":720},debug=False)
    env.run(agents)
    f=env.steps[-1][0].observation.farms
    own=int(f[seat].money); rival=int(f[1-seat].money)
    return {"seed":seed,"seat":seat,"weed":weed,"late_hire":late_hire,"margin":own-rival,"own":own,"v51":rival,"win":int(own>rival)}

def metrics(rows):
    m=[r["margin"] for r in rows]
    return {"games":len(rows),"wins":sum(r["win"] for r in rows),"mean_margin":sum(m)/len(m),"min":min(m),"max":max(m)}

def main():
    mem=torch.load(ROOT/"output"/"winner_memory_v4.pt",map_location="cpu",weights_only=False)
    wins=sorted([g for g in mem["games"] if int(g["margin"])>0],key=lambda g:int(g["margin"]),reverse=True)[:3]
    games=[(int(g["seed"]),int(g["seat"])) for g in wins]
    games += [(s,seat) for s in range(22990000,22990004) for seat in (0,1)]
    snapshot=ROOT/"output"/"causal_v4_1_actor.pt"
    modes=[
        ("baseline",False,False),
        ("weed",True,False),
        ("weed_latehire",True,True),
    ]
    out={}
    ctx=mp.get_context("spawn")
    for name,weed,late in modes:
        tasks=[(s,seat,weed,late,snapshot) for s,seat in games]
        with ctx.Pool(processes=6) as pool:
            rows=list(pool.imap_unordered(run,tasks,chunksize=1))
        known=[r for r in rows if (r["seed"],r["seat"]) in games[:3]]
        held=[r for r in rows if (r["seed"],r["seat"]) not in games[:3]]
        out[name]={"known":sorted(known,key=lambda r:(r["seed"],r["seat"])),"known_metrics":metrics(known),"heldout":sorted(held,key=lambda r:(r["seed"],r["seat"])),"heldout_metrics":metrics(held)}
        print(name,out[name]["known_metrics"],out[name]["heldout_metrics"],flush=True)
    (ROOT/"output"/"labor_ab_v4_1.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
if __name__=="__main__":
    mp.freeze_support(); main()
