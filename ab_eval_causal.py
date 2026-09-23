import json
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"vendor"))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))

from winner_train import _run_chunk, exact_games, metrics

def explicit_games(snapshot, games, cfg):
    task=(
        str(ROOT/"assets"/"parent_promoted_v2.pt"),
        str(snapshot),
        str(ROOT/"assets"/"v51_main.py"),
        [(int(s),int(seat)) for s,seat in games],
        False,
        float(cfg.get("min_temperature",0.90)),
        float(cfg.get("residual_scale",1.0)),
        float(cfg.get("base_keep_bias",2.3)),
        int(cfg.get("decision_every",24)),
        99991,
    )
    return _run_chunk(task)

def heldout(snapshot,cfg):
    return exact_games(
        ROOT/"assets"/"parent_promoted_v2.pt",
        snapshot,
        ROOT/"assets"/"v51_main.py",
        list(range(22990000,22990004)),
        8,
        False,
        float(cfg.get("min_temperature",0.90)),
        float(cfg.get("residual_scale",1.0)),
        float(cfg.get("base_keep_bias",2.3)),
        int(cfg.get("decision_every",24)),
        99992,
        both_seats=True,
    )

def compact(rows):
    return [
        {
            "seed":r["seed"],"seat":r["seat"],
            "margin":r["margin"],"win":r["win"],
            "own":r["own_money"],"v51":r["v51_money"],
        } for r in rows
    ]

def main():
    latest=torch.load(ROOT/"output"/"winner_v4_latest.pt",map_location="cpu",weights_only=False)
    cfg=dict(latest.get("config",{}))
    mem=torch.load(ROOT/"output"/"winner_memory_v4.pt",map_location="cpu",weights_only=False)
    wins=sorted(
        [g for g in mem["games"] if int(g["margin"])>0],
        key=lambda g:int(g["margin"]),
        reverse=True,
    )[:3]
    games=[(int(g["seed"]),int(g["seat"])) for g in wins]
    snapshots={
        "baseline":ROOT/"output"/"winner_v4_latest.pt",
        "causal":ROOT/"output"/"causal_v4_1_actor.pt",
    }
    result={}
    for name,snapshot in snapshots.items():
        known=explicit_games(snapshot,games,cfg)
        held=heldout(snapshot,cfg)
        result[name]={
            "known":compact(known),
            "known_metrics":metrics(known),
            "heldout":compact(held),
            "heldout_metrics":metrics(held),
        }
        print(name.upper(),"KNOWN",result[name]["known_metrics"],flush=True)
        for x in result[name]["known"]: print(" ",x,flush=True)
        print(name.upper(),"HELDOUT",result[name]["heldout_metrics"],flush=True)
        for x in result[name]["heldout"]: print(" ",x,flush=True)
    out=ROOT/"output"/"causal_ab_eval_v4_1.json"
    out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    b=result["baseline"]["heldout_metrics"]["mean_margin"]
    c=result["causal"]["heldout_metrics"]["mean_margin"]
    kb=result["baseline"]["known_metrics"]["wins"]
    kc=result["causal"]["known_metrics"]["wins"]
    print(f"DELTA heldout_mean_margin={c-b:+.1f} known_wins={kb}->{kc}",flush=True)

if __name__=="__main__":
    main()
