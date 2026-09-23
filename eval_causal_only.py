import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"vendor"),str(ROOT),str(ROOT/"src")]
from winner_train import _run_chunk, exact_games, metrics

def main():
    st=torch.load(ROOT/"output"/"winner_v4_latest.pt",map_location="cpu",weights_only=False)
    cfg=dict(st.get("config",{}))
    mem=torch.load(ROOT/"output"/"winner_memory_v4.pt",map_location="cpu",weights_only=False)
    wins=sorted([g for g in mem["games"] if int(g["margin"])>0],key=lambda g:int(g["margin"]),reverse=True)[:3]
    games=[(int(g["seed"]),int(g["seat"])) for g in wins]
    snap=ROOT/"output"/"causal_v4_1_actor.pt"
    common=(str(ROOT/"assets"/"parent_promoted_v2.pt"),str(snap),str(ROOT/"assets"/"v51_main.py"))
    task=(common[0],common[1],common[2],games,False,float(cfg.get("min_temperature",0.9)),float(cfg.get("residual_scale",1.0)),float(cfg.get("base_keep_bias",2.3)),int(cfg.get("decision_every",24)),7771)
    known=_run_chunk(task)
    held=exact_games(ROOT/"assets"/"parent_promoted_v2.pt",snap,ROOT/"assets"/"v51_main.py",list(range(22990000,22990004)),4,False,float(cfg.get("min_temperature",0.9)),float(cfg.get("residual_scale",1.0)),float(cfg.get("base_keep_bias",2.3)),int(cfg.get("decision_every",24)),7772,both_seats=True)
    out={"known":[{"seed":r["seed"],"seat":r["seat"],"margin":r["margin"],"win":r["win"],"own":r["own_money"],"v51":r["v51_money"]} for r in known],"known_metrics":metrics(known),"heldout":[{"seed":r["seed"],"seat":r["seat"],"margin":r["margin"],"win":r["win"],"own":r["own_money"],"v51":r["v51_money"]} for r in held],"heldout_metrics":metrics(held)}
    print("CAUSAL KNOWN",out["known_metrics"],flush=True)
    for x in out["known"]: print(" ",x,flush=True)
    print("CAUSAL HELDOUT",out["heldout_metrics"],flush=True)
    for x in out["heldout"]: print(" ",x,flush=True)
    (ROOT/"output"/"causal_only_eval_v4_1.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
if __name__=="__main__": main()
