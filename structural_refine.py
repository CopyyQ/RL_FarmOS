import argparse, copy, json, multiprocessing as mp, sys
from collections import Counter
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"vendor"),str(ROOT),str(ROOT/"src")]
from kaggle_environments import make
from continuous_runtime import ScriptedTrajectoryRuntime,HORIZONS
from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from kaggrl.v45_macro_data import load_v45_macro_data

STRUCT_OPS={"DIG","BUILD_PASTURE","BUILD_COOP","FEED","CARE","PLANT"}
HORIZON_SET=(24,48,72,96)

def unit_actions(action):
    return [action.get("farmer"),*(action.get("hands") or [])]

def profile(routes,rid,start,horizon):
    c=Counter()
    route=routes[int(rid)]
    end=min(len(route),int(start)+int(horizon))
    for action in route[int(start):end]:
        for x in unit_actions(action):
            if not x: continue
            op=str(x[0])
            if op=="PLANT" and len(x)>1: c[f"PLANT_{x[1]}"]+=1
            elif op in STRUCT_OPS: c[op]+=1
        for x in action.get("market") or []:
            if not x: continue
            if x[0]=="BUY_ANIMAL" and len(x)>=3:
                c[f"BUY_{x[1]}"]+=int(x[2])
            elif x[0]=="HIRE": c["HIRE"]+=1
    return dict(c)

def structural_score(p):
    return (
        5*p.get("BUY_COW",0)
        +3*p.get("BUILD_PASTURE",0)
        +2*p.get("BUY_SHEEP",0)
        +p.get("DIG",0)
    )

def distance(a,b):
    keys=set(a)|set(b)
    weights={"BUY_COW":8,"BUY_SHEEP":5,"BUY_GOOSE":4,
             "BUILD_PASTURE":5,"BUILD_COOP":5,"DIG":2,"HIRE":1,
             "FEED":1,"CARE":1}
    return sum(abs(a.get(k,0)-b.get(k,0))*weights.get(k,0.5) for k in keys)
def mutate(game,step,rid,horizon,route_to_class):
    g=copy.deepcopy(game)
    found=False
    for r in g["records"]:
        if int(r["step"])!=int(step): continue
        found=True
        r["chosen_route_id"]=int(rid)
        r["route_action"]=int(route_to_class[int(rid)])
        r["horizon"]=int(horizon)
        r["horizon_action"]=HORIZONS.index(int(horizon))
        break
    if not found: raise KeyError(step)
    return g

def run_task(task):
    game,step,rid,horizon,parent,opponent,route_to_class,weed=task
    g=(
        copy.deepcopy(game)
        if int(step) < 0
        else mutate(game,step,rid,horizon,route_to_class)
    )
    rt=ScriptedTrajectoryRuntime(parent,g["records"])
    learner=V4HybridRolloutAgent(
        option_policy=rt,min_option_confidence=0.0,
        enable_market_race_ordering=True,
        enable_idle_weed_labor=bool(weed),
        allow_all_routes=True,
        enable_structural_route_gate=False,
    )
    seat=int(g["seat"]); seed=int(g["seed"]); opp=str(opponent)
    agents=[learner,opp] if seat==0 else [opp,learner]
    env=make("kaggriculture",configuration={"seed":seed,"episodeSteps":720},debug=False)
    env.run(agents)
    f=env.steps[-1][0].observation.farms
    own=int(f[seat].money); rival=int(f[1-seat].money)
    return {"seed":seed,"seat":seat,"step":int(step),"route":int(rid),
            "horizon":int(horizon),"margin":own-rival,"own":own,"v51":rival}

def choose_tasks(winners,routes,route_ids,route_to_class,top_per_direction):
    tasks=[]
    metadata=[]
    for g in winners:
        for rec in g["records"]:
            step=int(rec["step"])
            if step<144 or step>600: continue
            orig=int(rec["chosen_route_id"])
            for horizon in HORIZON_SET:
                base=profile(routes,orig,step,horizon)
                scored=[]
                for rid in route_ids:
                    alt=profile(routes,rid,step,horizon)
                    d=distance(base,alt)
                    if d<=0: continue
                    scored.append((structural_score(alt)-structural_score(base),d,rid,alt))
                positive=sorted([x for x in scored if x[0]>0],key=lambda x:(x[0],x[1]),reverse=True)[:top_per_direction]
                negative=sorted([x for x in scored if x[0]<0],key=lambda x:(x[0],-x[1]))[:top_per_direction]
                diverse=sorted(scored,key=lambda x:x[1],reverse=True)[:1]
                seen={orig}
                for delta,d,rid,alt in positive+negative+diverse:
                    if rid in seen: continue
                    seen.add(rid)
                    tasks.append((g,step,rid,horizon))
                    metadata.append({"seed":int(g["seed"]),"seat":int(g["seat"]),"step":step,
                                     "original_route":orig,"route":rid,"horizon":horizon,
                                     "structural_delta":delta,"distance":d,
                                     "base_profile":base,"alt_profile":alt})
    return tasks,metadata
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--memory",default=str(ROOT/"output"/"winner_memory_v4.pt"))
    ap.add_argument("--top-winners",type=int,default=3)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--top-per-direction",type=int,default=1)
    ap.add_argument(
        "--selection",
        choices=("mixed","cow_more","cow_less","diverse"),
        default="mixed",
    )
    ap.add_argument("--weed",action="store_true")
    ap.add_argument("--max-tasks",type=int,default=0)
    ap.add_argument("--output",default=str(ROOT/"output"/"structural_refine_v4_2.json"))
    args=ap.parse_args()
    mem=torch.load(args.memory,map_location="cpu",weights_only=False)
    winners=sorted([g for g in mem["games"] if int(g["margin"])>0],
                   key=lambda g:int(g["margin"]),reverse=True)[:args.top_winners]
    payload=torch.load(ROOT/"assets"/"parent_promoted_v2.pt",map_location="cpu",weights_only=False)
    route_ids=tuple(int(x) for x in payload["route_ids"])
    route_to_class={r:i for i,r in enumerate(route_ids)}
    routes,_,_=load_v45_macro_data()
    raw,meta=choose_tasks(
        winners,routes,route_ids,route_to_class,args.top_per_direction
    )
    pairs=list(zip(raw,meta))
    if args.selection=="cow_more":
        pairs=[
            pair for pair in pairs
            if float(pair[1]["structural_delta"]) > 0
        ]
        pairs.sort(
            key=lambda pair: (
                float(pair[1]["structural_delta"]),
                float(pair[1]["distance"]),
            ),
            reverse=True,
        )
    elif args.selection=="cow_less":
        pairs=[
            pair for pair in pairs
            if float(pair[1]["structural_delta"]) < 0
        ]
        pairs.sort(
            key=lambda pair: (
                float(pair[1]["structural_delta"]),
                -float(pair[1]["distance"]),
            )
        )
    elif args.selection=="diverse":
        pairs.sort(
            key=lambda pair: float(pair[1]["distance"]),
            reverse=True,
        )
    if int(args.max_tasks) > 0:
        pairs=pairs[:int(args.max_tasks)]
    raw=[pair[0] for pair in pairs]
    meta=[pair[1] for pair in pairs]
    keymeta={(m["seed"],m["seat"],m["step"],m["route"],m["horizon"]):m for m in meta}
    parent=ROOT/"assets"/"parent_promoted_v2.pt"; opponent=ROOT/"assets"/"v51_main.py"
    tasks=[(g,s,r,h,parent,opponent,route_to_class,args.weed) for g,s,r,h in raw]
    baseline_tasks=[
        (g,-1,-1,1,parent,opponent,route_to_class,args.weed)
        for g in winners
    ]
    print(f"STRUCTURAL tasks={len(tasks)} winners={len(winners)} workers={args.workers}",flush=True)
    ctx=mp.get_context("spawn")
    with ctx.Pool(processes=min(args.workers,len(baseline_tasks))) as pool:
        baseline_rows=list(pool.imap_unordered(run_task,baseline_tasks,chunksize=1))
    baseline_map={
        (int(r["seed"]),int(r["seat"])):int(r["margin"])
        for r in baseline_rows
    }
    print(f"BASELINES {baseline_map}",flush=True)
    rows=[]
    with ctx.Pool(processes=args.workers) as pool:
        for i,row in enumerate(pool.imap_unordered(run_task,tasks,chunksize=1),1):
            m=keymeta[(row["seed"],row["seat"],row["step"],row["route"],row["horizon"])]
            row.update(m)
            row["baseline_margin"]=baseline_map[(int(row["seed"]),int(row["seat"]))]
            row["gain"]=int(row["margin"])-int(row["baseline_margin"])
            rows.append(row)
            if i%max(1,args.workers)==0 or i==len(tasks):
                partial={
                    "schema":"farmos_structural_counterfactual_v4_2_partial",
                    "completed":i,
                    "tasks":len(tasks),
                    "baselines":baseline_rows,
                    "rows":rows,
                }
                Path(args.output).write_text(
                    json.dumps(partial,indent=2),encoding="utf-8"
                )
                print(f"[STRUCTURAL] {i}/{len(tasks)}",flush=True)
    by={}
    for r in rows:
        key=(r["seed"],r["seat"],r["step"],r["horizon"])
        by.setdefault(key,[]).append(r)
    summary=[]
    for key,items in by.items():
        best=max(items,key=lambda x:x["margin"])
        summary.append({"seed":key[0],"seat":key[1],"step":key[2],"horizon":key[3],
                        "best_route":best["route"],"best_margin":best["margin"],
                        "baseline_margin":best["baseline_margin"],
                        "gain":best["gain"],
                        "original_route":best["original_route"],
                        "structural_delta":best["structural_delta"],
                        "best_profile":best["alt_profile"],
                        "base_profile":best["base_profile"]})
    out={"schema":"farmos_structural_counterfactual_v4_2",
         "tasks":len(tasks),"winner_count":len(winners),
         "baselines":baseline_rows,"rows":rows,"summary":summary}
    Path(args.output).write_text(json.dumps(out,indent=2),encoding="utf-8")
    for x in sorted(summary,key=lambda x:x["best_margin"],reverse=True)[:20]:
        print("[STRUCT]",x,flush=True)
if __name__=="__main__":
    mp.freeze_support(); main()
