from pathlib import Path
import torch

from winner_train import exact_games, skill_teacher_records_to_batch, skill_bc_update
from v45_skill_runtime import V45SkillRuntime
from kaggrl.v4_farm_supervisor import MICRO_TASKS

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/"assets/parent_promoted_v2.pt"
SNAP=ROOT/"runs/v46_econ_shadow/winner_v4_latest.pt"
OPP=ROOT/"assets/v51_main.py"
SEEDS=[26092307,26092310,26092317,26092320,26092327,26092330]

def main():
    rows=exact_games(
        PARENT,SNAP,OPP,SEEDS,6,False,1.0,1.0,2.3,24,1056,
        both_seats=False,skill_cutover_step=720,skill_keep_penalty=2.75,
        skill_shadow_start_step=0,skill_confidence_threshold=0.70,
    )
    records=[r for g in rows for r in (g.get("skill_teacher_records") or [])]
    rt=V45SkillRuntime(str(PARENT),str(SNAP),stochastic=False,temperature=1.0,
        residual_scale=1.0,base_keep_bias=2.3,decision_every=24,
        allowed_market_modes=None,seed=123,skill_cutover_step=720,
        skill_keep_penalty=2.75,skill_shadow_start_step=0,
        skill_confidence_threshold=0.70)
    dims=torch.load(PARENT,map_location="cpu",weights_only=False)
    batch=skill_teacher_records_to_batch([{"skill_teacher_records":records}],
        int(dims["hidden_dim"]),int(dims["clock_dim"]),torch.device("cpu"))
    val=skill_bc_update(rt.base.actor,batch,None,temperature=1.0,epochs=0,
        minibatch=1024,keep_penalty=2.75,update=False)
    print("SUMMARY rows",val["rows"],"acc",round(val["acc"],4),
          "nonkeep",round(val["nonkeep_acc"],4),
          "core_recall",round(val["core_min_acc"],4),
          "core_f1",round(val["core_min_f1"],4),flush=True)
    conf=val["confusion"]
    pairs=[]
    for t,row in enumerate(conf):
        for p,c in enumerate(row):
            if t!=p and c:
                pairs.append((int(c),t,p))
    pairs.sort(reverse=True)
    print("TOP_CONFUSIONS",flush=True)
    for c,t,p in pairs[:30]:
        print(c, MICRO_TASKS[t], "->", MICRO_TASKS[p], flush=True)
    print("TASK_METRICS",flush=True)
    for i,name in enumerate(MICRO_TASKS):
        n=int(val["target_counts"][i])
        if n:
            print(name,"n",n,
                  "P",round(float(val["per_task_precision"][i]),4),
                  "R",round(float(val["per_task_recall"][i]),4),
                  "F1",round(float(val["per_task_f1"][i]),4),flush=True)

if __name__=="__main__":
    main()
