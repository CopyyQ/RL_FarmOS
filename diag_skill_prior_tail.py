from pathlib import Path
import copy
import torch
from winner_train import exact_games, skill_teacher_records_to_batch, skill_bc_update
from v45_skill_runtime import V45SkillRuntime
from kaggrl.v4_farm_supervisor import MICRO_TASKS

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/"assets/parent_promoted_v2.pt"
SNAP=ROOT/"runs/v46_econ_shadow/winner_v4_latest.pt"
OPP=ROOT/"assets/v51_main.py"
VAL_SEEDS=[26092407,26092410,26092417,26092420,26092427,26092430]
NATURAL=[1950,36410,12170,9663,9756,2849,14386,2374,4083,773,6690,1440]
POWERS=(0.75,0.85,1.00)

def make_actor_optimizer(device,checkpoint):
    rt=V45SkillRuntime(str(PARENT),str(SNAP),stochastic=False,temperature=1.0,
        residual_scale=1.0,base_keep_bias=2.3,decision_every=24,
        allowed_market_modes=None,seed=123,skill_cutover_step=720,
        skill_keep_penalty=2.75,skill_shadow_start_step=0,
        skill_confidence_threshold=0.70)
    actor=rt.base.actor.to(device)
    prefixes=("micro_trunk.","micro_task.","micro_value.")
    params=[p for n,p in actor.named_parameters() if n.startswith(prefixes)]
    opt=torch.optim.AdamW(params,lr=0.000125,weight_decay=1e-5)
    if checkpoint.get("micro_optimizer_state"):
        opt.load_state_dict(copy.deepcopy(checkpoint["micro_optimizer_state"]))
        for g in opt.param_groups:g["lr"]=0.000125
    return actor,opt

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows=exact_games(PARENT,SNAP,OPP,VAL_SEEDS,6,False,1.0,1.0,2.3,24,1056,
        both_seats=False,skill_cutover_step=720,skill_keep_penalty=2.75,
        skill_shadow_start_step=0,skill_confidence_threshold=0.70)
    vals=[r for g in rows for r in (g.get("skill_teacher_records") or [])]
    replay=torch.load("/tmp/v46_prior_ab_teacher.pt",map_location="cpu",weights_only=False)["records"]
    dims=torch.load(PARENT,map_location="cpu",weights_only=False)
    hdim=int(dims["hidden_dim"]);cdim=int(dims["clock_dim"])
    val_batch=skill_teacher_records_to_batch([{"skill_teacher_records":vals}],hdim,cdim,device)
    checkpoint=torch.load(SNAP,map_location="cpu",weights_only=False)
    for power in POWERS:
        actor,opt=make_actor_optimizer(device,checkpoint)
        tb=skill_teacher_records_to_batch([{"skill_teacher_records":replay}],hdim,cdim,device,
            natural_counts=NATURAL,prior_power=power)
        torch.manual_seed(456789)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(456789)
        skill_bc_update(actor,tb,opt,temperature=1.0,epochs=4,minibatch=512,
            keep_penalty=2.75,update=True)
        v=skill_bc_update(actor,val_batch,opt,temperature=1.0,epochs=0,minibatch=1024,
            keep_penalty=2.75,update=False)
        wi=MICRO_TASKS.index("WATER");fi=MICRO_TASKS.index("FERTILIZE")
        hi=MICRO_TASKS.index("HARVEST");di=MICRO_TASKS.index("DIG_WEED")
        print("RESULT",power,"acc",round(v["acc"],4),"nonkeep",round(v["nonkeep_acc"],4),
              "core_f1",round(v["core_min_f1"],4),
              "WATER",tuple(round(float(x),4) for x in (v["per_task_precision"][wi],v["per_task_recall"][wi],v["per_task_f1"][wi])),
              "FERT",tuple(round(float(x),4) for x in (v["per_task_precision"][fi],v["per_task_recall"][fi],v["per_task_f1"][fi])),
              "HARV",round(float(v["per_task_f1"][hi]),4),
              "DIG",round(float(v["per_task_f1"][di]),4),flush=True)
        pairs=[]
        for t,row in enumerate(v["confusion"]):
            for p,c in enumerate(row):
                if t!=p and c:pairs.append((int(c),t,p))
        pairs.sort(reverse=True)
        print("CONF",power,"/".join(f"{MICRO_TASKS[t]}->{MICRO_TASKS[p]}:{c}" for c,t,p in pairs[:6]),flush=True)

if __name__=="__main__":
    main()
