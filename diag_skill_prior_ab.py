from pathlib import Path
import copy
import random
import tempfile
import torch

from winner_train import (
    exact_games, split_skill_teacher_rows, update_skill_teacher_memory,
    skill_teacher_class_counts, skill_teacher_records_to_batch, skill_bc_update,
)
from v45_skill_runtime import V45SkillRuntime
from kaggrl.v4_farm_supervisor import MICRO_TASKS

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/"assets/parent_promoted_v2.pt"
SNAP=ROOT/"runs/v46_econ_shadow/winner_v4_latest.pt"
OPP=ROOT/"assets/v51_main.py"
SEEDS=list(range(26092401,26092433))
POWERS=(0.0,0.35,0.50,0.65)

def make_actor_optimizer(device, checkpoint):
    rt=V45SkillRuntime(
        str(PARENT),str(SNAP),stochastic=False,temperature=1.0,
        residual_scale=1.0,base_keep_bias=2.3,decision_every=24,
        allowed_market_modes=None,seed=123,skill_cutover_step=720,
        skill_keep_penalty=2.75,skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
    )
    actor=rt.base.actor.to(device)
    prefixes=("micro_trunk.","micro_task.","micro_value.")
    params=[p for n,p in actor.named_parameters() if n.startswith(prefixes)]
    opt=torch.optim.AdamW(params,lr=0.000125,weight_decay=1e-5)
    state=checkpoint.get("micro_optimizer_state")
    if state:
        opt.load_state_dict(copy.deepcopy(state))
        for group in opt.param_groups:
            group["lr"]=0.000125
    return actor,opt

def summarize(tag,val):
    print("RESULT",tag,
          "acc",round(val["acc"],4),
          "nonkeep",round(val["nonkeep_acc"],4),
          "core_recall",round(val["core_min_acc"],4),
          "core_f1",round(val["core_min_f1"],4),flush=True)
    focus=("KEEP","WATER","FERTILIZE","HARVEST","DIG_WEED","DEPLOY_ANIMAL")
    for name in focus:
        i=MICRO_TASKS.index(name)
        print("FOCUS",tag,name,
              "n",int(val["target_counts"][i]),
              "P",round(float(val["per_task_precision"][i]),4),
              "R",round(float(val["per_task_recall"][i]),4),
              "F1",round(float(val["per_task_f1"][i]),4),flush=True)
    pairs=[]
    for t,row in enumerate(val["confusion"]):
        for p,c in enumerate(row):
            if t!=p and c:
                pairs.append((int(c),t,p))
    pairs.sort(reverse=True)
    print("CONF",tag,"/".join(
        f"{MICRO_TASKS[t]}->{MICRO_TASKS[p]}:{c}"
        for c,t,p in pairs[:10]
    ),flush=True)

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE",device,flush=True)
    rows=exact_games(
        PARENT,SNAP,OPP,SEEDS,32,False,1.0,1.0,2.3,24,1056,
        both_seats=False,skill_cutover_step=720,skill_keep_penalty=2.75,
        skill_shadow_start_step=0,skill_confidence_threshold=0.70,
    )
    train_rows,val_rows=split_skill_teacher_rows(rows)
    natural=skill_teacher_class_counts(train_rows)
    print("SPLIT","train",len(train_rows),"val",len(val_rows),flush=True)
    print("NATURAL","/".join(f"{MICRO_TASKS[i]}:{c}" for i,c in enumerate(natural)),flush=True)

    tmp=Path("/tmp/v46_prior_ab_teacher.pt")
    if tmp.exists():
        tmp.unlink()
    replay,replay_counts=update_skill_teacher_memory(
        tmp,train_rows,per_class_limit=2048,iteration=1056
    )
    print("REPLAY","/".join(f"{MICRO_TASKS[i]}:{c}" for i,c in enumerate(replay_counts)),flush=True)

    dims=torch.load(PARENT,map_location="cpu",weights_only=False)
    hdim=int(dims["hidden_dim"]); cdim=int(dims["clock_dim"])
    val_records=[r for g in val_rows for r in (g.get("skill_teacher_records") or [])]
    val_batch=skill_teacher_records_to_batch(
        [{"skill_teacher_records":val_records}],hdim,cdim,device
    )
    checkpoint=torch.load(SNAP,map_location="cpu",weights_only=False)

    for power in POWERS:
        actor,opt=make_actor_optimizer(device,checkpoint)
        train_batch=skill_teacher_records_to_batch(
            [{"skill_teacher_records":replay}],hdim,cdim,device,
            natural_counts=(natural if power>0 else None),prior_power=power,
        )
        torch.manual_seed(456789)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(456789)
        train=skill_bc_update(
            actor,train_batch,opt,temperature=1.0,epochs=4,minibatch=512,
            keep_penalty=2.75,update=True,
        )
        val=skill_bc_update(
            actor,val_batch,opt,temperature=1.0,epochs=0,minibatch=1024,
            keep_penalty=2.75,update=False,
        )
        print("TRAIN",power,"loss",round(train["loss"],5),
              "acc",round(train["acc"],4),"core_f1",round(train["core_min_f1"],4),flush=True)
        summarize(f"p{power:.2f}",val)
        del actor,opt,train_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__=="__main__":
    main()
