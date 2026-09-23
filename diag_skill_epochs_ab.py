from pathlib import Path
import copy, torch
from winner_train import exact_games, skill_teacher_records_to_batch, skill_bc_update
from v45_skill_runtime import V45SkillRuntime
from kaggrl.v4_farm_supervisor import MICRO_TASKS

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/"assets/parent_promoted_v2.pt"; SNAP=ROOT/"runs/v46_econ_shadow/winner_v4_latest.pt"; OPP=ROOT/"assets/v51_main.py"
VAL_SEEDS=[26092407,26092410,26092417,26092420,26092427,26092430]
NATURAL=[1950,36410,12170,9663,9756,2849,14386,2374,4083,773,6690,1440]
EPOCHS=(4,6,8,12)

def make(device,ck):
    rt=V45SkillRuntime(str(PARENT),str(SNAP),stochastic=False,temperature=1.0,residual_scale=1.0,
        base_keep_bias=2.3,decision_every=24,allowed_market_modes=None,seed=123,skill_cutover_step=720,
        skill_keep_penalty=2.75,skill_shadow_start_step=0,skill_confidence_threshold=0.70)
    a=rt.base.actor.to(device)
    ps=[p for n,p in a.named_parameters() if n.startswith(("micro_trunk.","micro_task.","micro_value."))]
    o=torch.optim.AdamW(ps,lr=0.000125,weight_decay=1e-5)
    if ck.get("micro_optimizer_state"):
        o.load_state_dict(copy.deepcopy(ck["micro_optimizer_state"]))
        for g in o.param_groups:g["lr"]=0.000125
    return a,o

def main():
    d=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows=exact_games(PARENT,SNAP,OPP,VAL_SEEDS,6,False,1.0,1.0,2.3,24,1056,both_seats=False,
        skill_cutover_step=720,skill_keep_penalty=2.75,skill_shadow_start_step=0,skill_confidence_threshold=0.70)
    vals=[r for g in rows for r in (g.get("skill_teacher_records") or [])]
    replay=torch.load("/tmp/v46_prior_ab_teacher.pt",map_location="cpu",weights_only=False)["records"]
    dims=torch.load(PARENT,map_location="cpu",weights_only=False); hd=int(dims["hidden_dim"]);cd=int(dims["clock_dim"])
    vb=skill_teacher_records_to_batch([{"skill_teacher_records":vals}],hd,cd,d)
    ck=torch.load(SNAP,map_location="cpu",weights_only=False)
    wi=MICRO_TASKS.index("WATER");fi=MICRO_TASKS.index("FERTILIZE");hi=MICRO_TASKS.index("HARVEST")
    for ep in EPOCHS:
        a,o=make(d,ck)
        tb=skill_teacher_records_to_batch([{"skill_teacher_records":replay}],hd,cd,d,natural_counts=NATURAL,prior_power=1.0)
        torch.manual_seed(456789)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(456789)
        tr=skill_bc_update(a,tb,o,temperature=1.0,epochs=ep,minibatch=512,keep_penalty=2.75,update=True)
        v=skill_bc_update(a,vb,o,temperature=1.0,epochs=0,minibatch=1024,keep_penalty=2.75,update=False)
        def f(i):return (round(float(v["per_task_precision"][i]),4),round(float(v["per_task_recall"][i]),4),round(float(v["per_task_f1"][i]),4))
        print("RESULT ep",ep,"train_loss",round(tr["loss"],5),"acc",round(v["acc"],4),"nonkeep",round(v["nonkeep_acc"],4),
              "core_f1",round(v["core_min_f1"],4),"WATER",f(wi),"FERT",f(fi),"HARV",f(hi),flush=True)
        pairs=[]
        for t,row in enumerate(v["confusion"]):
            for p,c in enumerate(row):
                if t!=p and c:pairs.append((int(c),t,p))
        pairs.sort(reverse=True)
        print("CONF ep",ep,"/".join(f"{MICRO_TASKS[t]}->{MICRO_TASKS[p]}:{c}" for c,t,p in pairs[:5]),flush=True)

if __name__=="__main__":main()
