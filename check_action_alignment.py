from pathlib import Path
from kaggle_environments import make
from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from v45_skill_runtime import V45SkillRuntime

ROOT=Path(__file__).resolve().parent
PARENT=str(ROOT/"assets/parent_promoted_v2.pt")
SNAP=str(ROOT/"runs/v46_econ_shadow/winner_v4_latest.pt")
OPP=str(ROOT/"assets/v51_main.py")

def pos(farm, idx):
    return farm.farmer if idx==0 else farm.hands[idx-1]

def tile(farm, p):
    return farm.tiles[p[1]][p[0]]

def acts(a):
    a=a or {}
    return [a.get("farmer") or ["PASS"]] + list(a.get("hands") or [])

rt=V45SkillRuntime(PARENT,SNAP,stochastic=False,temperature=1.0,residual_scale=1.0,
    base_keep_bias=2.3,decision_every=24,allowed_market_modes=None,seed=123,
    skill_cutover_step=720,skill_keep_penalty=2.75,skill_shadow_start_step=0,
    skill_confidence_threshold=0.70)
agent=V4HybridRolloutAgent(option_policy=rt,min_option_confidence=0.0,enable_market_race_ordering=True)
env=make("kaggriculture",configuration={"seed":26092301,"episodeSteps":720},debug=False)
env.run([agent,OPP])
seat=0
shown=0
for i in range(len(env.steps)-1):
    ps=env.steps[i][seat]
    cs=env.steps[i+1][seat]
    pf=ps.observation.farms[seat]
    cf=cs.observation.farms[seat]
    pa=acts(getattr(ps,"action",None))
    ca=acts(getattr(cs,"action",None))
    n=min(1+len(pf.hands),1+len(cf.hands))
    for u in range(n):
        p=pos(pf,u)
        q=pos(cf,u)
        if p!=q: continue
        a=tile(pf,p); b=tile(cf,q)
        if isinstance(a,dict) and isinstance(b,dict) and a.get("kind")=="PLANT" and b.get("kind")=="PLANT":
            if not a.get("watered_today",False) and b.get("watered_today",False):
                print("WATER_TRANS",i,"unit",u,"pos",p,"prev_action",pa[u] if u<len(pa) else None,
                      "curr_action",ca[u] if u<len(ca) else None,
                      "before",dict(a),"after",dict(b),flush=True)
                shown+=1
                if shown>=8: raise SystemExit
