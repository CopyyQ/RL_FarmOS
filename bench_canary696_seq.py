from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'src'),str(ROOT/'rollout'),str(ROOT)]
from kaggle_environments import make
from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from v45_skill_runtime import V45SkillRuntime
from kaggrl.v4_farm_supervisor import farm_transition_reward

PARENT=str(ROOT/'assets/parent_promoted_v2.pt')
SNAP='/tmp/farmos_v45_conservative_smoke/winner_v4_latest.pt'
OPP=str(ROOT/'assets/v51_main.py')
CASES=[(23205684,1),(24510001,0),(24510002,1),(24510003,0)]
CUTS=(672,)

def run(cutover,seed,seat):
    rt=V45SkillRuntime(
        PARENT,SNAP,stochastic=False,temperature=1.0,
        residual_scale=1.0,base_keep_bias=2.3,decision_every=24,
        allowed_market_modes=None,seed=seed+seat,
        skill_cutover_step=cutover,
        skill_keep_penalty=4.5,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
    )
    agent=V4HybridRolloutAgent(
        option_policy=rt,min_option_confidence=0.0,
        enable_market_race_ordering=True,
    )
    env=make('kaggriculture',configuration={'seed':seed,'episodeSteps':720},debug=False)
    env.run([agent,OPP] if seat==0 else [OPP,agent])
    farms=env.steps[-1][0].observation.farms
    own=int(farms[seat].money); rival=int(farms[1-seat].money)
    deaths=escapes=0
    for i in range(len(env.steps)-1):
        _,st=farm_transition_reward(
            env.steps[i][seat].observation,
            env.steps[i+1][seat].observation,
            env.steps[i][seat].action or {},seat,
        )
        deaths += int(st.get('plant_deaths',0))
        escapes += int(st.get('animal_escapes',0))
    skills=[r for r in rt.micro_records if r.get('skill_mode',False)]
    nonkeep=sum(int(r.get('task_action',0)) != 0 for r in skills)
    return own-rival,deaths,escapes,len(skills),nonkeep

def main():
    rows=[]
    for cut in CUTS:
        for seed,seat in CASES:
            x=run(cut,seed,seat)
            rows.append((cut,seed,seat,*x))
            print('CASE',cut,seed,seat,'margin',x[0],'deaths',x[1],'esc',x[2],'skill',x[3],'nonkeep',x[4],flush=True)
    for cut in CUTS:
        xs=[r for r in rows if r[0]==cut]
        winner=[r for r in xs if r[1]==23205684][0]
        fresh=[r for r in xs if r[1]!=23205684]
        print('SUMMARY',cut,
              'winner',winner[3],
              'fresh_mean',round(sum(r[3] for r in fresh)/len(fresh),1),
              'deaths',round(sum(r[4] for r in fresh)/len(fresh),1),
              'esc',round(sum(r[5] for r in fresh)/len(fresh),1),
              'skill',round(sum(r[6] for r in fresh)/len(fresh),1),
              'nonkeep',round(sum(r[7] for r in fresh)/len(fresh),1),
              flush=True)

if __name__=='__main__':
    main()
