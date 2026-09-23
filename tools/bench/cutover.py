from __future__ import annotations
import multiprocessing as mp
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT/'src'),str(ROOT/'rollout'),str(ROOT)]
from kaggle_environments import make
from rollout.v4_hybrid_agent import V4HybridRolloutAgent
from v45_skill_runtime import V45SkillRuntime
from kaggrl.v4_farm_supervisor import farm_transition_reward

PARENT=str(ROOT/'assets/parent_promoted_v2.pt')
SNAP=os.environ.get('FARMOS_CHECKPOINT', str(ROOT/'runs/v46_econ_shadow/winner_v45_skill_stage_safe.pt'))
OPP=str(ROOT/'assets/v51_main.py')


def one(args):
    cutover,seed,seat=args
    rt=V45SkillRuntime(
        PARENT,SNAP,stochastic=False,temperature=1.0,
        residual_scale=1.0,base_keep_bias=2.3,decision_every=24,
        allowed_market_modes=None,seed=seed+seat,
        skill_cutover_step=cutover,
    )
    agent=V4HybridRolloutAgent(
        option_policy=rt,min_option_confidence=0.0,
        enable_market_race_ordering=True,
    )
    env=make('kaggriculture',configuration={'seed':seed,'episodeSteps':720},debug=False)
    agents=[agent,OPP] if seat==0 else [OPP,agent]
    env.run(agents)
    farms=env.steps[-1][0].observation.farms
    own=int(farms[seat].money); rival=int(farms[1-seat].money)
    deaths=escapes=0
    for i in range(len(env.steps)-1):
        _,st=farm_transition_reward(
            env.steps[i][seat].observation,
            env.steps[i+1][seat].observation,
            env.steps[i][seat].action or {},seat,
        )
        deaths+=int(st.get('plant_deaths',0)); escapes+=int(st.get('animal_escapes',0))
    skill_records=[r for r in rt.micro_records if r.get('skill_mode',False)]
    skill_nonkeep=sum(int(r.get('task_action',0)) != 0 for r in skill_records)
    ss=dict(getattr(rt,'skill_stats',{}) or {})
    return (
        cutover,seed,seat,own-rival,deaths,escapes,len(skill_records),skill_nonkeep,
        int(ss.get('samples',0)),int(ss.get('nonkeep_proposals',0)),
        int(ss.get('executed_sampled',0)),int(ss.get('keep_samples',0)),
    )


def main():
    cuts=(720,696,672)
    cases=[]
    for c in cuts:
        cases.append((c,23205684,1))
        cases.append((c,24510001,0))
        cases.append((c,24510002,1))
        cases.append((c,24510003,0))
        cases.append((c,24510004,1))
    ctx=mp.get_context('spawn')
    with ctx.Pool(processes=8) as pool:
        rows=pool.map(one,cases)
    for c in cuts:
        xs=[]
        for row in rows:
            if row[0]==c:
                xs.append(row)
        winner=None
        fresh=[]
        for row in xs:
            if row[1]==23205684:
                winner=row
            else:
                fresh.append(row)
        mean=sum(row[3] for row in fresh)/len(fresh)
        death=sum(row[4] for row in fresh)/len(fresh)
        esc=sum(row[5] for row in fresh)/len(fresh)
        skill=sum(row[6] for row in fresh)/len(fresh)
        nonkeep=sum(row[7] for row in fresh)/len(fresh)
        samples=sum(row[8] for row in fresh)/len(fresh)
        proposals=sum(row[9] for row in fresh)/len(fresh)
        executed=sum(row[10] for row in fresh)/len(fresh)
        keep=sum(row[11] for row in fresh)/len(fresh)
        takeover=executed/max(1.0,proposals)
        print(
            'CUT',c,'WINNER',winner[3],'FRESH_MEAN',round(mean,1),
            'PLANT_DEATH',round(death,1),'ESCAPE',round(esc,1),
            'SKILL_RECORDS',round(skill,1),'NONKEEP',round(nonkeep,1),
            'SAMPLES',round(samples,1),'PROPOSALS',round(proposals,1),
            'EXECUTED',round(executed,1),'KEEP',round(keep,1),
            'TAKEOVER',round(takeover,3),flush=True,
        )

if __name__=='__main__':
    main()
