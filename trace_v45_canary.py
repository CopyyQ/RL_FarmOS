import torch
from pathlib import Path
from collections import Counter, defaultdict

from winner_train import exact_games, save_snapshot
from continuous_runtime import ContinuousActor, init_actor, load_actor_state_compatible, load_parent

ROOT = Path(__file__).resolve().parent
TEMP = Path('/tmp/farmos_v4_5_shadow_replay')
SEEDS = [23205684, 26100001, 26100002, 26100003, 26100004]

def main():
    ck = torch.load(TEMP/'winner_v4_latest.pt', map_location='cpu', weights_only=False)
    payload, _ = load_parent(ROOT/'assets/parent_promoted_v2.pt', 'cpu')
    actor = ContinuousActor(
        int(payload['hidden_dim']), int(payload['clock_dim']),
        len(payload['route_ids']), len(payload['market_modes'])
    )
    init_actor(actor)
    load_actor_state_compatible(actor, ck['actor_state'])
    snap = TEMP/'_trace_actor.pt'
    save_snapshot(actor, snap, ck.get('config') or {})

    rows = exact_games(
        ROOT/'assets/parent_promoted_v2.pt', snap, ROOT/'assets/v51_main.py',
        SEEDS, 8, False, 1.0, 1.0, 2.3, 24, 1015,
        pool=None, both_seats=True,
        skill_cutover_step=696,
        skill_keep_penalty=4.5,
        skill_shadow_start_step=0,
        skill_confidence_threshold=0.70,
    )

    task = Counter()
    replaced = Counter()
    skill_ops = Counter()
    conf = defaultdict(list)
    examples = defaultdict(list)
    for g in rows:
        for t in g.get('skill_exec_trace', []):
            name = t['task']
            mop = str((t.get('macro_action') or ['PASS'])[0])
            sop = str((t.get('skill_action') or ['PASS'])[0])
            task[name] += 1
            replaced[(mop, name)] += 1
            skill_ops[sop] += 1
            conf[name].append(float(t['confidence']))
            if len(examples[(mop,name)]) < 4:
                examples[(mop,name)].append((g['seed'],g['seat'],t['step'],t['unit_idx'],t['macro_action'],t['skill_action'],round(t['confidence'],3)))
    print('TASK_COUNTS', dict(task))
    print('SKILL_OPS', dict(skill_ops))
    print('REPLACED_TOP')
    for k,v in replaced.most_common(30):
        print(k, v, 'mean_conf', round(sum(conf[k[1]])/len(conf[k[1]]),3), 'examples', examples[k])
    print('GAME_MARGINS', [(g['seed'],g['seat'],g['margin'],len(g.get('skill_exec_trace',[]))) for g in rows])

if __name__ == '__main__':
    main()
