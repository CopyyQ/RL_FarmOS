import torch
from pathlib import Path
from winner_train import exact_games, save_snapshot
from continuous_runtime import ContinuousActor, init_actor, load_actor_state_compatible, load_parent
from kaggrl.v4_farm_supervisor import MICRO_TASKS

ROOT = Path(__file__).resolve().parent

def main():
    ck = torch.load(ROOT/'output/winner_v4_latest.pt', map_location='cpu', weights_only=False)
    payload, parent = load_parent(ROOT/'assets/parent_promoted_v2.pt', 'cpu')
    actor = ContinuousActor(
        int(payload['hidden_dim']),
        int(payload['clock_dim']),
        len(payload['route_ids']),
        len(payload['market_modes']),
    )
    init_actor(actor)
    load_actor_state_compatible(actor, ck['actor_state'])
    snap = ROOT/'output/_skill_teacher_probe.pt'
    save_snapshot(actor, snap, ck.get('config') or {})
    rows = exact_games(
        ROOT/'assets/parent_promoted_v2.pt',
        snap,
        ROOT/'assets/v51_main.py',
        [25100001,25100002,25100003,25100004],
        4, False, 1.0, 1.0, 2.3, 24, 1013,
        pool=None, both_seats=False,
        skill_cutover_step=720,
        skill_keep_penalty=4.5,
    )
    counts = {k:0 for k in MICRO_TASKS}
    total = 0
    for g in rows:
        for rec in g.get('skill_teacher_records', []):
            idx = int(rec['teacher_task_action'])
            counts[MICRO_TASKS[idx]] += 1
            total += 1
    print('TEACHER_TOTAL', total)
    for k,v in counts.items():
        print(f'{k}={v}')

if __name__ == '__main__':
    main()
