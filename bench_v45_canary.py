import torch
from pathlib import Path

from winner_train import exact_games, save_snapshot, metrics
from continuous_runtime import ContinuousActor, init_actor, load_actor_state_compatible, load_parent

ROOT = Path(__file__).resolve().parent
TEMP = Path('/tmp/farmos_v4_5_shadow_replay')
SEEDS = [23205684, 26100001, 26100002, 26100003, 26100004, 26100005, 26100006]

def snapshot_from_checkpoint():
    ck = torch.load(TEMP/'winner_v4_latest.pt', map_location='cpu', weights_only=False)
    payload, parent = load_parent(ROOT/'assets/parent_promoted_v2.pt', 'cpu')
    actor = ContinuousActor(
        int(payload['hidden_dim']),
        int(payload['clock_dim']),
        len(payload['route_ids']),
        len(payload['market_modes']),
    )
    init_actor(actor)
    load_actor_state_compatible(actor, ck['actor_state'])
    snap = TEMP/'_canary_actor.pt'
    save_snapshot(actor, snap, ck.get('config') or {})
    return snap

def summarize(rows, cut):
    m = metrics(rows)
    winner = [g for g in rows if int(g['seed']) == 23205684 and int(g['seat']) == 1]
    fresh = [g for g in rows if int(g['seed']) != 23205684]
    fm = metrics(fresh)
    print(
        f'CUT={cut} winner_seat1={winner[0]["margin"] if winner else None:+d} '
        f'fresh_mean={fm["mean_margin"]:+.1f} fresh_median={fm["median_margin"]:+.1f} '
        f'fresh_wins={fm["wins"]}/{fm["games"]} '
        f'plant_deaths={fm["plant_deaths"]} animal_escapes={fm["animal_escapes"]} '
        f'skill_records={fm["skill_records"]} skill_nonkeep={fm["skill_nonkeep"]}'
    )
    margins = [(g['seed'], g['seat'], g['margin']) for g in fresh]
    print('MARGINS', cut, margins)

def main():
    snap = snapshot_from_checkpoint()
    for cut in (672, 648):
        rows = exact_games(
            ROOT/'assets/parent_promoted_v2.pt',
            snap,
            ROOT/'assets/v51_main.py',
            SEEDS,
            8,
            False,
            1.0,
            1.0,
            2.3,
            24,
            1015,
            pool=None,
            both_seats=True,
            skill_cutover_step=cut,
            skill_keep_penalty=4.5,
            skill_shadow_start_step=0,
            skill_confidence_threshold=0.70,
        )
        summarize(rows, cut)

if __name__ == '__main__':
    main()
