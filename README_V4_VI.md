# WINNER V4 - Winner Amplification

V4 kế thừa V3 nhưng sửa đúng bottleneck đã thấy trong log: V3 tìm được exact win +283 nhưng deterministic validation không học lại được winner.

## Ba chế độ

1. SEARCH
   - chưa có exact winner;
   - exact fresh rollout + PPO + good/near-win replay.

2. AMPLIFY
   - đã có winner nhưng actor deterministic chưa tái hiện;
   - fresh exploration bị PAUSE;
   - scripted exact replay xác minh trajectory;
   - BC winner-only mạnh trên route + market + horizon;
   - exact actor reproduction cùng seed/seat sau mỗi amplification round.

3. EXPLOIT+EXPLORE
   - actor đã tái hiện exact winner;
   - fresh rollout được mở lại;
   - winner/near/good buffer vẫn replay mỗi iteration;
   - winner được recheck định kỳ;
   - nếu quên winner, rollback về winner_v4_reproduced.pt.

## Dùng trực tiếp dữ liệu V3 hiện tại

Lần đầu V4, chạy:

```bat
TRAIN_WINNER_V4_RTX3060.bat --reset --import-v3-output "D:\NguyenAnhQuyet_PLAB\gpu_v51_elite_v3\output" --workers 8 --train-seeds-per-iter 8
```

V4 tự tìm:
- `elite_v3_best.pt` (fallback `elite_v3_latest.pt`)
- `elite_memory_v3.pt`

Nếu memory V3 chứa winner +283, V4 sẽ verify exact trajectory trước, sau đó cố deterministic reproduction ngay trước khi tiếp tục fresh exploration.

Các lần chạy sau chỉ cần:

```bat
TRAIN_WINNER_V4_RTX3060.bat --workers 8 --train-seeds-per-iter 8
```

V4 tự resume `output\winner_v4_latest.pt`.

## Console quan trọng

```text
[WINNER-VERIFY] PASS seed=... seat=... margin=+283
[MODE] ... AMPLIFY_ONLY
[WINNER-AMP] ... teacher_margin=+283 actor_margin=...
[WINNER-GATE] PASS ...
[WINNER-BC] ... acc(route/market/horizon)=...
[WINNER-RECHECK] PASS ...
```

Mục tiêu đầu tiên là thấy:

```text
[WINNER-GATE] PASS
```

Nó có nghĩa actor deterministic đã tự thắng lại đúng seed/seat của winner, không chỉ nhớ offline logits.

## Buffer

V4 chia:
- WIN: margin > 0
- NEAR_WIN: -2000 <= margin <= 0
- GOOD: -8000 <= margin < -2000
- OTHER: < -8000

Winner-only amplification loại bỏ hoàn toàn OTHER/GOOD/NEAR khỏi batch khi đang cố reproduce một winner.

Trong replay thường:
- WIN weight cao nhất
- NEAR_WIN tiếp theo
- GOOD tiếp theo
- OTHER không đi vào Winner-BC.

## Validation

```bat
VALIDATE_V4.bat --seeds 23990000:10 --workers 8
```

Validation exact full-720, both seats, vendored Kaggriculture engine.

Nếu game chạy đúng nhưng chưa 100%:

```text
runtime_status: PASS
target_status: NOT_YET_MET
```

Exit code vẫn = 0.

## Output

- `output/winner_v4_latest.pt` - auto resume
- `output/winner_v4_best.pt` - best fixed held-out eval
- `output/winner_v4_reproduced.pt` - anchor đã exact reproduce winner
- `output/winner_memory_v4.pt` - persistent WIN/NEAR/GOOD memory
- `output/winner_v4_metrics.csv`
- `output/winner_v4_summary.json`
- `output/validation_v4.json`

## Hard gate cuối

20/20 -> 100/100 -> 500/500 exact fresh games, both seats.
