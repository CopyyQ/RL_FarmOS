# FarmOS Core Knowledge — Source of Truth

> **Purpose:** Persistent project memory for future development sessions.
>
> **Rule:** Read this file before making architectural, training, reward, route, market, or portfolio changes.
>
> **Maintenance rule:** When a conclusion is disproven, update or replace it here. Do not preserve a wrong conclusion as if it were still current. Historical failures belong in **Rejected / Superseded Findings** with the evidence that invalidated them.

Last consolidated: 2026-09-23  
Canonical branch: `farmosv1`  
Canonical Git remote: `https://github.com/CopyyQ/RL_FarmOS.git`

---

## 1. Current objective

Build a FarmOS/Kaggriculture CPU-compatible policy that can beat the current v51 opponent reliably and later generalize to multiple strong opponents rather than overfit into a v51-only counter.

Current development order:

1. Preserve stable micro execution.
2. Improve strategic economic understanding.
3. Adapt portfolio after shop reveal.
4. Improve market/endgame monetization.
5. Add risk-aware strategic replanning.
6. Only then do tail-sensitive RL/fine-tuning.
7. Validate on 64–128+ fresh paired games before claiming superiority.

Do **not** redesign the whole model or add a larger model without evidence that representation capacity is the bottleneck.

---

## 2. Stable architecture that should be preserved unless evidence says otherwise

Current useful stack:

- Hierarchical macro policy.
- Macro route / market / horizon decisions.
- ACT/KEEP micro-skill gate.
- Existing learned micro skill execution.
- V4.5/V4.6 skill curriculum.
- CPU-compatible Kaggle runtime.
- Tail-risk observability.
- Shadow economic-value estimator.
- Shadow multi-horizon economic critic.

The micro layer is not currently the main suspected bottleneck. Most recent evidence points to strategic economic allocation and portfolio selection.

---

## 3. Canonical code milestones already promoted

These commits are on `farmosv1` and are considered current:

- `f54d82d` — recover stalled rollout worker pool.
- `b6ac0eb` — tail-risk observability and catastrophe tracing.
- `a7035b9` — shadow economic value estimator.
- `1cc5d57` — shadow multi-horizon economic critic.
- `b807d55` — validate multi-horizon economic critic.

Important: experimental projected-crop / portfolio-switch code is **not yet promoted**. Do not assume it is part of canonical `farmosv1`.

---

## 4. Current benchmark baseline

Most useful fresh baseline currently available:

- Snapshot: current V4.6/V4.7 shadow policy around iteration 1205.
- Seeds: `23239000..23239015`.
- Both seats.
- 32 games total.
- Opponent: v51.

Baseline metrics:

- Mean margin: approximately **-12,258.8**
- Median margin: approximately **-11,181**
- p10 margin: approximately **-20,720.9**
- CVaR10: approximately **-21,063**
- Worst margin: approximately **-21,334**
- Wins: **2 / 32**

This is the current practical reference point for fresh-game improvement.

Do not compare a candidate only against the pathological seed `23238530`. A candidate must also improve or preserve fresh paired performance.

---

## 5. Canonical catastrophe case

The most deeply analyzed catastrophic trajectory is:

- Seed: `23238530`
- Seat: `1`
- Iteration context: `1205`
- Temperature: approximately `1.00242003614173`
- Replay final margin: approximately **-68,771**
- Original observed worst in the live iteration: approximately **-68,674**

Replay reproduced the live catastrophe within about 100 margin points, so the trace is considered representative.

Observed late cash path:

- Day 24: ours ~53,912, v51 ~53,405, gap ~**+507**
- Day 25: gap ~**-355**
- Day 26: gap ~**-9,231**
- Day 27: gap ~**-32,089**
- Day 28: gap ~**-50,332**
- Day 29: gap ~**-68,771**

Key fact: our cash continued to increase. This was **not a simple farm-death or zero-cash collapse**.

---

## 6. Verified economic root-cause findings

### 6.1 Cash margin is an incomplete state signal

On the catastrophic seed, cash could still look healthy while economic value was already deeply negative.

Observed shadow estimates around the late game:

- Day 24 cash gap: about **+350**
- Day 24 visible economic-value gap: about **-15,922**
- Day 25 cash gap: about **+339**
- Day 25 visible economic-value gap: about **-27,116**

Therefore:

> A positive cash gap can hide a large latent productive/value disadvantage.

This is a verified architectural gap in a cash-centric policy.

### 6.2 Economic critic predicts danger earlier than cash

Held-out catastrophic seed `23238530` showed approximately:

- Day 9: `P(final < -20k) > 0.50`
- Day 16: `P(final < -50k) > 0.50`
- Day 19: cash gap still positive, predicted terminal margin around **-46k**
- Day 24: cash gap still positive, predicted terminal margin around **-58.5k**
- Actual final: about **-68.8k**

Validation comparison:

- Economic Critic terminal-margin MAE: approximately **11,735**
- Cash-gap baseline MAE: approximately **21,652**
- Raw economic-gap baseline MAE: approximately **19,783**

So the critic improved terminal-margin prediction by roughly 41–46% relative to these simple baselines.

At a risk threshold around `0.25`, the critic was conservative but high precision in the evaluated set:

- `final < -30k`: precision around **97.1%**
- `final < -50k`: precision around **93.2%**

Recall is not yet strong enough to directly control policy globally. Keep it shadow / advisory until further validation.

### 6.3 Portfolio adaptation after shop reveal is the strongest current causal hypothesis

The most important verified market/portfolio observation:

- In the catastrophic seed, v51 had roughly **10 TOMATO plants** from around day 19.
- Our policy had **0 TOMATO plants** in the same period.
- The town configuration included `PIZZA_SHOP`, creating recurring demand for TOMATO.
- TOMATO price rose strongly, approximately:
  `127 → 202 → 265 → 340 → 429 → 531 → 646 → 732 → 764 → 826`.
- STRAWBERRY, which our policy continued to produce, fell approximately:
  `164 → 120 → 57 → 5 → 1`.

Therefore the core failure is not simply harvest quantity. The two players can harvest/sell similar unit counts while receiving radically different economic value per unit.

### 6.4 The opponent did not simply hoard a huge shed then liquidate

Deep forensic inspection found that shed inventory at sampled late-game boundaries was often near zero for both players.

So the earlier hypothesis:

> “v51 mainly accumulated a huge private inventory and then dumped it at the end”

is **not supported for seed 23238530**.

The stronger explanation is:

> v51 maintained a more valuable productive portfolio and sold high-value output as it was produced, while shop-driven scarcity increased the value of those products.

### 6.5 Market scarcity is nonlinear and shop demand matters much more than the old ROI model represented

Old crop economics approximately used:

`units × current_price × demand_boost`

with a small shop-demand bonus such as roughly:

`1 + 0.12 × demand`.

This is too weak for real market dynamics where shop consumption lowers shared market inventory and `market_price()` can rise nonlinearly.

A forward-priced estimator was implemented experimentally using:

1. current market inventory,
2. future shop/town consumption,
3. expected production from both farms,
4. projected market inventory,
5. `market_price(projected_inventory)`,
6. sequential sale revenue,
7. projected crop ROI.

On seed `23238530`, this estimator ranked **TOMATO #1** repeatedly around days 16–22 while the actual policy continued many WHEAT investments.

This is strong evidence that portfolio selection is under-reacting to future scarcity.

---

## 7. Route findings

### 7.1 There is a real one-time route handoff window

Observed route compatibility:

- Route 0 and route 108 first diverge around step 145.
- At step 144, alternate strategic routes can still be compatible.
- Later, compatibility can shrink until only the inherited route remains.

On seed `23238530` around step 144:

- Current/inherited route: 0
- Macro base route: 121
- `shop_replan=True`
- Actor kept route 0

This means the route system has a narrow handoff window after shop reveal.

### 7.2 But hard-forcing the macro route did NOT solve the catastrophe

Targeted A/B:

- Control mean: about **-25,238**
- Shop-force mean: about **-25,292**
- Catastrophic seed improved only about **+87**

Therefore:

> Route handoff is a real architectural limitation, but it is not sufficient as the root cause of the -68k catastrophe.

Do not hardcode route 121 or any particular route as the fix.

---

## 8. Rejected / superseded findings

### 8.1 “Primary cause is liquidity collapse” — REJECTED

Opening cash can drop near zero as part of normal investment behavior.

The catastrophic seed later recovered cash and continued earning money. The dominant failure was relative economic growth, not simply lack of liquidity.

Classifier logic was corrected so opening investment drawdown is not automatically labeled a liquidity collapse.

### 8.2 “Primary cause is production collapse” — REJECTED for the canonical catastrophe

The catastrophic and normal comparison showed broadly similar:

- animal count,
- plant count,
- harvest activity,
- sell activity,
- no major animal-escape problem.

The value of what was being produced mattered more than raw activity volume.

### 8.3 “Primary cause is failure to sell in the endgame” — REJECTED

Our policy was still harvesting and selling significant unit counts late game.

The issue was mainly value per unit / portfolio mix / market regime, not a total absence of selling.

### 8.4 Hard macro-route force — REJECTED

It did not materially improve the catastrophic case and slightly worsened targeted mean performance.

### 8.5 Global atomic PLANT sanitizer — REJECTED as a policy improvement

The engine has an atomic PLANT rule:

- if PLANT requests for a crop exceed available seeds in a step,
- all requests for that crop can become PASS.

The sanitizer keeps the valid subset and converts the excess to PASS.

Locally this is logically valid, and targeted 10-game tests looked excellent:

- Mean ~`-25,238 → -16,780`
- Worst ~`-68,771 → -31,214`
- `<-50k`: `2 → 0`
- Wins: `0 → 2`

However 32 fresh paired games strongly failed:

Control:
- Mean ~**-12,258.8**
- CVaR10 ~**-21,063**
- Worst ~**-21,334**
- Wins **2**

Atomic sanitizer:
- Mean ~**-14,730.5**
- CVaR10 ~**-42,179**
- Worst ~**-53,940**
- Wins **0**

Paired:
- 12 improved
- 20 worse
- Mean delta approximately **-2,472**
- Worst paired regression approximately **-45,372**

It reduced invalid operations, but performance got worse. Therefore:

> Fewer invalid actions does not necessarily mean a stronger strategic policy.

Do **not** enable the atomic sanitizer globally.

### 8.6 Why atomic sanitizer behaved inconsistently

Forensic comparison found first state divergence around **step 18 / day 0**:

- Control could lose an entire PLANT group because requests exceeded available seed.
- Sanitizer preserved one valid PLANT.
- This created roughly one extra plant at day 0.

That single early plant caused large butterfly effects:

- some seeds improved by tens of thousands,
- other seeds became tens of thousands worse.

Because shop/scarcity regime is not yet known at day 0, there is no reliable causal gate for globally preserving that plant.

### 8.7 Animal-buy filter — REJECTED

Ablation showed the animal-only filter worsened the representative games.

### 8.8 Endgame unviable-PLANT filter — REJECTED

Ablation showed dropping late plant actions globally did not improve the representative set and could worsen performance.

### 8.9 Missing-seed-only filter — NOT PROMOTABLE

It looked strong on targeted seeds, but its effect was equivalent to the atomic sanitizer's early butterfly behavior and did not generalize to the 32-game fresh benchmark.

---

## 9. Current strongest next direction

### Feed-reserve-aware post-shop portfolio adaptation

The experimental crop-switch logic currently overprotects WHEAT whenever animals exist.

That rule is too coarse:

> “has animals” does not imply “every future WHEAT investment must be protected.”

The next design should estimate feed reserve coverage.

Concept:

`available wheat + expected wheat production - expected feed demand`

If feed reserve is safely positive, excess crop investment may switch from WHEAT to a higher projected-ROI crop such as TOMATO.

Important constraints:

- Only apply **after shop reveal**, not in the day-0 opening.
- Preserve the number of investment actions.
- Prefer rewriting `BUY_SEED/PLANT` target crop rather than creating extra actions.
- Require a significant projected ROI advantage, not a tiny improvement.
- Do not cause animal starvation/escape.
- Do not globally turn on atomic PLANT sanitization.
- Do not force a fixed route.
- Keep runtime CPU-compatible.

Suggested conservative switch gate:

1. Shop regime is known.
2. Alternative crop is viable before endgame.
3. Feed reserve remains above a safety buffer.
4. Projected ROI of alternative crop exceeds current crop by a substantial ratio, initially around 1.2–1.35.
5. Action count is unchanged.
6. Required seed acquisition remains feasible.

---

## 10. Projected crop estimator experimental status

Experimental worktree used during analysis:

`/tmp/farmosv1-projected-econ`

This is **not canonical** and may contain rejected experiments mixed together.

Verified useful idea:

- forward-priced crop ROI.

Not yet validated for promotion:

- projected crop switching,
- feed-reserve-aware switching,
- any combined investment overlay.

Do not merge this worktree wholesale.

Port only the smallest candidate that wins fresh paired benchmarks.

---

## 11. Benchmark promotion protocol

Never promote from one pathological seed.

Recommended gates:

### Stage A — targeted
5 representative seeds × 2 seats.

Must include:
- canonical catastrophe,
- normal seeds,
- known regression seeds.

### Stage B — fresh
At least 16 fresh seeds × 2 seats = 32 games.

Reject immediately if:
- mean materially worsens,
- CVaR10 worsens strongly,
- worst tail becomes materially worse,
- wins collapse,
- plant deaths / animal escapes worsen materially.

### Stage C — promotion
64–128+ fresh paired games.

Track:

- mean margin,
- median margin,
- win rate,
- p10,
- CVaR10,
- worst margin,
- count below -20k,
- count below -30k,
- count below -50k,
- plant deaths,
- animal escapes,
- invalid ops,
- harvested units,
- sold units.

A candidate that improves mean but worsens tail substantially is **not** a promotion candidate.

---

## 12. Practical performance milestones

Current fresh mean is about `-12k` vs v51.

Useful milestones:

### Milestone 1
- mean > -8k
- CVaR10 > -18k

### Milestone 2
- mean > -5k
- CVaR10 > -12k

### Milestone 3
- mean > -2k
- catastrophe below -30k close to zero

### Win-ready evidence
- mean around 0 or positive,
- stable win rate increase,
- 128+ fresh paired games,
- no catastrophic safety regressions.

After consistently beating v51, move to a multi-opponent curriculum to avoid v51-specific overfitting.

---

## 13. Runtime / training operational notes

Last verified historical trainer state during this analysis:

- Live trainer had stopped around iteration 1205.
- A launcher attempted to use `../farmos_rl/.venv311/bin/python`, but that path was missing at the time.
- `/home/plab/miniconda3/bin/python3` had working NumPy/PyTorch for QA.

This state can change. Always check current processes and Python environment before resuming training.

Canonical active/research paths historically used:

- Experimental training repo:
  `/home/plab/Desktop/Nguyen_Anh_Quyet_PLAB/AI/gpu_v51_winner_v4`
- Clean maintained repo:
  `/home/plab/Desktop/Nguyen_Anh_Quyet_PLAB/AI/farmosv1`

Do not overwrite dirty experimental worktrees without inspecting them first.

---

## 14. Source-of-truth rules for future sessions

At the beginning of a new development session:

1. Read this file.
2. Check `git log -10 --oneline` on `farmosv1`.
3. Check running trainer/process state.
4. Check latest benchmark artifacts/logs.
5. Treat the sections above as current until contradicted by a stronger benchmark.

When new evidence contradicts this file:

- update the relevant canonical statement,
- move the old conclusion to **Rejected / Superseded Findings**,
- include the benchmark evidence that invalidated it,
- update the current next action,
- commit the knowledge update.

Do not append contradictory conclusions without resolving which one is current.

---

## 15. Latest stage-by-stage forensic finding

See `knowledge/STAGE_MONEY_FORENSICS.md` for the full transaction-level analysis.

Strongest verified result:

- seed `23238530` reproduces roughly `-68k` on both seats;
- v51 builds 10 TOMATO plants while our policy builds 0;
- v51 later sells 80 TOMATO for about **65,833**;
- this accounts for about **95% of the final-phase cash-gap deterioration**;
- before those sales, our cash can still be positive while the economic horizon gap is already around **-26k**;
- seed `23238531` does not create the same TOMATO scarcity regime and finishes around `-4.2k`, showing that the catastrophic failure is strongly regime-dependent.

The 100-candidate ACT/confidence/cutover tournament found only about **+155/game** at best and many top configs collapsed to the same `cutover=696` behavior. Treat this family as a minor tuning axis, not the main gap source.

---

## 16. Current next action

**Build the first 100-candidate economic/portfolio tournament.**

The first candidate family must test shop-aware forward allocation rather than runtime thresholds.

Priority variables:

- product sink rate from the unlocked-shop multiset,
- projected price at first-yield horizon,
- feed reserve days,
- crop-switch ROI threshold,
- activation day / shop-reveal gate,
- under-supply ratio,
- critic risk gate,
- market hold/sell horizon.

Hard design constraints:

- preserve action count where possible,
- do not alter day-0 opening without evidence,
- protect only the WHEAT needed for feed safety,
- do not globally enable rejected atomic PLANT sanitization,
- do not hard-force routes,
- keep Kaggle CPU compatibility.

Primary targeted proof:

- recognize the `23238530` TOMATO regime and materially improve it;
- leave the `23238531` non-TOMATO regime mostly unchanged.

Promotion target:

- at least about +2k/game on fresh validation, or
- a major tail-risk improvement without mean regression.

Only after a winning economic/portfolio family is found should RL reward shaping or larger architectural changes be revisited.
