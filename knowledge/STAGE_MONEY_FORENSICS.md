# Stage-by-Stage Money Gap Forensics

Last updated: 2026-09-23

This document records the strongest current causal evidence for why the V4.6/V4.7 policy can lose by tens of thousands of coins against v51.

## Executive finding

The catastrophic loss is not mainly caused by micro-action quality, route forcing, harvest volume, selling frequency, or cash starvation.

The dominant failure is:

> The policy uses a relatively fixed production portfolio and does not reallocate production according to the realized shop-demand / scarcity regime.

In the canonical catastrophic seed `23238530`, v51 invests in TOMATO before the price spike, while our policy continues WHEAT / STRAWBERRY / existing animal products.

By the final phase:

- v51 sells **80 TOMATO**
- TOMATO revenue = **65,833**
- our TOMATO revenue = **0**
- final margin = approximately **-68k**
- the 65,833 TOMATO revenue alone explains about **95% of the final-phase cash-gap deterioration**

Both seats reproduce the same pattern, so this is not a seat artifact.

---

## Test setup

Canonical catastrophic seed:
- seed `23238530`
- both seats tested
- seat 0 final margin: approximately `-67,859`
- seat 1 final margin: approximately `-68,771`

Normal comparison seed:
- seed `23238531`
- training-parity seat
- final margin: approximately `-4,229`

Same actor family and v51 opponent were used.

Phases:

1. `A_PRE_SHOP`: days 0–2
2. `B_EARLY_SHOP`: days 3–8
3. `C_SIGNAL_BUILD`: days 9–14
4. `D_PORTFOLIO_DIVERGE`: days 15–20
5. `E_LATENT_GAP`: days 21–24
6. `F_CASH_CONVERSION`: days 25–29

---

## Phase A — days 0–2: opening is nearly symmetric

Catastrophic seed cash-gap change: about `-31`.

Our end portfolio:
- WHEAT 7
- MELON 10
- COW 3
- SHEEP 2

v51:
- WHEAT 7
- MELON 12
- COW 3
- SHEEP 2

This phase does not explain the catastrophe.

There is a small structural opening disadvantage because v51 establishes two more MELON plants.

---

## Phase B — days 3–8: first shop information appears, but behavior is still nearly identical

Catastrophic seed cash-gap change: about `-357`.

Our and v51 portfolios remain extremely similar:
- WHEAT 5
- STRAWBERRY 20
- MELON roughly 10 vs 12
- COW 8
- SHEEP 4

No TOMATO production exists yet.

The first shop regime for seed `23238530` begins to reveal repeated PIZZA / FARMERS-market demand, but the policy has not yet adapted production.

This phase is still too early to explain the large final loss.

---

## Phase C — days 9–14: generic baseline disadvantage appears

Catastrophic seed cash gap:
- start approximately `-388`
- end approximately `-3,581`
- phase gap change approximately **-3,193**

Normal comparison seed shows a very similar phase loss:
- phase gap change approximately **-2,824**

Therefore this roughly 3k disadvantage is mostly a general baseline weakness, not the catastrophe mechanism.

Main realized revenue differences in the catastrophic seed include:
- MELON: ours ~12,749 vs v51 ~14,678
- MILK: ours ~8,501 vs v51 ~9,141
- WHEAT: ours ~1,709 vs v51 ~1,910

This is worth optimizing later, but it does not explain the `-68k` tail.

---

## Phase D — days 15–20: the decisive portfolio divergence begins

This is the most important strategic phase.

Cash gap appears to improve:

- start approximately `-3,581`
- end approximately **+1,402**
- apparent improvement approximately **+4,983**

But the reason is deceptive.

During this phase v51 spends heavily:

- BUY_LAND: ~4,000
- BUY_SEED TOMATO: 10 seeds, ~500
- more HIRE spend
- more fertilizer spend

v51 spends roughly 5.3k more than us during the phase.

Therefore our apparent cash lead is mostly:

> v51 converting cash into future productive assets.

At the end of the phase:

Our crops:
- WHEAT 25
- STRAWBERRY 33
- TOMATO 0

v51:
- WHEAT 25
- STRAWBERRY 33
- TOMATO **10**

TOMATO price has already climbed to about **202**.

This is the critical decision point.

The policy sees a favorable cash position, but economically v51 has made the superior long-horizon investment.

---

## Phase E — days 21–24: hidden economic loss becomes visible before cash collapses

Cash still looks safe:

- start gap approximately `+1,402`
- end gap approximately **+507**
- phase cash loss only ~`-895`

But economic horizon value collapses:

- horizon gap at start near `+204`
- horizon gap at end approximately **-26,097**

So at day 24:

> cash says roughly equal, but productive/economic value says we are already down ~26k.

End portfolio:

Our crops:
- WHEAT 37
- CARROT 4
- STRAWBERRY 17
- TOMATO 0

v51:
- WHEAT 28
- CARROT 13
- STRAWBERRY 17
- TOMATO **10**

Market prices:
- TOMATO ~**531**
- STRAWBERRY ~**1**
- MILK ~11
- WOOL ~1
- WHEAT ~45

This is the clearest proof that the policy is holding the wrong production mix.

---

## Phase F — days 25–29: latent value converts into cash

Cash gap:

- start approximately `+507`
- final approximately **-68,771**
- phase deterioration approximately **-69,278**

Exact realized TOMATO sales by v51:

- units: **80**
- revenue: **65,833**
- average realized price: about **822.9 / unit**

Our TOMATO:
- units: **0**
- revenue: **0**

Thus v51 TOMATO revenue alone explains approximately:

`65,833 / 69,278 ≈ 95%`

of the final-phase cash-gap deterioration.

Our phase-F sales include many low-value products:

- STRAWBERRY: 64 units → ~684 revenue → ~10.7/unit
- MILK: 74 units → ~955 revenue → ~12.9/unit
- WOOL: 44 units → ~60 revenue → ~1.36/unit

v51 instead adds:

- TOMATO: 80 units → ~65,833 revenue → ~822.9/unit

The issue is not that we stopped harvesting or selling.

The issue is that we spent labor harvesting and selling products whose prices had collapsed while v51 had positioned production into a scarce high-value product.

---

## Why seed 23238530 creates a TOMATO regime

Final shop draw:

- PIZZA_SHOP ×3
- FARMERS_MARKET ×3
- PET_CAFE ×2

Approximate per-day town/shop sink at full shop set:

- TOMATO: ~37 units/day
- WHEAT: ~37 units/day
- STRAWBERRY: ~19 units/day
- MILK: ~19 units/day
- CARROT: ~43 units/day
- WOOL: ~1 unit/day

The game removes TOMATO from shared market inventory much faster than the farms initially replenish it.

This pushes TOMATO price:

`60 → 68 → 81 → 202 → 531 → 892`

Our policy never creates TOMATO production.

v51 does.

---

## Why the normal comparison seed does not catastrophically fail

Seed `23238531` final shop draw:

- ICE_CREAM_SHOP
- PIZZA_SHOP
- PET_CAFE
- YARN_STORE ×2
- BRUNCH_SPOT ×2
- SMOOTHIE_SHOP

Approximate final sink:

- TOMATO: ~7/day
- WOOL: ~25/day
- STRAWBERRY: ~25/day
- WHEAT: ~25/day
- MILK: ~19/day

TOMATO price only reaches about **81**, while WOOL reaches about **225**.

Our fixed portfolio happens to include more SHEEP, so in the final phase:

- our WOOL revenue ~7,341
- v51 WOOL revenue ~6,171

Final margin is only about `-4,229`.

This is the mirror-image evidence:

> The current policy can perform reasonably when the realized shop regime happens to reward its fixed portfolio, but it fails badly when the shop regime rewards a product it never pivots into.

---

## What the 100-candidate runtime tournament proved

The 100-candidate runtime pilot searched:

- ACT threshold
- skill confidence threshold
- cutover step

Best observed screening improvement was only about:

- +155 margin/game

Top candidates largely collapsed to the same `cutover=696` behavior.

Therefore:

> Runtime threshold tuning is not the main source of the ~12k average gap and cannot explain the ~68k catastrophe.

Do not spend a large Stage-2 budget on this family unless there is a new reason.

---

## Strongest causal chain

The current best-supported causal chain is:

```text
shop draw creates product-specific sink
        ↓
shared market inventory falls unevenly
        ↓
future price opportunity becomes highly asymmetric
        ↓
v51 invests into the under-supplied demanded product
        ↓
our policy keeps a mostly fixed portfolio
        ↓
v51 temporarily looks worse in cash because it invests
        ↓
our cash-centric policy does not react
        ↓
latent economic gap grows
        ↓
scarce product matures
        ↓
v51 sells at extreme price
        ↓
cash gap explodes late
```

---

## Architectural consequence

The missing capability is not merely a better threshold.

The policy needs an explicit **shop-aware forward portfolio allocator**.

Minimum information required:

1. unlocked shop multiset,
2. per-product sink rate,
3. current shared market inventory,
4. projected inventory at first-yield horizon,
5. current farm production by product,
6. rival visible production by product,
7. crop/animal time-to-first-yield,
8. expected yield before season end,
9. feed reserve requirement,
10. expected sequential sale revenue,
11. capex / labor / land cost,
12. risk of market glut before liquidation.

The decision should compare expected future return of the **next unit of investment**, not just current price.

---

## Immediate next experiment

Do not run another 100-candidate threshold search.

Create 100 economic/portfolio candidates around:

- sink-aware projected price,
- first-yield horizon,
- feed-reserve protection,
- crop switch ROI threshold,
- switch activation day,
- product under-supply ratio,
- critic risk gate,
- market hold/sell horizon.

Candidate promotion target:

- at least +2k/game on fresh validation, or
- a large tail-risk reduction without mean regression.

The first candidate family should specifically test whether the policy can recognize the `23238530` TOMATO regime while leaving `23238531` mostly unchanged.
