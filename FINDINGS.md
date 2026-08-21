# Findings

**Which MPC cost function best reduces traffic shockwaves on a highway merge, using Variable Speed Limit signs?**

QUT capstone (EGH400). SUMO is ground truth; a macroscopic traffic model runs inside a nonlinear MPC controller that sets the speed limits. This document is the standalone record of what was measured. `README.md` covers how to run things.

Status: **results are established but one framing decision is open** — see [What is still open](#what-is-still-open).

---

## Method in one paragraph

A fixed SUMO lane-merge scenario (nine 200 m two-lane segments feeding a 500 m one-lane bottleneck, ~4000 veh/h for 1000 s) breaks down and forms a shockwave. A nonlinear MPC (CasADi + IPOPT) reads segment densities from SUMO every 30 s, predicts 210 s ahead with a macroscopic model, and posts new speed limits. Swapping only the cost function isolates its effect. Every result is **paired on random seed** against a no-control run of the same seed, with a two-sided Wilcoxon signed-rank test over 10 seeds. Scoring is always on SUMO output, never on the model that chose the control.

**Primary metric** — vehicle-weighted acceleration RMS (Zegeye et al. 2013, eqs. 16–19), the shockwave measure. **Secondary** — total time spent (delay), TET/TIT surrogate safety, CO₂.

---

## 1. The prediction model dominated everything

The single largest effect found. The controller's model must actually predict SUMO over the 210 s horizon it optimises across. The model in use for most of the project did not.

R² against SUMO at 210 s, versus a "assume nothing changes" persistence reference:

| prediction model | R² density | R² speed | verdict |
|---|---|---|---|
| METANET (as originally configured) | +0.16 | +0.07 | **loses to persistence** |
| Bounded-METANET (Zhao, Roncoli & Yildirimoglu 2025, TR-C) | +0.46 | **+0.65** | beats persistence |
| Asymmetric-anticipation METANET (bespoke) | **+0.54** | +0.57 | beats persistence |
| *persistence reference* | +0.16 | +0.47 | — |

The original model was optimising over a 210 s forecast that carried no more information than assuming nothing happens. **A cost-function comparison run through such a model is not measuring the cost functions.**

Both replacements are parity-tested against their independent reference implementations to ~1e-12 (`validation/test_bounded_parity.py`, `validation/test_antic_parity.py`). Bounded-METANET is the published, citable option and is the project's chosen model; the bespoke variant is retained as a robustness check.

**Consequence: the cost-function ranking is a property of the (model, cost) pair, not of the cost function.** Run on both models, the rankings do not agree — each model's best cost is not the other's. Any claim of the form "cost function X is best" must state the model.

## 2. The reference matters as much as the result

Scoring against *no control* flatters anything that merely lowers the speed limit — a constant sign would do the same for free. Measured on the same 10 seeds:

| policy | shockwave | p | delay |
|---|---|---|---|
| **constant 84 km/h sign, no controller at all** | **−5.7%** | **0.004** | +2.2% |
| best of 18 cost functions (`accel_delay`) | −5.9% | 0.002 | +1.6% |

A dumb constant sign matched the entire MPC. Every claim in this project is therefore reported against **both** no-control and the constant-sign reference. Note also that **throughput is 1112 vehicles for every policy tested** — all demand eventually clears — so throughput cannot discriminate between policies in this scenario and is never quoted as if it could.

## 3. The speed limit was not actually doing anything

The model let the sign in as `Veq = min(V_fd(ρ), (1+α)·u)` with α = +0.1 — i.e. it assumed drivers travel at **1.10×** the posted limit. Measured in SUMO, they travel at **0.89×**:

```
edge      posted  achieved  ratio
n5n6        93.6      82.8   0.885
n8n9        99.0      88.4   0.893
```

The cap was 24% too permissive, so the fundamental-diagram term always won and the sign was **inert on over 99% of segment-steps**. Confirmed directly on the model's Jacobian: `dVeq/du` was *exactly* 0.0000 at every density when the sign sat at the legal limit — a gradient method had no descent direction at all.

Four fixes that forced the sign to move (start it low, raise its rate limit, cap it, drop the smoothness penalty) all raised its activity to 20–48% and **all made results worse**, while destabilising the solver. The plateau was a symptom of the optimiser correctly finding nothing to gain, not a bug hiding a gain.

The fix that works is structural: `vsl_scaled=True` makes the cap `(u / u_legal) · v_free`, so the posted limit scales free-flow speed proportionally. At the legal limit nothing changes; below it the cap binds immediately at every segment, with no dead zone. It is one-way by construction (a sign can only slow traffic) and correctly inert inside a queue, where no sign re-accelerates anyone.

## 4. Headline result

Bounded-METANET + scaled VSL cap + `accel_delay` cost, 10 seeds, paired:

| | shockwave | p | delay | sign active | solver failures |
|---|---|---|---|---|---|
| **Bounded-METANET (published)** | **−15.9%** | **0.002** | +13.8% | 22% of steps | 10% |
| asymmetric-anticipation (bespoke) | −17.6% | 0.002 | +12.9% | 18% of steps | 6% |
| constant-sign reference | −5.7% | 0.004 | +2.2% | — | — |

Best cost functions on the bespoke model without the actuator fix, for comparison:

| cost function | shockwave | p | delay |
|---|---|---|---|
| `accel_delay` | −5.9% | 0.002 | +1.6% |
| `accel_tts_queue` | −5.2% | 0.002 | +2.1% |
| `accel_dens_tts` | −4.9% | 0.004 | +2.7% |
| `accel_dvsl` | −4.8% | 0.049 | +2.1% |

Cost functions combining an **acceleration term with a delay term** consistently top the ranking. Acceleration alone is degenerate (uniformly slow traffic has no acceleration); delay alone does nothing.

## 5. Honest nulls

Two results that do not support intuitive claims, reported because they were measured:

**Safety.** 70 instrumented runs, six cost functions, 10 seeds, paired. **No cost function moves TET or TIT significantly** (every p ≥ 0.23), against no-control medians of TET 10,392 s and TIT 2,295 s. A real −5.9% shockwave reduction bought no measurable safety gain. There is a directional pattern — across the six costs, shockwave and safety move together (r = +0.86, p = 0.027) — but with six points that is suggestive, not established.

**Emissions.** Across 28 cost/model combinations, **not one significant CO₂ change**; the whole range is −2.4% to +3.2%, and shockwave reduction is uncorrelated with it (r = −0.07, p = 0.81). The standard argument that smoother flow reduces fuel from harsh acceleration is **not supported by this scenario's data**.

## 6. A metric correction that matters

The Zegeye acceleration metric was being computed at a 1 s step on 200 m cells. Its spatial term is only an acceleration if a vehicle crosses one cell per step (the CFL condition `T·v_free ≈ L`); at 90 km/h a car covers 25 m in a second, not 200, so the condition was violated **8×** and a spatial *gradient* was being read as an acceleration.

| | 1 s (as originally used) | 8 s (CFL-valid) |
|---|---|---|
| peak "acceleration" | 25.0 m/s² (2.5 g) | 3.1 m/s² |
| share of metric above 4.5 m/s² | 82% | 0% |
| accel RMS, no control | 0.943 | **0.331** |
| measured controller effect | −15.5% | **−17.3%** |

**Relative comparisons survived** — the effect is slightly larger when computed properly — so rankings stand. **Absolute values did not**: figures stored before 2026-08-21 are ~2.85× inflated and must not be quoted. `validation/control_authority.py` now aggregates to 8 s by default.

---

## What is still open

**The −16% is smoother driving, not less congestion.** The density field shows the jam *displaced upstream and stretched*, not removed: peak density 185 → 185 veh/km/lane, share of the field above 60 veh/km/lane essentially unchanged, and the road clears later (~1750 s → ~1950 s). The controller converts stop-start traffic into uniformly slow traffic — which is what the acceleration metric rewards, and it costs ~13% delay.

This is a genuine and statistically strong effect, but it is a *smoothness/comfort* result, not a *congestion* result. Which claim the thesis makes determines which configuration and which figures are correct, and that decision is not yet taken.

Also outstanding: the controller does not release the signs when the road drains (it stops solving but leaves the last limits posted); TET/TIT have ~23% seed variation so 30 seeds would be needed to resolve few-percent safety effects; and the full 18-cost sweep has not yet been run on Bounded-METANET with the actuator fix.

---

## Reproducing

```bash
python validation/test_bounded_parity.py        # model port is correct
python validation/prediction_skill.py --model bounded --seeds 3
python experiments/fixed_vsl.py --seeds 10      # the constant-sign reference
python control/phase_d.py --screen --seeds 10   # the cost-function sweep
python control/compare_models.py                # the comparison tables
python control/plot_density_compare.py --fix bm_scaled --seed 5
```

Result tables live in `data/*.json` and are tracked, so the tables above can be re-derived without re-running SUMO. `data/sumo_baseline.npz` is a fixed ground-truth baseline and is never regenerated.
