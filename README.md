# MPC for Highway Decongestion and Safety

QUT capstone (EGH400). Reduce traffic shockwaves at a motorway lane drop by
applying Model Predictive Control to Variable Speed Limit signs, and find which
**cost functions** best trade off mobility against safety.

- **SUMO** simulates individual vehicles and is treated as ground truth.
- A **macroscopic model** runs inside the controller to predict traffic cheaply.
- The **MPC** picks speed limits by optimising a cost function over that model.

---

## Quick start

```bash
# 1. Does the model match SUMO?  (uses the cached baseline; ~30 s)
python validation/metanet_vs_sumo.py

# 2. How accurate is it at the horizon the controller uses?  (~2 min)
python validation/metanet_prediction_error.py

# 3. Run the closed loop: SUMO + MPC together              (~5 min)
python control/main_metanet_v1.py
```

Figures are written to `docs/figures/` (created on first run) and result
tables to `data/`.
Scripts can be run from any directory; paths resolve to the project root.

**Requirements:** Python 3.11, SUMO (with `traci` on the path), and
`numpy scipy matplotlib casadi sym-metanet csnlp`.

---

## Where everything lives

| folder | what it is | start here |
|---|---|---|
| [`model/`](model/) | The traffic models. METANET and its variants. | `model/README.md` |
| [`control/`](control/) | The MPC controller and the SUMO closed loop. | `control/README.md` |
| [`validation/`](validation/) | Does the model match SUMO? Figures and error metrics. | `validation/README.md` |
| [`calibration/`](calibration/) | Fitting model parameters to SUMO. | `calibration/README.md` |
| [`experiments/`](experiments/) | Reference implementations of the two replacement models, kept so the parity tests can run. The full bake-off (87 candidates) is not part of this repository. | — |
| [`sumo/`](sumo/) | The SUMO scenario: network, routes, config. | — |
| `data/` | Result tables. `sumo_baseline.npz` is the ground truth and is **fixed**. | — |

Every number quoted in [`FINDINGS.md`](FINDINGS.md) can be re-derived from the
tracked `data/*.json` files without re-running SUMO.

---

## The scenario

A 2.3 km motorway: nine 200 m sections with two lanes, then a 496 m section
with one lane. That lane drop is the bottleneck. Traffic is fed in at
4000 veh/h for 1000 s, then the supply stops.

Two facts about it that are easy to get wrong, and both have cost time before:

- **The run must be at least 2500 s.** SUMO still has 328 vehicles queued at
  t = 1000 s and does not clear until t ≈ 1773 s. Any window ending at 1000 s
  cuts off the entire drain.
- **The declared demand is not the realised demand.** Once the entrance jams,
  SUMO cannot insert vehicles, so actual inflow collapses to near zero and then
  surges again. Feed the model the *measured* inflow, not the nominal 4000.

---

## Current state

| model | open-loop RMSE | prediction @210 s (held out) |
|---|---|---|
| **Bounded-METANET** (recommended) | **18.8 km/h** | 0.784 |
| expA asymmetric anticipation | 26.1 | **0.862** |
| Han first-order | 27.2 | 0.663 |
| original METANET | 36.0 | 0.046 |

The original METANET model predicts poorly at the 210 s horizon the controller
uses. Two replacements were found and both are ported into `model/`; see
[`FINDINGS.md`](FINDINGS.md) §1 for how they were chosen and what changing the
model does to the results.

**Phase D (comparing cost functions) has run.** The model work above was
needed first, because a controller cannot be judged on a model that cannot
predict.

---

## How to do the two things you will actually want to do

### Run SUMO on its own

```bash
python validation/metanet_vs_sumo.py --rerun-sumo
```

Runs SUMO headless for 2500 s with no control, records per-edge density and
speed every second, caches it to `data/sumo_baseline.npz`, and plots it against
the model.

To watch it, open `control/main_metanet_v1.py` and set `USE_GUI = True`, or run
SUMO directly:

```bash
sumo-gui -c sumo/simulation2.sumocfg --end 2500
```

**Do not regenerate `data/sumo_baseline.npz` casually.** Every calibrated model
is scored against that exact file.

### Test a cost function (Phase D)

Eighteen objectives are defined in `control/cost_functions.py`. To add one, add
an entry to the `COSTS` dict — a weighted combination of the terms above it,
each a CasADi expression of density `rho`, speed `v`, queue `w` and the speed
limits `v_ctrl` (the decision variables). Then:

```bash
python control/phase_d.py --screen --seeds 10 --costs my_new_cost
python control/compare_models.py
```

Scoring is always on **SUMO** output, never on the model that chose the control.

Three things already established that will shape any new cost function:

- **Acceleration alone is degenerate** — uniformly slow traffic has no
  acceleration, so it collapses to "slow everything down". It must be paired
  with a delay term. The best objectives found all combine the two.
- **Delay alone does nothing.**
- **Throughput cannot discriminate**: 1112 vehicles get through under every
  policy tested, so it must not be quoted as if it separates them.
- Results must be reported against **both** no-control and the constant-sign
  reference (`experiments/fixed_vsl.py`), because a constant sign matches the
  best controller found. See [`FINDINGS.md`](FINDINGS.md) §2.

---

## Reading order for someone new

1. [`FINDINGS.md`](FINDINGS.md) — what the project set out to answer, what was
   measured, and what is still open. Start here.
2. `model/README.md` — the prediction models and how they differ.
3. `control/README.md` — the MPC and the cost functions.
4. `validation/README.md` — how model accuracy and the shockwave metric are
   measured. Note §6 of FINDINGS: the metric's time step is not a free choice.
