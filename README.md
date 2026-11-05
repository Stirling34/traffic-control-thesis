# MPC for Highway Decongestion and Safety

QUT capstone (EGH400/EGH490), Joshua Stirling, supervised by Dr Guilherme
Fróes Silva. Nonlinear Model Predictive Control (MPC) sets Variable Speed
Limit (VSL) signs, and the project compares **which MPC cost functions best
reduce traffic shockwaves, and at what cost to mobility and safety**.

**The report is [`Final_Report.pdf`](Final_Report.pdf).** It is the reference
for the method, results and conclusions; this README covers how the code is
laid out and how to reproduce the report. [`EXPERIMENTS.md`](EXPERIMENTS.md)
is the log of every experiment run and what each one found.

- **SUMO** (microscopic) is the ground truth; every result is measured in SUMO.
- **Bounded-METANET** (macroscopic, built with `sym-metanet` on CasADi) is the
  prediction model inside the controller.
- **The NMPC** is CasADi + `csnlp` + IPOPT, talking to SUMO over TraCI.
- **Two roads**: a 2→1 lane merge (standing queue) and a straight two-lane
  road with one slow vehicle (travelling jam). Report §6.1, Table 4.

---

## Setup

```bash
pip install -r requirements.txt      # Python 3.11
```

SUMO 1.24.0 must be installed separately with `sumo` on `PATH`. Developed on
Windows; the launchers in `control/*.ps1` are PowerShell, everything else is
plain Python and can be run from any directory. SUMO's per-run output goes to
`%TEMP%\egh400_sumo_scratch` (keep it out of OneDrive).

---

## Repository layout

| path | what it is |
|---|---|
| [`model/`](model/) | Bounded-METANET prediction model and all calibrated parameter sets |
| [`control/`](control/) | NMPC controller, cost functions, closed-loop runner, experiment runner, report analysis and figures |
| [`validation/`](validation/) | SUMO metrics (shockwave, safety), prediction-model studies, parity tests |
| [`calibration/`](calibration/) | Fitting the model to the straight road |
| [`sumo/`](sumo/) | Both SUMO scenarios and the scripts that build them |
| `experiments/` | Only the two numpy reference models that calibration and the parity tests import (the 87-model bake-off that chose Bounded-METANET is not shared) |
| `data/` | Fixed SUMO baselines, calibrated parameters, and the raw per-run result rows behind every table and figure |

`data/` contents:

| folder / file | holds | report |
|---|---|---|
| `sweep_final/` | round 42: every cost function on both roads, realistic sign, final config, 10 seeds (1,190 runs) | §7.1, §7.2, App. A |
| `sweep_final_ssm/` | round 42b: safety (SSM-instrumented) runs of the headline objectives | §7.3 |
| `sweep_bmjam/` | round 41: `tts:1\|jam:5` on the jam-calibrated model, idealised / realistic sign, drivers 1.5×, solver test | §5.5, §7.3, §7.4 |
| `sweep_straight/` | rounds 11–37 on the straight road, incl. instrumented no-control and rule baselines and rule tuning | §6.3, §7.3, §7.4 |
| `prediction_study/` | 600 s forecast skill and sign-response branches on both roads | §4.4, §7.5 |
| `sumo_baseline.npz`, `straight_baseline.npz` | fixed uncontrolled SUMO runs the models are calibrated against — **never regenerate** | §4.3 |
| `params_*.json` | straight-road calibration outputs | Table 1 |
| `wave_validation.json` | shockwave metric checked against vehicle trajectories | §3.3 |
| `sign_check.jsonl` | does each headline objective's cost respond to the sign? | — |

Each `sweep_*/` folder has `jobs.json` (the manifest) and `shard<k>.json`
(one list of result rows per worker; one row = one SUMO run).

---

## Reproducing the report

These need only the shared data (no SUMO, seconds each). Figures go to
`docs/final_report/figures/`, which is created on demand and not tracked.

| report item | command | reads |
|---|---|---|
| Fig. 1 | `python control/plot_report_model.py` ¹ | `sumo_baseline.npz` |
| Fig. 2 | `python control/plot_report_model.py` ¹ (runs one uncontrolled SUMO run) | — |
| Table 1 | `PARAMS_BM`, `PARAMS_BM_JAM_DIS` in `model/metanet_model.py` | — |
| Table 2 | `python validation/analyse_prediction_study.py` | `prediction_study/` |
| Figs. 3–6, Table 6, Table 7 | `python control/plot_report_results.py`, `python control/analyse_final.py` | `sweep_final/` |
| Tables 10–13 (App. A) | `python control/make_appendix_tables.py` | `sweep_final/` |
| Table 8 | `python control/analyse_final_ssm.py` | `sweep_final_ssm/`, `sweep_straight/`, `sweep_bmjam/` |
| Table 9 | `python control/analyse_robustness.py` | `sweep_bmjam/`, `sweep_final/`, `sweep_straight/` |
| Fig. 7, §7.5 numbers | `python control/plot_sign_response.py`, `python validation/analyse_prediction_study.py` | `prediction_study/` |
| §5.5 solver test | `python control/analyse_overnight.py` (round 41 section) | `sweep_bmjam/` |

¹ One script draws both Section 4 figures.

Checked from a fresh export of the repository: Appendix A, `summary.json` and
Figs. 1, 3–7 regenerate identical to the report's copies.

Re-running the experiments themselves needs SUMO and takes about a day on a
6-core laptop for round 42:

```powershell
powershell -File control\launch_sweep.ps1 -Round final -Dir data\sweep_final -N 6
python control\sweep.py --status --dir data\sweep_final
```

Workers skip jobs whose rows already exist, so relaunching the same command
resumes. The prediction study is `control\run_prediction_study.ps1`.

---

## The final configuration

Defined in `control/sweep.py` (`build_final_manifest`) and run by
`control/phase_d.py::run_closed_loop`:

- replan every 60 s, horizon Np = 7 moves (420 s), limits rounded to 10 km/h
  ("realistic sign"); limits in [40 km/h, legal]; rate limit 60 km/h/move
- IPOPT, max 300 iterations, no wall-clock limit (deterministic), warm start;
  on failure the previous limits are held
- model: `PARAMS_BM` on the merge, `PARAMS_BM_JAM_DIS` on the straight road
- each cost term enters at its road's **matched weight** (`FINAL_W_MERGE`,
  `FINAL_W_STRAIGHT`; report Table 10), set by `validation/term_scale.py`
- baselines: no control, and the reactive rule (v_det 30 km/h, post 40 km/h on
  the 9 signs, hold 120 s)
- 10 seeds, paired on seed, Wilcoxon signed-rank

Cost functions are strings of `term:weight` joined by `|`, e.g.
`tts:1|jam:2.5`. Report names → code names:

| report | code | report | code |
|---|---|---|---|
| Total time spent | `tts` | Speed variance | `speed_var` |
| Delay | `delay` | Speed drop | `speed_grad` |
| Terminal | `terminal` | Conflict risk | `conflict_risk` |
| Entrance queue | `queue` | Control variation | `dvsl` |
| Jam | `jam` | Upstream congestion | `upstream_queue` |
| Acceleration | `accel` | Squared density | `density_sq` |
| Temporal acceleration | `accel_temporal` | Density tracking | `density_track` |
| Density-weighted accel. | `accel_dens` | Bottleneck density | `bneck_density` |
| Density rate | `dens_rate` | Bottleneck discharge | `bneck_flow` |

To add a term, write `term_<name>(S)` in `control/cost_functions.py` (smooth
CasADi only: no `max`, `abs` or hard indicators) and register it in `_TERMS`.
To run a new experiment, add a `build_<name>_manifest` to `control/sweep.py`
and register it in `--round`. Use a new `--dir`, and keep job fields
key-stable so stored rows still resume.

---

## Things the next person must know

The full list, with evidence, is in [`EXPERIMENTS.md`](EXPERIMENTS.md) §3.

- **Measure whole-road congestion, including the entrance queue.** Many
  objectives "reduce shockwaves" by holding traffic back off the measured road.
- **Report the sign behaviour with every number.** The idealised sign
  (30 s moves, continuous limits) flatters the MPC; the rule is unaffected.
- **Match term weights before comparing objectives.** At weight 1, terms
  differ in influence by up to 1000×.
- **MPC rows are not bit-reproducible across machines.** No-control and rule
  rows are. Pair MPC rows only with rows from the same machine.
- **Never use fewer than 6 seeds** for a ranking; always pair on seed.
- **Parity-test any change to the model** (`validation/test_*parity*.py`, ~1e-12).
