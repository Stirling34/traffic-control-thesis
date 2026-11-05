# Experiment log

What was run, in order, and what each stage found. Results quoted in the
final report come from the stages marked **[report]**; everything else is
the path that led there. Rows from stages whose data is not in `data/` stayed
on the author's machine.

All runs: SUMO is the ground truth; results are paired on seed against no
control (and later the reactive rule), Wilcoxon signed-rank; "jam" means
vehicles below 30 km/h.

---

## 1. Stages

### Jul–Aug 2026 — prediction model (merge road)

- Replaced the Assessment-2 LWR + linear MPC (OSQP) with METANET
  (`sym-metanet`) + nonlinear MPC (CasADi/IPOPT).
- Standard METANET predicted badly at the controller's horizon (correlation
  0.05 against SUMO at 210 s, held out). The anticipation term either let
  density run away or blurred the queue (report Fig. 1).
- **Model bake-off** (`experiments/`, not shared): 87 candidates on one locked
  benchmark. **Bounded-METANET** (Zhao et al. 2025) was chosen as published
  and citable; it is ported into `model/metanet_model.py` as `PARAMS_BM`
  (parity 6e-13 against the numpy original).
- **Actuator fix**: SUMO drivers drive ~0.89× the sign, so the original speed
  cap left the sign inert on ~99% of steps. `vsl_scaled=True` caps the target
  speed at `(u / u_legal)·v_free` (report eq. 4).

### Aug 2026 — first cost-function comparison (short merge road)

- 22 objectives (`control/phase_d.py`). Apparent smoothing correlated with
  **entrance-queue growth** (r = −0.85): objectives were metering vehicles off
  the measured road.
- Fix: a **3.6 km uncontrolled approach** added in front of the signs
  (`sumo/build_long_net.py`). Round 1 (1,760 runs) confirmed that the queue
  then stays visible, and that smoothing is paid for in upstream delay.
- The Zegeye acceleration metric must be aggregated at 8 s (CFL); values
  stored before 2026-08-21 are about 2.85× inflated.

### Sep 1–8 — shockwave programme (merge road, rounds 2–10, 3,160 runs)

- `accel_rms` is ~90% spatial and scales with speed, so slowing traffic
  scores as "smoothing". It was replaced by `validation/wave_metrics.py`
  (congestion-front reach, congested space-time), which was validated against
  vehicle trajectories (`validate_wave_trajectories.py`).
- **0 of 39 policies shortened the queue front.** Three purpose-built wave
  terms (`accel_temporal`, `upstream_queue`, `dens_rate`) also failed. Safety
  tracked the front (r(TIT, reach) = +0.66).
- Cause: excess demand at a permanent bottleneck must be stored somewhere; the
  model also has no mechanism by which a sign raises throughput.

### Sep 11–19 — straight road (rounds 11–17, ~1,900 runs, `data/sweep_straight`)

- New scenario (`sumo/build_straight_net.py`): two lanes throughout, near
  capacity, one slow vehicle at 15% of the limit for 240 s.
- The first sweep was confounded: the activation gate switched the controller
  off during the jam. Fixed, then rerun.
- On the idealised sign MPC removed 44–60% of jam vehicle-hours across
  loading, severity, duration and horizon. Removing the `dvsl` restraint term
  mattered more than the choice of objective.
- A model recalibrated on this road (`PARAMS_BM_STRAIGHT`) **predicted better
  and controlled worse**.

### Sep 19–22 — realism and the rule (rounds 18–37, `data/sweep_straight`)

- Safety instrumentation, the reactive rule (Dutch AID / SPECIALIST style),
  repeated disturbances, disturbance location, and a deployable sign (60 s
  moves, 10 km/h steps).
- **With a realistic sign the MPC lost most of its effect; the rule lost
  none.** The rule was tuned (v_det 30 km/h) on seeds 1–10 and confirmed on
  unseen seeds 11–30 (−81% jam at 4600 veh/h) **[report §6.3]**.
- When drivers do 1.5× the posted limit the rule holds; the MPC degrades.
- MPC rows are not reproducible under a wall-clock solver budget, so all later
  rounds use a deterministic budget (300 iterations).

### Sep 29–Oct 6 — why the MPC behaves as it does (rounds 38–41)

- Under `PARAMS_BM` the `tts` objective is **flat in the sign**. IPOPT's
  barrier centring, not the forecast, set the posted limit (shown by moving a
  bound the sign never reaches; `validation/flat_objective_check.py`).
- **Jam-clearance calibration** (`calibration/calibrate_jam.py`):
  `PARAMS_BM_JAM_DIS` discharges a jam at 4687 vs SUMO's ~4700 veh/h
  **[report Table 1, §4.3]**.
- A **`jam` cost term** was added and the 7.2 km/h rate limit lifted. Round 41
  (`data/sweep_bmjam`): `tts:1|jam:5` cuts the jam on a realistic sign but
  still trails the rule **[report §7.4, Table 9]**.
- A stronger solver halved failures without changing the jam result
  **[report §5.5]**.

### Oct 9–12 — final comparison **[report]**

- **Matched weights** (`validation/term_scale.py`): each term scaled to move
  the sign as much as `tts` does (report Table 10).
- **Round 42** (`data/sweep_final`, 1,190 runs, 0 errors): 39 cost functions
  on the merge, 37 on the straight road at 4600 and 4800 veh/h, final
  configuration, realistic sign, 10 seeds **[report §7.1–7.2, App. A]**.
- **Prediction study** (`data/prediction_study`): forecast skill against
  persistence, and the model's predicted vs SUMO's actual response to fixed
  sign plans **[report Table 2, Fig. 7]**.
- **Round 42b** (`data/sweep_final_ssm`): safety of the headline objectives;
  `tts:1|dvsl:0.1` was added on 2026-10-20 **[report Table 8]**.
- **Sign check** (`data/sign_check.jsonl`): every headline objective's cost
  responds to the sign, but IPOPT's start point picks between local optima, so
  close objectives should be quoted as a group, not ranked.

---

## 2. Headline results (detail in the report)

- **Merge**: no cost function reduced congestion; 14 of 39 increased it.
  TTS + jam (×4) and TTS + terminal reduced the sharpest speed drop at no
  time cost.
- **Straight**: TTS paired with a jam, conflict-risk or speed-variance term
  cut congestion 58–71% and hard braking 65–72%, for 14–61 s/veh. Acceleration,
  density-rate and squared-density objectives raised whole-road congestion.
- **The reactive rule matched or beat every MPC objective** (−68/−70%
  congestion for +7/+8 s/veh). The main limit is that the model mispredicts
  the effect of its own sign (right direction 72% of the time, about a quarter
  of the size).

---

## 3. Dead ends and pitfalls (do not repeat)

- One sym-metanet `Link` with per-segment vector parameters: the library
  assumes scalar link parameters. Use one link per homogeneous stretch.
- Textbook η = 60 on 200 m segments: anticipation is too strong and produces
  stop-go stripes. Use η ≈ 30, or Bounded-METANET.
- Stock `MainstreamOrigin` at v = v_free: infinite gradient, NaN Jacobian.
  Use `SmoothMainstreamOrigin`.
- IPOPT's default `bound_relax_factor`: ρ goes slightly negative and ρ^a
  becomes NaN. Set it to 0.
- Solving on an empty road, or from an all-zero initial guess, fails. Gate
  activation on density and cold-start from a hold-limits rollout.
- Fitting the model's queue density to SUMO's (~180 veh/km/lane) corrupts the
  free-flow branch. Accept ρ_max ≈ 100–120 and document it.
- Scoring smoothing without the entrance queue: metering looks like smoothing.
- `accel_rms` as a shockwave measure rewards uniformly slow traffic.
- Ranking on congestion-front **growth** speed: not reliable against
  trajectories. Reach and recovery are.
- Planting a jam by fully closing the road: the controller's NLP fails on
  empty cells. Use one slow vehicle.
- Comparing objectives at weight 1: weights differ in effect by 30–1000×.
- Expecting plain `tts` to cut the jam on the straight road: restricting
  costs time there, so a correct model makes `tts` refuse to restrict.
- Wall-clock solver budgets make MPC rows machine-dependent. Use the
  deterministic iteration budget.
- Mixing SSM-instrumented and plain rows: they are not bit-identical.
- Conclusions from fewer than 6 seeds: 3-seed rankings reversed at 10.
- Running two processes that write the same results file: one clobbered a
  finished run. One writer per shard (`control/launch_sweep.ps1` enforces it).
- More than 6 MPC workers on a 16 GB laptop: the page file grows until the
  disk fills.
