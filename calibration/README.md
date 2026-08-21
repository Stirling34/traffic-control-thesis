# calibration/ — fitting the model to SUMO

| file | what it does |
|---|---|
| `calibrate.py` | Fits all seven METANET parameters by differential evolution. |
| `metanet_param_sweep.py` | Sweeps the fundamental-diagram shape, relaxation and anticipation. |
| `prediction_skill_sweep.py` | Sweeps for prediction skill rather than open-loop fit. |
| `fd_factorial.py` | Fundamental-diagram family crossed with the capacity drop. |
| `capacity_drop_eval.py` | Does the capacity drop help? Prediction, control authority, fidelity. |
| `test_capacity_drop.py` | Acceptance tests: regression, hysteresis, jam wave, drop magnitude. |

```bash
python calibration/calibrate.py --iters 250     # a few minutes on 8 cores
python calibration/test_capacity_drop.py        # run after touching the model
```

## Why this folder exists

Until August 2026 exactly **one** parameter had ever been fitted: `rho_crit`,
solved from the measured discharge, which is a single scalar. The
fundamental-diagram exponent, the relaxation time and the anticipation
regularisation were textbook defaults, and the anticipation coefficient was
hand-tuned until a numerical artefact went away. Published work fits all of them
jointly against the full trajectory.

Doing it properly is worth roughly 14% on speed error, and it generalises to
held-out data.

## The rules, learned the hard way

**Fit on t < 1250 s only, and score on t >= 1250 s.** Otherwise you are
measuring how well you memorised one trajectory.

**Physical validity is a constraint, not a preference.** Optimising fit alone
once selected a model with peak density 387 veh/km/lane, against a physical jam
density of 182, that discharged 823 veh/h and never cleared. Every objective
here penalises: peak density above 185, discharge outside 10% of the measured
2325 veh/h, and a road that never empties.

A subtler version of the same trap appeared later. The benchmark originally
bounded density only from *above*, so models that never formed a queue at all
passed as valid and scored well simply by staying near free flow. Bound it from
below as well.

**Anchor `v_free` to the observed free-flow speed**, about 84 to 86 km/h. Not to
the posted limits, and not free. Left unconstrained the optimiser drives it down
to about 74 km/h, because the error is dominated by the long congested phase.
The consequence is not cosmetic: a speed limit above `1.1 * v_free` then becomes
exactly inert, so the controller loses its whole upper range.

**Floor the relaxation time `tau`.** Left free it collapses to about 4.6 s
against a 5 s step, which is numerically marginal and guts the second-order
claim. Pinning it at the textbook 18 s improves the held-out score anyway.

**Fit for the job the model does.** Adding prediction skill to the objective,
computed only inside the calibration window, took one model from 0.701 to 0.784
held-out. The literature supports this (Airaldi, De Schutter and Dabiri 2025);
calibrating for fidelity and calibrating for control performance give different
answers, and only the second is what the project is judged on.
