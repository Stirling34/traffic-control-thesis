# validation/ — does the model match SUMO?

| file | question it answers |
|---|---|
| `metanet_vs_sumo.py` | Does the model reproduce the whole episode? Produces the main comparison figures. |
| `metanet_prediction_error.py` | How accurate is it at the horizon the controller actually uses? |
| `metric_audit.py` | Is the error metric itself fair? Checks five ways it could mislead. |
| `prediction_visuals.py` | Scatter and profile plots showing what the prediction error looks like. |
| `open_loop_diagnosis.py` | Where in the run does the model diverge, and why does the heat map hide it? |
| `control_authority.py` | Does changing the speed limits change anything, in SUMO and in the model? |

```bash
python validation/metanet_vs_sumo.py               # uses the cached baseline
python validation/metanet_vs_sumo.py --rerun-sumo  # re-runs SUMO
python validation/metanet_prediction_error.py
```

## The two tests, and why both are needed

**Whole-episode (open loop).** Run the model for 2500 s from an empty road with
only the inflow, and compare. This is the protocol every published METANET
validation uses, so it is the number a reviewer will compare against the
literature.

**Prediction.** Take SUMO's measured state at 223 different times, roll the
model forward 210 s, compare. This is what the controller actually depends on.

**They disagree, and that matters.** One variant tested scored the best
open-loop fit of anything and the worst prediction, which is the signature of
fitting a trajectory rather than dynamics. **Do not select a model on
open-loop RMSE.**

## How the error is measured

Prediction is benchmarked against **persistence**, meaning assume nothing
changes. A model that cannot beat that is contributing nothing.

Only cells carrying traffic are scored. On empty road SUMO reports -1 m/s and
the substituted limit differs from the model's free-flow speed, which is
bookkeeping rather than error.

Correlation is the clearest single measure: it asks whether the model's picture
of the road looks like SUMO's at all. `metric_audit.py` confirmed the metric is
fair. In particular the error is **not** a timing offset; shifting the
comparison by up to 180 s in either direction does not improve the fit.

One subtlety worth knowing: pooled correlation mixes two things. Removing the
static per-segment offset first (anomaly correlation) shows the original model
has broadly the right dynamics in the wrong *place*. Pooled is still the right
metric for MPC, since a controller needs to know where the congestion is.

## Reading the figures

`metanet_vs_sumo_baseline.png` uses a **per-panel** density colour scale,
because SUMO's queue reaches 185 veh/km/lane while METANET's congested branch
carries the same flow at about 77. Forcing one shared scale made the model look
empty. Per-panel scaling compares the *shape*; the speed row uses a shared scale
and is directly comparable.

The main figure is cropped to 1000 s for readability. **Do not draw conclusions
about draining from it** — SUMO does not clear until 1773 s. The full-range
version is written alongside it as `..._full.png`.
