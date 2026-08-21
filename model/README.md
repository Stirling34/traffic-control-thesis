# model/ — the traffic models

The macroscopic model that runs inside the MPC. It has to predict how traffic
will evolve over the next few minutes, cheaply enough to be solved hundreds of
times per control step.

| file | what it does |
|---|---|
| `metanet_model.py` | METANET built with the `sym-metanet` library, plus two optional extensions. Compiles the dynamics to a CasADi function `F(x, u, d)`. |
| `ExtraData.py` | Parses the SUMO network file to get lengths, lanes and speed limits. |

## What the model is

Nine 200 m two-lane segments, then two segments making up the 496 m one-lane
bottleneck. State is density `rho` and speed `v` per segment plus the origin
queue `w`. Input `u` is the speed limit per segment. Disturbance `d` is inflow.

```python
from model.metanet_model import MetanetModel, MetanetParams, load_geometry

model = MetanetModel(load_geometry(), MetanetParams())
x_next, q = model.F(x, u, d)          # one 5 s step
res = model.simulate(demand_vph)      # whole episode
```

## The two optional extensions, both off by default

**Capacity drop** (`alpha_cd`). Real bottlenecks discharge more slowly *after*
breaking down than before, and do not recover until the queue clears. Plain
METANET has no such memory. This scales the equilibrium speed by a factor
driven by the density of the segment **upstream**, so the drop follows a moving
jam rather than being pinned to one place.

```
Veq_i = min( V_fd(rho_i) * (1 - alpha_cd * z_{i-1}) , (1 + alpha) * v_ctrl_i )
```

Set `alpha_cd=0.25, rho_cd_ref=80.0` to enable. At 0 it reproduces the plain
model exactly; that regression test is in `calibration/test_capacity_drop.py`.

**Triangular fundamental diagram** (`fd_type="triangular"`). Decouples jam
density from capacity. Improves congested-regime accuracy, hurts free flow.

## Things that will bite you

- **`v_free` is a property of the traffic, not the sign.** SUMO's drivers do
  about 84 to 86 km/h on an empty road, against posted limits of 84.6 to 99.
  Fit it to the observation. If an optimiser drives it down to chase error, a
  speed limit above `1.1 * v_free` becomes **exactly inert** and control
  authority is silently lost.
- **The anticipation term is a diffusion term.** It smooths density differences
  flat, and it is the main reason plain METANET's predictions collapse toward a
  uniform speed. It is also acting as the numerical stabiliser, so it cannot
  simply be turned down. See `experiments/expA_asym_antic/`.
- **The mainstream origin saturates.** Demand above about 4352 veh/h produces a
  bit-identical road state; the surplus only inflates the queue.
  `check_demand_sensitivity()` warns when this is happening.
