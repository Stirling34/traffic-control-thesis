# control/ — the MPC controller and the SUMO loop

| file | what it does |
|---|---|
| `nmpc_metanet.py` | The nonlinear MPC. Builds an NLP over the METANET dynamics with `csnlp`, solves it with IPOPT. |
| `main_metanet_v1.py` | The closed loop: SUMO and the controller talking over TraCI. |

## How the loop works

Every 30 seconds:

1. Read density, speed and the insertion queue from SUMO over TraCI.
2. Load those into the model as its current state.
3. Solve for the speed limits that minimise the cost over the next 210 s.
4. Apply the first move to the SUMO edges with `setMaxSpeed`.
5. Step SUMO forward and repeat.

```bash
python control/main_metanet_v1.py        # headless
```

Set `USE_GUI = True` to watch it. `CONTROL_SCHEME = 0` gives the no-control
baseline.

Solve time is 0.1 to 0.9 s against a 30 s budget, so there is roughly 30x
headroom. This is comfortably real-time capable already.

## The cost function

Currently density tracking plus a penalty on changing the limits:

```
J = sum || rho/rho_max - rho_target ||^2  +  alpha * sum || d v_ctrl / v_ub ||^2
```

**This is the thing the project exists to replace.** Phase D compares
alternatives. Write a new one as a CasADi expression of the model states and
swap it in here. The decision variables are the speed limits `v_ctrl`; the
states available are `rho`, `v` and the queue `w`.

## Known defects, not yet fixed

- **The demand forecast is wrong.** It assumes 4000 veh/h until t = 1000 then
  zero. SUMO's realised inflow collapses to about 200 veh/h around t = 850 and
  then climbs back to about 3600 as its backlog flushes. MPC performance depends
  heavily on forecast quality, so this matters.
- **The model is the original METANET**, which predicts poorly at 210 s. Two
  better models exist in `experiments/`; neither is wired in here yet.

## Numerical traps already fixed, leave these alone

- `bound_relax_factor: 0`. IPOPT otherwise lets density sit slightly negative,
  and `rho^1.867` is then NaN.
- `SmoothMainstreamOrigin`. The origin flow term has an infinite gradient at
  exactly free-flow speed, which NaNs the constraint Jacobian.
- The controller sleeps until every segment carries traffic. On an empty road
  the density target is unreachable and the solver grinds to max iterations.
