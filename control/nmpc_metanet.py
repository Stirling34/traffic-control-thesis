# -*- coding: utf-8 -*-
"""
Nonlinear MPC for VSL control, using the METANET model (metanet_model.py) as
the prediction model and CasADi + IPOPT as the solver.

This replaces the linearise-then-QP controller in lwr_v1.solve_mpc: the exact
nonlinear METANET dynamics F(x, u, d) enter the NLP directly (multiple
shooting), so there is no finite-difference linearisation and no OSQP
convergence fragility.

The NLP is built ONCE (structure fixed); each control step only changes the
parameters (current state, demand forecast, previous VSL) — csnlp/IPOPT then
re-solves with warm start.

Cost function (same simple form as the LWR MPC, translated to METANET units):
    J = sum_k || rho_k / rho_max - rho_target ||^2
      + alpha * sum_moves || Delta v_ctrl / v_ub ||^2
with a hard rate limit |Delta v_ctrl| <= dv_max per control move (incl. the
first move vs the last applied VSL).

Note on the target: the LWR controller used rho_target = 0.6 of jam, i.e.
1.2x Greenshields critical density (which is 0.5 of jam). METANET's critical
density is rho_crit/rho_max ~ 0.22 of jam, so the faithful translation keeps
the "1.2x critical" intent rather than the raw 0.6 number — 0.6 of jam is deep
congestion in METANET and tracking it would *reward* congestion.
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc

from model.metanet_model import MetanetModel
from control.cost_functions import COSTS, build_cost


class NmpcVslController:
    """Nonlinear MPC over the METANET model. Solves for the VSL (km/h) of each
    controlled (2-lane) segment; the first move of the plan is applied to SUMO
    and held for one control interval."""

    def __init__(
        self,
        model: MetanetModel,
        Np: int = 7,           # prediction horizon, in control intervals
        Nc: int = 5,           # control horizon, in control intervals
        M: int = 6,            # model steps per control interval (6*5 s = 30 s)
        alpha: float = 0.1,    # weight on VSL smoothness (as in the LWR MPC)
        rho_target_frac: float = None,  # target density as fraction of rho_max;
                                        # default 1.2 * rho_crit / rho_max
        v_min_kmh: float = 20.0,        # lowest VSL the signs may display
        v_max_kmh: float = None,        # cap the UPPER bound below the legal
                                        # limit, so the entire feasible set lies
                                        # where the VSL actually bites
        dv_max_kmh: float = 7.2,        # rate limit per 30 s move (= 2 m/s, as LWR)
        ipopt_max_iter: int = 300,
        cost: str = "density_track",   # see control/cost_functions.py
    ) -> None:
        self.model = model
        p = model.p
        self.M = M
        self.Np = Np
        n_seg, n_ctrl = model.n_seg, model.n_ctrl

        if rho_target_frac is None:
            rho_target_frac = 1.2 * p.rho_crit / p.rho_max
        self.rho_target_frac = rho_target_frac

        mpc = Mpc[cs.SX](
            nlp=Nlp[cs.SX](sym_type="SX"),
            prediction_horizon=Np * M,
            control_horizon=Nc * M,
            input_spacing=M,
        )

        # variables: states are only lower-bounded (METANET can go slightly
        # negative numerically); the dynamics themselves keep them sane.
        rho, _ = mpc.state("rho", n_seg, lb=0)
        v, _ = mpc.state("v", n_seg, lb=0)
        w, _ = mpc.state("w", 1, lb=0)
        ub_vec = model.v_ctrl_ub.reshape(-1, 1).copy()
        if v_max_kmh is not None:
            ub_vec = np.minimum(ub_vec, float(v_max_kmh))
        self.v_ctrl_ub_used = ub_vec.ravel()
        v_ctrl, _ = mpc.action("v_ctrl", n_ctrl, lb=v_min_kmh, ub=ub_vec)
        d = mpc.disturbance("d", 1)

        # exact nonlinear METANET dynamics as equality constraints
        mpc.set_nonlinear_dynamics(lambda x, u, d_: model.F(x, u, d_)[0])

        # previous applied VSL (parameter) for the first-move Delta terms
        v_ctrl_last = mpc.parameter("v_ctrl_last", (n_ctrl, 1))
        dv = cs.diff(cs.horzcat(v_ctrl_last, v_ctrl), 1, 1)

        # hard rate limit per move (same 2 m/s per control decision as LWR MPC)
        mpc.constraint("rate_up", dv, "<=", dv_max_kmh)
        mpc.constraint("rate_down", dv, ">=", -dv_max_kmh)

        # cost: assembled from control/cost_functions.py so Phase D can swap
        # objectives without touching the rest of the controller
        v_scale = model.v_ctrl_ub.reshape(-1, 1)
        S = {
            "rho": rho, "v": v, "w": w, "v_ctrl": v_ctrl, "dv": dv,
            "Lkm": np.array(model.seg_lengths_km).reshape(-1, 1),
            "lam": np.array(model.seg_lanes, float).reshape(-1, 1),
            "T_h": p.T, "T_s": p.T_sec, "p": p,
            "v_free": p.v_free_main, "v_scale": v_scale,
            "rho_target_frac": rho_target_frac,
            # normaliser so a weight of 1.0 means comparable influence across
            # costs: roughly the uncontrolled TTS over the horizon, in veh-h
            "tts_scale": 100.0,
            # constant so the acceleration term stays a sum, not a ratio
            "accel_scale": 2000.0,
        }
        self.cost_name = cost
        mpc.minimize(build_cost(cost, S))

        opts = {
            "expand": True,
            "print_time": False,
            "ipopt": {
                # About 3% of (state, v_ctrl_last) combinations stall: IPOPT
                # cycles at constant objective for hundreds of iterations and
                # then reports Maximum_Iterations_Exceeded. Measured worst case
                # 77 s, against a 30 s control interval. A run that has not
                # converged by ~300 iterations here never does, so cap it and
                # cap the clock. This does not make those instances solve, it
                # bounds what they cost. See docs/research/
                # 2026-08-18-solver-robustness.md.
                "max_iter": ipopt_max_iter,
                "max_wall_time": 8.0,
                # Return a good-enough answer rather than nothing. Some stalls
                # are merely slow rather than infeasible, and a solution at
                # 1e-4 is far more useful to the controller than holding the
                # previous limits. Treated as success in solve() below.
                "acceptable_tol": 1e-4,
                "acceptable_iter": 5,
                "acceptable_constr_viol_tol": 1e-4,
                "sb": "yes",
                "print_level": 0,
                # rho^a is NaN for rho < 0; forbid IPOPT's default slight
                # bound relaxation so states can never leave [lb, ub]
                "bound_relax_factor": 0.0,
            },
        }
        mpc.init_solver(solver="ipopt", opts=opts)

        self.mpc = mpc
        self._sol_prev = None
        self.stats = []  # (success, status, solve_time_s, cost) per solve

    def solve(self, rho0, v0, w0, demand_forecast_vph, v_ctrl_last):
        """One MPC solve.

        rho0, v0 : current METANET state per segment (veh/km/lane, km/h)
        w0       : current origin queue (veh)
        demand_forecast_vph : array of length Np*M (veh/h per model step)
        v_ctrl_last : VSL currently applied (km/h, per controlled segment)

        Returns (v_ctrl_first_move, info dict). On solver failure the previous
        VSL is returned unchanged and info["success"] is False.
        """
        d_hat = np.asarray(demand_forecast_vph, dtype=float).reshape(1, -1)
        assert d_hat.shape[1] == self.Np * self.M, (
            f"demand forecast must have {self.Np * self.M} entries, "
            f"got {d_hat.shape[1]}"
        )
        v_ctrl_last = np.asarray(v_ctrl_last, dtype=float).reshape(-1, 1)

        # The mainstream-origin flow term (-a*log(v/v_free))^(1/a) has an
        # infinite gradient at v == v_free, which NaNs the constraint Jacobian
        # if the measured speed sits exactly at free flow (it does at t=0).
        # Feed the solver speeds fractionally below free flow instead.
        p = self.model.p
        v_cap = np.array(
            [p.v_free_main] * self.model.n_ctrl
            + [p.v_free_bneck] * (self.model.n_seg - self.model.n_ctrl)
        )
        v0 = np.minimum(np.asarray(v0, dtype=float), 0.995 * v_cap)
        rho0 = np.asarray(rho0, dtype=float)
        w0 = np.atleast_1d(np.asarray(w0, dtype=float))

        # Without a previous solution to warm-start from, an all-zeros guess
        # makes IPOPT grind (dynamically infeasible, far from optimum). Seed
        # with a feasible rollout of the model under the current VSL instead.
        vals0 = self._sol_prev
        if vals0 is None:
            vals0 = self._rollout_guess(rho0, v0, w0, v_ctrl_last, d_hat)

        t0 = time.time()
        sol = self.mpc.solve(
            pars={
                "rho_0": rho0,
                "v_0": v0,
                "w_0": w0,
                "d": d_hat,
                "v_ctrl_last": v_ctrl_last,
            },
            vals0=vals0,
        )
        dt = time.time() - t0

        status = str(getattr(sol, "status", "?"))
        # IPOPT's "acceptable" exit is a converged-enough solution, not a
        # failure. csnlp reports success=False for it, so accept it here
        # explicitly rather than discarding a usable answer and holding
        # the previous limits.
        success = (bool(getattr(sol, "success", False))
                   or status == "Solved_To_Acceptable_Level")
        cost = float(sol.f) if success else np.nan
        self.stats.append((success, status, dt, cost))

        if not success:
            # Drop the warm start. Otherwise the next solve is seeded from a
            # solution that is now several control steps stale, which is what
            # turns one failure into a run of them: measured 39 failures in 81
            # solves in closed loop, against ~3% on isolated states. Clearing it
            # makes the next call rebuild a feasible rollout from the CURRENT
            # measured state.
            self._sol_prev = None
            return v_ctrl_last.ravel(), {
                "success": False, "status": status, "solve_time": dt,
            }

        self._sol_prev = sol.vals
        plan = np.asarray(sol.vals["v_ctrl"])  # (n_ctrl, n_moves)
        return np.asarray(plan[:, 0]).ravel(), {
            "success": True, "status": status, "solve_time": dt,
            "cost": cost, "plan": plan,
        }

    def _rollout_guess(self, rho0, v0, w0, v_ctrl_last, d_hat):
        """Dynamically feasible initial guess: hold the current VSL and roll
        the model forward over the prediction horizon."""
        N = self.Np * self.M
        u = np.asarray(v_ctrl_last, dtype=float).ravel()
        x = self.model.pack_x(rho0, v0, w0)
        rho_g, v_g, w_g = [rho0], [v0], [w0]
        for k in range(N):
            x = np.asarray(self.model.F(x, u, float(d_hat[0, k]))[0]).ravel()
            r_k, v_k, w_k = self.model.unpack_x(x)
            rho_g.append(r_k)
            v_g.append(v_k)
            w_g.append(w_k)
        n_moves = self.mpc.actions["v_ctrl"].shape[1]
        return {
            "rho": np.array(rho_g).T,
            "v": np.array(v_g).T,
            "w": np.array(w_g).T,
            "v_ctrl": np.tile(u.reshape(-1, 1), (1, n_moves)),
        }


if __name__ == "__main__":
    # Smoke test in pure simulation (no SUMO): closed loop METANET <-> NMPC.
    from model.metanet_model import MetanetParams, load_geometry

    geom = load_geometry()
    model = MetanetModel(geom, MetanetParams())
    ctrl = NmpcVslController(model)
    p = model.p
    M = ctrl.M

    sim_steps = int(1000 / p.T_sec)  # 1000 s
    demand = np.full(sim_steps + ctrl.Np * M, 4000.0)
    x = model.x0_empty()
    u = model.v_ctrl_ub.copy()

    for k in range(sim_steps):
        if k % M == 0:
            rho0, v0, w0 = model.unpack_x(x)
            # VSL inactive until every segment carries traffic — while the road
            # is (partly) empty the density target is unreachable and the NLP
            # just burns iterations
            if rho0.min() > 1.0:
                u, info = ctrl.solve(rho0, v0, w0, demand[k : k + ctrl.Np * M], u)
                if k % (M * 5) == 0:
                    print(f"t={k*p.T_sec:5.0f}s  ok={info['success']} "
                          f"{info['solve_time']:.2f}s  u={np.round(u, 0)}")
        x = np.asarray(model.F(x, u, demand[k])[0]).ravel()

    rho_f, v_f, w_f = model.unpack_x(x)
    n_fail = sum(1 for s in ctrl.stats if not s[0])
    t_avg = np.mean([s[2] for s in ctrl.stats])
    print(f"\nSolves: {len(ctrl.stats)}, failures: {n_fail}, "
          f"avg solve time {t_avg:.2f}s")
    print("final rho:", np.round(rho_f, 1))
