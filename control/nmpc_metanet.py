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
        dv_space_kmh: float = None,     # hard limit on the DROP between adjacent
                                        # signs. None = off, which is the
                                        # behaviour of every result before
                                        # 2026-09-01. See the constraint below.
        ipopt_max_iter: int = 300,
        ipopt_max_wall: float = 8.0,
        cost: str = "density_track",   # see control/cost_functions.py
        driver_lag_s: float = None,     # tell the model how drivers adopt a
        driver_comp: float = None,      # posted limit; see the block below.
        sim_dt_s: float = 1.0,          # plant step the lag is applied at
        retry_cold: bool = False,       # on a failed WARM solve, retry once
                                        # from a fresh rollout guess
        on_fail: str = "hold",          # "hold" = keep the current sign (the
                                        # behaviour of every stored result);
                                        # "shift" = take the next move of the
                                        # last successful plan, if any
        ipopt_extra: dict = None,       # extra IPOPT options (e.g.
                                        # {"mu_strategy": "adaptive"})
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

        # DRIVER-AWARE PREDICTION (added 2026-09-29). By default the model
        # assumes the traffic adopts a posted limit instantly and exactly, which
        # is what SUMO did in every run before round 28. Rounds 28/35 made the
        # PLANT's drivers lag or under-comply (phase_d.run_closed_loop,
        # `actuation_lag_s` / `compliance`) without telling the controller, so
        # the MPC planned for drivers it did not have. These two options put the
        # plant's own driver model into the prediction, so the comparison is
        # between an MPC that knows how its drivers respond and one that does
        # not. Both reproduce the plant's code exactly, not its description:
        #
        #   driver_lag_s  phase_d closes 1/L of the gap to the posted value
        #       every simulation second, i.e. a FIRST-ORDER lag with time
        #       constant L (63 % adopted after L s), not a linear ramp. The
        #       enforced value becomes a model state `v_enf`; over one model
        #       step of n = T_sec/sim_dt plant seconds the traffic sees
        #       u + (z - u) r^j for j = 0..n-1 (r = 1 - 1/L), so the model is
        #       driven by their MEAN and the state advances by r^n.
        #   driver_comp   phase_d enforces min(comp * posted, legal). For
        #       comp > 1 every posted value at or above legal/comp gives the
        #       SAME enforced speed (the legal limit), a flat region with zero
        #       gradient in which IPOPT cannot move -- the 2026-08-19 plateau
        #       again. So the posted upper bound is lowered to legal/comp, where
        #       enforced = comp * posted holds exactly with no min(), and the
        #       previous posted value is clipped to the same bound in solve()
        #       (posting 84.6 and 56.4 at comp 1.5 are the same road).
        #
        # The two are exclusive: combined, a lagged value can exceed legal/comp
        # and the min() returns, which is not needed for any experiment run.
        if driver_lag_s is not None and driver_comp is not None:
            raise ValueError("driver_lag_s and driver_comp are not supported "
                             "together")
        self.driver_lag_s = None if driver_lag_s is None else float(driver_lag_s)
        self.driver_comp = None if driver_comp is None else float(driver_comp)
        if self.driver_comp is not None and self.driver_comp > 1.0:
            ub_vec = np.minimum(ub_vec, model.v_ctrl_ub.reshape(-1, 1)
                                / self.driver_comp)
        self.v_ctrl_ub_used = ub_vec.ravel()
        v_ctrl, _ = mpc.action("v_ctrl", n_ctrl, lb=v_min_kmh, ub=ub_vec)
        d = mpc.disturbance("d", 1)

        if self.driver_lag_s is not None:
            if self.driver_lag_s <= 1.0:
                raise ValueError("driver_lag_s must exceed one plant step")
            n_sub = int(round(p.T_sec / float(sim_dt_s)))
            if abs(n_sub * float(sim_dt_s) - p.T_sec) > 1e-9:
                raise ValueError("model step must be a multiple of sim_dt_s")
            r = 1.0 - 1.0 / self.driver_lag_s
            self._lag_c = float(np.mean(r ** np.arange(n_sub)))
            self._lag_rn = float(r ** n_sub)
            mpc.state("v_enf", n_ctrl, lb=0)

        # exact nonlinear METANET dynamics as equality constraints
        nx = 2 * n_seg + 1
        self._nx = nx
        if self.driver_lag_s is not None:
            c_lag, rn_lag = self._lag_c, self._lag_rn

            def _dyn_lag(x, u, d_):
                z = x[nx:, :]
                return cs.vertcat(model.F(x[:nx, :], u + (z - u) * c_lag, d_)[0],
                                  u + (z - u) * rn_lag)
            mpc.set_nonlinear_dynamics(_dyn_lag)
        elif self.driver_comp is not None:
            g_comp = self.driver_comp
            mpc.set_nonlinear_dynamics(
                lambda x, u, d_: model.F(x, g_comp * u, d_)[0])
        else:
            mpc.set_nonlinear_dynamics(lambda x, u, d_: model.F(x, u, d_)[0])

        # previous applied VSL (parameter) for the first-move Delta terms
        v_ctrl_last = mpc.parameter("v_ctrl_last", (n_ctrl, 1))
        dv = cs.diff(cs.horzcat(v_ctrl_last, v_ctrl), 1, 1)

        # hard rate limit per move (same 2 m/s per control decision as LWR MPC)
        mpc.constraint("rate_up", dv, "<=", dv_max_kmh)
        mpc.constraint("rate_down", dv, ">=", -dv_max_kmh)

        # SPATIAL rate limit between adjacent signs. Added 2026-09-01; the
        # controller had only the TEMPORAL limit above, so it was free to post
        # 20 km/h next to 85 km/h across a 200 m boundary.
        #
        # This is how the literature imposes safety, and it is a HARD CONSTRAINT
        # everywhere, never a cost term. Hegyi, De Schutter & Hellendoorn 2003
        # (TRR 1852) sec. 3.4 bound all three ways a driver can meet a different
        # limit -- the sign changes where they are; they cross into a new
        # segment; or both at once -- each at v_maxdiff = 10 km/h over 500 m
        # segments. van de Weg et al. 2019 do the same. Carlson et al. bound
        # Delta b <= 0.2 between consecutive gantries; the I-24 deployment and
        # Oregon OR 217 both use 10 mph.
        #
        # Hegyi (2004) sec. 4.2.4 also explains why the SOFT version fails, which
        # is what this project measured: "putting constraints on the traffic
        # state could result in infeasible problems... alternatively an extra
        # term can be included that penalizes the violation. A disadvantage of
        # this second approach is that the weight of this extra term also has to
        # be tuned, and that THE CONSTRAINTS MAY BE VIOLATED." The `speed_grad`
        # and `conflict_risk` families were flat across a 16-fold weight sweep.
        #
        # One-sided, as in both sources: only the DROP a driver decelerates into
        # is bounded. Speeding up into a higher downstream limit is not a
        # conflict, and bounding it symmetrically would stop the controller
        # releasing the sign behind a queue.
        #
        # Segments run upstream -> downstream, so the drop a driver meets moving
        # from segment i to i+1 is v_ctrl[i] - v_ctrl[i+1].
        # SOFT, with a heavy penalty, rather than hard. Hegyi (2004) sec. 4.2.4
        # warns that "putting constraints on the traffic state could result in
        # infeasible problems", and that is not hypothetical here: measured, a
        # HARD version returns Infeasible_Problem whenever the currently applied
        # sign already violates the bound, because the 7.2 km/h temporal rate
        # limit cannot repair a 60 km/h gap inside one move. In closed loop the
        # applied sign is normally the controller's own previous solution and so
        # already satisfies it -- but a solver failure holds the previous limits
        # and the activation gate gets switched on and off, so a violating
        # anchor does occur, and a hard bound would turn that into a cascade of
        # infeasible solves. The slack lets the controller recover over a few
        # moves at a cost, which is the deployable behaviour anyway.
        self.dv_space_kmh = dv_space_kmh
        slack_terms = []
        if dv_space_kmh is not None:
            drop = v_ctrl[:-1, :] - v_ctrl[1:, :]
            _, _, s1 = mpc.constraint("space_drop", drop, "<=",
                                      float(dv_space_kmh), soft=True)
            slack_terms.append(s1)
            # the diagonal case: the driver crosses into the next segment AND
            # the sign changes between moves (Hegyi's third inequality)
            if n_ctrl > 1:
                diag = v_ctrl_last[:-1, :] - v_ctrl[1:, 0]
                _, _, s2 = mpc.constraint("space_diag", diag, "<=",
                                          float(dv_space_kmh), soft=True)
                slack_terms.append(s2)

        # cost: assembled from control/cost_functions.py so Phase D can swap
        # objectives without touching the rest of the controller
        v_scale = model.v_ctrl_ub.reshape(-1, 1)
        S = {
            "rho": rho, "v": v, "w": w, "v_ctrl": v_ctrl, "dv": dv,
            "Lkm": np.array(model.seg_lengths_km).reshape(-1, 1),
            "lam": np.array(model.seg_lanes, float).reshape(-1, 1),
            "T_h": p.T, "T_s": p.T_sec, "p": p,
            "v_free": p.v_free_main, "v_scale": v_scale,
            # exposure normaliser for the safety terms, so a weight of 1.0 means
            # the same thing as it does for the others
            "rho_crit": p.rho_crit,
            # The BOTTLENECK's own critical density, which is what a bottleneck
            # regulator's setpoint must be. It was never exposed, so
            # `term_bottleneck_density` normalised the bottleneck by the
            # MAINLINE value (35.14 against 49.40 under PARAMS_BM) and charged
            # from 71% of the density that maximises discharge. See
            # `term_bneck_legacy` for the old behaviour, kept for reproducibility.
            "rho_crit_bneck": (p.rho_crit if p.rho_crit_bneck is None
                               else p.rho_crit_bneck),
            "rho_target_frac": rho_target_frac,
            # normaliser so a weight of 1.0 means comparable influence across
            # costs: roughly the uncontrolled TTS over the horizon, in veh-h
            "tts_scale": 100.0,
            # constant so the acceleration term stays a sum, not a ratio
            "accel_scale": 2000.0,
        }
        self.cost_name = cost
        J = build_cost(cost, S)
        # Penalty on the spatial-gradient slack. Normalised the same way as
        # `term_dvsl` (per km/h of v_scale, squared, averaged) and weighted
        # heavily, so the bound behaves as a constraint on any trajectory that
        # can satisfy it and only bends where the anchor makes that impossible.
        for s in slack_terms:
            J = J + 50.0 * cs.sumsqr(s / float(np.mean(v_scale))) / max(s.numel(), 1)
        mpc.minimize(J)

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
                "max_wall_time": ipopt_max_wall,
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
        if ipopt_extra:
            opts["ipopt"].update(ipopt_extra)
        mpc.init_solver(solver="ipopt", opts=opts)

        self.mpc = mpc
        self._sol_prev = None
        self.stats = []  # (success, status, solve_time_s, cost) per solve
        # FAILURE HANDLING (added 2026-10-06). Diagnosed on the jam-objective MPC
        # (round 41 config): 17 of 19 logged failures are
        # Maximum_Iterations_Exceeded, the rest Error_In_Step_Computation. Both
        # options default OFF, so every stored result is reproduced exactly.
        if on_fail not in ("hold", "shift"):
            raise ValueError("on_fail must be 'hold' or 'shift'")
        self.retry_cold = bool(retry_cold)
        self.on_fail = on_fail
        self._last_plan = None      # (n_ctrl, n_moves) plan of the last success
        self._since_ok = 0          # solves since that plan was made
        self.n_retry = self.n_retry_ok = self.n_shift = 0

    def solve(self, rho0, v0, w0, demand_forecast_vph, v_ctrl_last,
              v_enf0=None):
        """One MPC solve.

        rho0, v0 : current METANET state per segment (veh/km/lane, km/h)
        w0       : current origin queue (veh)
        demand_forecast_vph : array of length Np*M (veh/h per model step)
        v_ctrl_last : VSL currently applied (km/h, per controlled segment)
        v_enf0   : with driver_lag_s, the limit the traffic is currently
                   enforcing (the plant's lagged value). Ignored otherwise.

        Returns (v_ctrl_first_move, info dict). On solver failure the previous
        VSL is returned unchanged and info["success"] is False.
        """
        d_hat = np.asarray(demand_forecast_vph, dtype=float).reshape(1, -1)
        assert d_hat.shape[1] == self.Np * self.M, (
            f"demand forecast must have {self.Np * self.M} entries, "
            f"got {d_hat.shape[1]}"
        )
        v_ctrl_last = np.asarray(v_ctrl_last, dtype=float).reshape(-1, 1)
        if self.driver_comp is not None and self.driver_comp > 1.0:
            # see __init__: at comp > 1 anything posted above legal/comp is the
            # same road as legal/comp, so anchor the rate limit there
            v_ctrl_last = np.minimum(v_ctrl_last,
                                     self.v_ctrl_ub_used.reshape(-1, 1))
        z0 = None
        if self.driver_lag_s is not None:
            if v_enf0 is None:
                raise ValueError("driver_lag_s needs v_enf0, the enforced limit")
            z0 = np.asarray(v_enf0, dtype=float).reshape(-1, 1)

        # The mainstream-origin flow term (-a*log(v/v_free))^(1/a) has an
        # infinite gradient at v == v_free, which NaNs the constraint Jacobian
        # if the measured speed sits exactly at free flow (it does at t=0).
        # Feed the solver speeds fractionally below free flow instead.
        v_cap = np.asarray(self.model.seg_v_free, float)
        v0 = np.minimum(np.asarray(v0, dtype=float), 0.995 * v_cap)
        rho0 = np.asarray(rho0, dtype=float)
        w0 = np.atleast_1d(np.asarray(w0, dtype=float))

        # Without a previous solution to warm-start from, an all-zeros guess
        # makes IPOPT grind (dynamically infeasible, far from optimum). Seed
        # with a feasible rollout of the model under the current VSL instead.
        vals0 = self._sol_prev
        if vals0 is None:
            vals0 = self._rollout_guess(rho0, v0, w0, v_ctrl_last, d_hat, z0)

        pars = {
            "rho_0": rho0,
            "v_0": v0,
            "w_0": w0,
            "d": d_hat,
            "v_ctrl_last": v_ctrl_last,
        }
        if z0 is not None:
            pars["v_enf_0"] = z0
        t0 = time.time()
        sol = self.mpc.solve(pars=pars, vals0=vals0)
        dt = time.time() - t0

        status = str(getattr(sol, "status", "?"))
        # IPOPT's "acceptable" exit is a converged-enough solution, not a
        # failure. csnlp reports success=False for it, so accept it here
        # explicitly rather than discarding a usable answer and holding
        # the previous limits.
        success = (bool(getattr(sol, "success", False))
                   or status == "Solved_To_Acceptable_Level")

        if not success and self.retry_cold and self._sol_prev is not None:
            # The failed solve was warm-started from the previous solution,
            # which may be a poor guess for the current state. One retry from
            # a feasible rollout of the current state.
            self.n_retry += 1
            t1 = time.time()
            sol = self.mpc.solve(pars=pars, vals0=self._rollout_guess(
                rho0, v0, w0, v_ctrl_last, d_hat, z0))
            dt += time.time() - t1
            status = str(getattr(sol, "status", "?"))
            success = (bool(getattr(sol, "success", False))
                       or status == "Solved_To_Acceptable_Level")
            self.n_retry_ok += int(success)

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
            u_fail = v_ctrl_last.ravel()
            if self.on_fail == "shift" and self._last_plan is not None:
                # The standard MPC fallback: the move the last successful
                # plan scheduled for NOW, rather than freezing the sign.
                self._since_ok += 1
                k = min(self._since_ok, self._last_plan.shape[1] - 1)
                u_fail = np.asarray(self._last_plan[:, k]).ravel()
                self.n_shift += 1
            return u_fail, {
                "success": False, "status": status, "solve_time": dt,
                "fallback": self.on_fail,
            }

        self._sol_prev = sol.vals
        plan = np.asarray(sol.vals["v_ctrl"])  # (n_ctrl, n_moves)
        self._last_plan, self._since_ok = plan, 0
        return np.asarray(plan[:, 0]).ravel(), {
            "success": True, "status": status, "solve_time": dt,
            "cost": cost, "plan": plan,
        }

    def _rollout_guess(self, rho0, v0, w0, v_ctrl_last, d_hat, z0=None):
        """Dynamically feasible initial guess: hold the current VSL and roll
        the model forward over the prediction horizon (through the same driver
        model as the NLP, so the guess stays feasible when one is set)."""
        N = self.Np * self.M
        u = np.asarray(v_ctrl_last, dtype=float).ravel()
        x = self.model.pack_x(rho0, v0, w0)
        rho_g, v_g, w_g = [rho0], [v0], [w0]
        z = None if z0 is None else np.asarray(z0, dtype=float).ravel()
        z_g = [] if z is None else [z]
        for k in range(N):
            u_k = u
            if z is not None:
                u_k = u + (z - u) * self._lag_c
                z = u + (z - u) * self._lag_rn
                z_g.append(z)
            elif self.driver_comp is not None:
                u_k = self.driver_comp * u
            x = np.asarray(self.model.F(x, u_k, float(d_hat[0, k]))[0]).ravel()
            r_k, v_k, w_k = self.model.unpack_x(x)
            rho_g.append(r_k)
            v_g.append(v_k)
            w_g.append(w_k)
        n_moves = self.mpc.actions["v_ctrl"].shape[1]
        guess = {
            "rho": np.array(rho_g).T,
            "v": np.array(v_g).T,
            "w": np.array(w_g).T,
            "v_ctrl": np.tile(u.reshape(-1, 1), (1, n_moves)),
        }
        if z_g:
            guess["v_enf"] = np.array(z_g).T
        return guess


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
