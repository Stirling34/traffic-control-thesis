# -*- coding: utf-8 -*-
"""
Parity and correctness test for the DRIVER-AWARE NMPC (2026-09-29).

`NmpcVslController(driver_lag_s=..., driver_comp=...)` puts the plant's driver
model (phase_d.run_closed_loop `actuation_lag_s` / `compliance`) into the
prediction. Four things must hold before any closed-loop result from it is
trusted (CLAUDE.md / ledger: parity-test every change to a model):

  1. DEFAULT PARITY. With both options off, the controller is the controller
     every stored result used: same solves, bit for bit, against a frozen copy
     of the pre-change file (pass its path as argv[1]; skipped if absent).
  2. LAG FORMULA. The per-model-step mean and end state used in the NLP equal
     what phase_d's per-second recursion actually enforces, including across a
     change of target mid-horizon.
  3. NLP CONSISTENCY, in two parts so wiring and solver tolerance cannot be
     confused. (a) WIRING: rolling the controller's own optimal plan through
     the PLANT's driver model (per-second lag / min(comp*u, legal)) gives the
     same trajectory as rolling it through the NLP's closed-form driver model,
     to round-off. (b) The NLP's own states satisfy that model step by step to
     within IPOPT's acceptable_constr_viol_tol (1e-4). Checking the replayed
     trajectory against the NLP states directly conflates the two: measured,
     a comp=1.5 solve that exits Solved_To_Acceptable_Level carries a 6e-6
     one-step residual that compounds to 7e-5 over 42 steps.
  4. comp = 1 is a no-op.

Usage:  python validation/test_driver_aware.py [path/to/nmpc_metanet_orig.py]
"""

import importlib.util
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "sumo"))

import build_straight_net as bsn                                      # noqa: E402
from control.nmpc_metanet import NmpcVslController                    # noqa: E402
from model.metanet_model import (MetanetModel, MetanetParams, PARAMS_BM,  # noqa: E402
                                 load_geometry, straight_model_geometry)

FAILS = []


def check(name, ok, detail=""):
    print("%-58s %s %s" % (name, "PASS" if ok else "*** FAIL ***", detail))
    if not ok:
        FAILS.append(name)


def plant_lag(z0, targets, L, n_sub):
    """phase_d's recursion, verbatim: each second close 1/L of the gap; the
    value set at second s is what the traffic drives during second s+1.
    `targets` is one posted value per MODEL step. Returns the per-step mean of
    what was enforced and the enforced value at each step boundary."""
    a = np.array(z0, float)
    means, ends = [], [a.copy()]
    for tgt in targets:
        tgt = np.asarray(tgt, float)
        seen = []
        for _ in range(n_sub):
            seen.append(a.copy())                 # enforced during this second
            step = (tgt - a) / float(L)
            near = np.abs(tgt - a) <= np.abs(step)
            a = np.where(near, tgt, a + step)
        means.append(np.mean(seen, axis=0))
        ends.append(a.copy())
    return np.array(means), np.array(ends)


def loaded_state(model, demand, n_steps):
    """A congested METANET state: the merge-fitted model's capacity (3714
    veh/h) is below 4600, so a few minutes of that demand loads the road."""
    x = model.x0_empty()
    for _ in range(n_steps):
        x = np.asarray(model.F(x, model.v_ctrl_ub, demand)[0]).ravel()
    return model.unpack_x(x)


def main():
    geom = load_geometry(bsn.net_path(""), allow_straight=True)
    mgeom, _ = straight_model_geometry(geom, bsn.N_UP,
                                       n_seg=bsn.N_CTRL + bsn.N_DOWN)
    model = MetanetModel(mgeom, MetanetParams(**PARAMS_BM, vsl_scaled=True),
                         n_ctrl=bsn.N_CTRL)
    T = model.p.T_sec
    n_ctrl = model.n_ctrl
    legal = model.v_ctrl_ub.copy()
    kw = dict(cost="tts", ipopt_max_iter=300, ipopt_max_wall=1e6)

    states = [loaded_state(model, 4600.0, n) for n in (60, 120, 200)]
    d_hat = np.full(7 * 6, 4600.0)
    u_prev = [legal.copy(), np.full(n_ctrl, 70.0),
              np.linspace(60.0, 80.0, n_ctrl)]

    # ---- 1. default parity against the frozen pre-change file -------------
    orig_path = sys.argv[1] if len(sys.argv) > 1 else None
    if orig_path and os.path.exists(orig_path):
        spec = importlib.util.spec_from_file_location("nmpc_orig", orig_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        a = mod.NmpcVslController(model, **kw)
        b = NmpcVslController(model, **kw)
        worst = 0.0
        for (rho, v, w), up in zip(states, u_prev):
            ua, ia = a.solve(rho, v, w, d_hat, up)
            ub, ib = b.solve(rho, v, w, d_hat, up)
            worst = max(worst, float(np.max(np.abs(ua - ub))),
                        abs(ia.get("cost", 0.0) - ib.get("cost", 0.0)))
            if ia["status"] != ib["status"]:
                worst = np.inf
        check("1. defaults reproduce the pre-change controller", worst == 0.0,
              "(max |diff| %.1e over %d warm-started solves)" % (worst, 3))
    else:
        print("1. default parity: SKIPPED (no frozen copy given)")

    # ---- 2. lag formula vs the plant's per-second recursion ---------------
    for L in (10.0, 20.0, 40.0):
        c = NmpcVslController(model, driver_lag_s=L, **kw)
        n_sub = int(round(T))
        z0 = np.linspace(84.6, 50.0, n_ctrl)
        tg = [np.full(n_ctrl, 40.0)] * 6 + [np.full(n_ctrl, 84.6)] * 6
        pm, pe = plant_lag(z0, tg, L, n_sub)
        z = z0.copy()
        mm, me = [], [z.copy()]
        for u in tg:
            mm.append(u + (z - u) * c._lag_c)
            z = u + (z - u) * c._lag_rn
            me.append(z.copy())
        err = max(np.max(np.abs(np.array(mm) - pm)),
                  np.max(np.abs(np.array(me) - pe)))
        check("2. lag L=%g s: step means and states match plant" % L,
              err < 1e-10, "(max |diff| %.1e km/h)" % err)

    # ---- 3. NLP consistency: replay the optimal plan through the plant ----
    def u_eff_plant(ctrl, u, z):
        """What phase_d's plant enforces over one model step."""
        if z is not None:
            m, e = plant_lag(z, [u], ctrl.driver_lag_s, int(round(T)))
            return m[0], e[-1]
        if ctrl.driver_comp is not None:
            return np.minimum(ctrl.driver_comp * u, legal), None
        return u, None

    def u_eff_nlp(ctrl, u, z):
        """The NLP's closed form for the same thing."""
        if z is not None:
            return u + (z - u) * ctrl._lag_c, u + (z - u) * ctrl._lag_rn
        if ctrl.driver_comp is not None:
            return ctrl.driver_comp * u, None
        return u, None

    def replay(ctrl, rho, v, w, plan, z0, law):
        M = ctrl.M
        x = model.pack_x(rho, v, w)
        xs = [x]
        z = None if z0 is None else np.array(z0, float)
        for k in range(ctrl.Np * M):
            u = plan[:, min(k // M, plan.shape[1] - 1)]
            u_eff, z = law(ctrl, u, z)
            x = np.asarray(model.F(x, u_eff, d_hat[k])[0]).ravel()
            xs.append(x)
        return np.array(xs).T

    for label, extra, z0 in (
            ("lag L=20 s", dict(driver_lag_s=20.0), np.full(n_ctrl, 84.6)),
            ("lag L=40 s", dict(driver_lag_s=40.0),
             np.linspace(84.6, 45.0, n_ctrl)),
            ("comp 1.5", dict(driver_comp=1.5), None),
            ("comp 1.25", dict(driver_comp=1.25), None)):
        ctrl = NmpcVslController(model, **extra, **kw)
        rho, v, w = states[1]
        u, info = ctrl.solve(rho, v, w, d_hat, legal, v_enf0=z0)
        if not info["success"]:
            check("3. %s: NLP consistency" % label, False,
                  "(solve failed: %s)" % info["status"])
            continue
        vals = ctrl._sol_prev
        plan = np.asarray(info["plan"])
        X_nlp = np.vstack([np.asarray(vals["rho"]), np.asarray(vals["v"]),
                           np.asarray(vals["w"])])
        v0 = np.minimum(v, 0.995 * np.asarray(model.seg_v_free, float))
        X_plant = replay(ctrl, rho, v0, w, plan, z0, u_eff_plant)
        X_form = replay(ctrl, rho, v0, w, plan, z0, u_eff_nlp)
        wiring = float(np.max(np.abs(X_plant - X_form)))
        # one-step residual of the NLP's own states under its own model
        Z = (np.asarray(vals["v_enf"]) if "v_enf" in vals else None)
        res = 0.0
        for k in range(X_nlp.shape[1] - 1):
            u = plan[:, min(k // ctrl.M, plan.shape[1] - 1)]
            u_eff, _ = u_eff_nlp(ctrl, u, None if Z is None else Z[:, k])
            xn = np.asarray(model.F(X_nlp[:, k], u_eff, d_hat[k])[0]).ravel()
            res = max(res, float(np.max(np.abs(xn - X_nlp[:, k + 1]))))
        moved = float(np.max(np.abs(plan - legal[:, None])))
        extra_txt = ""
        if "driver_comp" in extra:
            extra_txt = ", posted ub %.2f" % ctrl.v_ctrl_ub_used.max()
        check("3a. %s: plant driver model = NLP driver model" % label,
              wiring < 1e-9, "(max |diff| %.1e; plan moves %.1f km/h%s)"
              % (wiring, moved, extra_txt))
        check("3b. %s: NLP states obey it (%s)" % (label, info["status"][:14]),
              res <= 1e-4, "(one-step residual %.1e)" % res)

    # ---- 4. comp = 1 is a no-op --------------------------------------------
    a = NmpcVslController(model, **kw)
    b = NmpcVslController(model, driver_comp=1.0, **kw)
    rho, v, w = states[2]
    ua, _ = a.solve(rho, v, w, d_hat, u_prev[1])
    ub, _ = b.solve(rho, v, w, d_hat, u_prev[1])
    err = float(np.max(np.abs(ua - ub)))
    check("4. comp = 1.0 gives the unaware controller's move", err < 1e-8,
          "(max |diff| %.1e)" % err)

    print("\nVERDICT: %s" % ("ALL PASS" if not FAILS else
                             "%d FAILED: %s" % (len(FAILS), ", ".join(FAILS))))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
