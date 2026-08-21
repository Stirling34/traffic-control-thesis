# -*- coding: utf-8 -*-
"""
Can prediction skill be recovered by tuning, and where does the smearing come from?

`metric_audit.py` showed METANET's prediction flattens: by 210 s it predicts a
nearly uniform speed (spread across cells 16 km/h vs SUMO's 30) and correlation
with reality falls to ~0. Density and speed degrade together, so it is not the
speed equation alone.

Prime suspect: the ANTICIPATION term,

    - (eta*T/tau) * (rho_down - rho) / (L * (rho + kappa))

which is mathematically a diffusion term — it pushes density differences flat.
eta was already cut 60 -> 30 for these 200 m segments (ledger 2026-07-21) to stop
stop-go striping. If it is still too strong, it would smear exactly the way we
observe. tau and kappa are swept alongside since all three enter that term.

Scored on what actually matters for prediction, not on open-loop heat maps:
  corr@60/120/210 -- does the predicted pattern still resemble reality?
  spread ratio    -- predicted speed spread / SUMO's. 1.0 = right amount of
                     structure; << 1 = flattened out.
  MAE@210

Usage:  python prediction_skill_sweep.py
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np

from model.metanet_model import MS_TO_KMH, MetanetModel, MetanetParams, load_geometry
from validation.metanet_vs_sumo import BASELINE_NPZ, DEMAND_END_S, DEMAND_VPH
from validation.metanet_prediction_error import HORIZON_S, rollout, sumo_state

LEADS = [60, 120, 210]
STRIDE = 25


def score(geom, sumo, params, t0s, limits):
    model = MetanetModel(geom, params)
    T = params.T_sec
    nh = int(HORIZON_S / T)
    nr = int(sumo["rho"].shape[0] / T)
    d = np.where(np.arange(nr) * T < DEMAND_END_S, DEMAND_VPH, 0.0)
    u = model.v_ctrl_ub
    PV = {l: [] for l in LEADS}
    AV = {l: [] for l in LEADS}
    AR = {l: [] for l in LEADS}
    for t0 in t0s:
        r0, v0 = sumo_state(sumo, t0, geom, model, limits)
        x0 = model.pack_x(r0, v0, float(sumo["pending"][t0]))
        k0 = int(t0 / T)
        w = d[k0:k0 + nh]
        if len(w) < nh:
            w = np.concatenate([w, np.zeros(nh - len(w))])
        _, V = rollout(model, x0, u, w)
        for l in LEADS:
            PV[l].append(V[int(l / T) - 1])
            ar, av = sumo_state(sumo, t0 + l, geom, model, limits)
            AV[l].append(av); AR[l].append(ar)
    out = {}
    for l in LEADS:
        pv, av, ar = np.array(PV[l]), np.array(AV[l]), np.array(AR[l])
        m = ar > 0.5
        out[f"corr{l}"] = float(np.corrcoef(pv[m], av[m])[0, 1])
        if l == LEADS[-1]:
            out["mae"] = float(np.abs(pv[m] - av[m]).mean())
            out["spread"] = float(pv[m].std() / av[m].std())
    return out


def show(title, rows):
    print(f"\n{title}")
    print(f"{'config':>28} {'corr60':>7} {'corr120':>8} {'corr210':>8} "
          f"{'MAE210':>7} {'spread':>7}")
    print("-" * 70)
    for lbl, s in rows:
        print(f"{lbl:>28} {s['corr60']:7.2f} {s['corr120']:8.2f} "
              f"{s['corr210']:8.2f} {s['mae']:7.1f} {s['spread']:7.2f}")


def main():
    geom = load_geometry()
    sumo = dict(np.load(BASELINE_NPZ))
    limits = np.array(geom.limits_ms) * MS_TO_KMH
    t0s = np.arange(60, sumo["rho"].shape[0] - HORIZON_S, STRIDE)
    print(f"{len(t0s)} rollouts per config. Baseline is eta=30, tau=18, kappa=40.")
    print("spread = predicted speed spread / SUMO's; 1.0 means the right amount "
          "of structure.")

    base = score(geom, sumo, MetanetParams(), t0s, limits)

    rows = [("BASELINE eta=30 tau=18 k=40", base)]
    for eta in (5.0, 10.0, 15.0, 60.0):
        rows.append((f"eta={eta:.0f}",
                     score(geom, sumo, MetanetParams(eta=eta), t0s, limits)))
    show("STAGE 1 - anticipation strength (the diffusion term)", rows)
    best_eta = max(rows, key=lambda r: r[1]["corr210"])
    eta_star = (float(best_eta[0].split("=")[1]) if "eta=" in best_eta[0]
                else MetanetParams().eta)
    print(f"  best corr210 at eta = {eta_star}")

    rows = [("BASELINE", base)]
    for tau in (5.0, 10.0, 30.0, 60.0):
        rows.append((f"eta={eta_star:.0f} tau={tau:.0f}",
                     score(geom, sumo, MetanetParams(eta=eta_star, tau_sec=tau),
                           t0s, limits)))
    show("STAGE 2 - speed relaxation time", rows)

    rows = [("BASELINE", base)]
    for kap in (5.0, 10.0, 100.0):
        rows.append((f"eta={eta_star:.0f} kappa={kap:.0f}",
                     score(geom, sumo, MetanetParams(eta=eta_star, kappa=kap),
                           t0s, limits)))
    show("STAGE 3 - anticipation regularisation", rows)

    rows = [("BASELINE T=5", base)]
    for T in (1.0, 2.0):
        rows.append((f"T={T:.0f}s",
                     score(geom, sumo, MetanetParams(T_sec=T), t0s, limits)))
    show("STAGE 4 - model time step (finer = less numerical smearing)", rows)


if __name__ == "__main__":
    main()
