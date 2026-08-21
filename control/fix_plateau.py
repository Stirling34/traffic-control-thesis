# -*- coding: utf-8 -*-
"""
Fixes for the inert-control plateau, tested head to head.

The problem: the VSL enters the model only as `Veq = min(V_fd(rho),
(1+alpha)*v_ctrl)`. Above that kink the control has exactly zero gradient. In
closed loop the applied limit sits a median 35.4 km/h above the kink, which is
4.9 moves at the 7.2 km/h rate limit, against a 7-move horizon. So the
controller would have to spend most of its horizon descending through a region
where moving costs something (the control-variation term) and gains nothing.
It mostly does not, and the limit is inert on ~99% of segment-steps.

Four candidate fixes, each attacking a different part of that sentence:

  baseline   as shipped
  kink_init  start the limits AT the kink instead of at the legal limit, so the
             descent never has to be paid for
  fast_rate  raise the rate limit so the binding region is reachable inside the
             horizon (25 km/h per 30 s move = 2 moves instead of 5)
  low_ub     cap the upper bound at 60 km/h so the whole feasible set is in the
             region where the control bites
  no_dvsl    remove the control-variation term, i.e. take the friction off the
             plateau
  carlson    replace the min() with the Carlson VSL-rate formulation, so the
             control reshapes the whole speed-density curve and has a non-zero
             gradient everywhere. Coefficients A and E measured from SUMO by
             calibration/vsl_fd.py, not imported.

Reported per fix: what fraction of applied limits actually bind, and what that
does to the shockwave and delay, paired against no control on the same seeds.

Usage:  python control/fix_plateau.py --seeds 3
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from model.metanet_model import (MS_TO_KMH, PARAMS_BM, PARAMS_S14,      # noqa: E402
                                 MetanetModel, MetanetParams, load_geometry)
from control.nmpc_metanet import NmpcVslController                    # noqa: E402
from validation.control_authority import shockwave_metrics            # noqa: E402
from validation.sumo_metrics import CONFIG                            # noqa: E402

COST = "speedvar_tts"          # the best-performing objective from Phase D
STEPS = 2500
OUTFILE = os.path.join(_ROOT, "data", "fix_plateau.json")

FIXES = {
    "baseline":  dict(),
    "kink_init": dict(kink_init=True),
    "fast_rate": dict(dv_max_kmh=25.0),
    "low_ub":    dict(v_max_kmh=60.0),
    "no_dvsl":   dict(cost="speedvar_tts_nodvsl"),
    # The unclamped Carlson curve. Kept only as the measured counter-example:
    # above ~40 veh/km/lane it lets a lower sign RAISE the predicted speed
    # (20.6 km/h under a 50 sign against 14.6 under an 88 sign at rho=80), so
    # any gain it shows is the optimiser spending free speed the road does not
    # have. Do not report this row as a result.
    "carlson":   dict(model=dict(vsl_mode="rate", vsl_one_way=False)),
    # The same reshaping, clamped so the sign can only ever slow traffic. This
    # is the physically admissible version: live in the undercritical branch,
    # where a sign genuinely does hold back the approach flow, and inert once a
    # queue has formed, because no sign re-accelerates a queue.
    # Pinned to the ORIGINAL rho_crit-only fit and the hard clamp, so the rows
    # already in data/fix_plateau.json stay reproducible after the re-fit.
    "carlson_oneway": dict(model=dict(vsl_mode="rate", vsl_one_way=True,
                                      vsl_clamp_eps=0.0), coef="rho_only"),
    # Same physics as carlson_oneway, with the two known model defects removed:
    # (A, E) refitted against the measured CAPACITY curve rather than against
    # rho_crit alone (max capacity error 17.2% -> 6.8%, and the model no longer
    # promises a windfall for slowing down), and the clamp smoothed so IPOPT is
    # not asked to differentiate a kink sitting on its own starting point.
    "carlson_refit": dict(model=dict(vsl_mode="rate", vsl_one_way=True)),
    # The capacity drop, switched ON and calibrated to what SUMO actually does.
    # validation/capacity_drop_test.py measures a REAL 11.8% drop at this merge
    # (2434 -> 2148 veh/h, 5 seeds), which is squarely in the 5-20% the
    # literature reports. The prediction model has had alpha_cd = 0 all along,
    # so it believes the bottleneck discharges the same whether or not a queue
    # has formed. Preventing or delaying breakdown therefore earns the optimiser
    # NOTHING in the model, and preventing breakdown is the entire mechanism by
    # which a VSL is supposed to work. rho_cd_ref is set to the density the
    # congested branch actually reaches (median 70), not rho_max: at rho_max
    # only 21% of alpha_cd would ever be realised.
    "capdrop":        dict(model=dict(alpha_cd=0.118, rho_cd_ref=70.0)),
    # The drop plus the reformulated VSL, i.e. give the controller both a reason
    # to act and a usable gradient with which to act.
    "capdrop_rate":   dict(model=dict(alpha_cd=0.118, rho_cd_ref=70.0,
                                      vsl_mode="rate", vsl_one_way=True)),
    # Horizon sweep. validation/prediction_skill.py shows the model's speed
    # correlation with SUMO falls from 0.79 at 30 s to 0.04 at 210 s, crossing
    # below the "assume nothing changes" reference at about 45 s. The controller
    # runs at 210 s, i.e. entirely outside the region where its model knows
    # anything. If that is really what limits performance, then SHORTENING the
    # horizon into the skilful region should help, even though it costs
    # look-ahead. This is the falsifiable version of that claim.
    "horizon_60":  dict(Np=2, Nc=2),
    "horizon_120": dict(Np=4, Nc=3),
    # The ported bake-off winner (PARAMS_S14): 210 s prediction R^2 0.543 on
    # density against the shipped model's 0.160. Note it makes the plateau
    # WORSE, not better: its calibrated v_free_main is 81.5 km/h, so under
    # Veq = min(V_fd, 1.1*u) the sign is inert until it drops below 74.1, a
    # 17.7 km/h dead zone at the legal limit against 8.3 on the shipped model.
    # So the rate reformulation should matter MORE here, which is the point of
    # running both.
    "s14":       dict(preset="s14"),
    "s14_rate":  dict(preset="s14", model=dict(vsl_mode="rate",
                                               vsl_one_way=True)),
    # The capacity drop on the ported model. It was tried on the shipped model
    # and made things worse, but that was through a forecast with no skill at
    # the control horizon (210 s speed R^2 +0.069), so a correct incentive could
    # only produce confident wrong actions. The ported model scores +0.565, so
    # the reason for the earlier failure no longer applies and it is worth one
    # more look. alpha_cd = 0.118 is the drop measured in SUMO by
    # validation/capacity_drop_test.py; rho_cd_ref = 70 is the upstream density
    # SUMO actually reaches while queued (median 70.3), which is where the drop
    # should be fully realised.
    "s14_capdrop": dict(preset="s14",
                        model=dict(alpha_cd=0.118, rho_cd_ref=70.0)),
    # Bounded-METANET (Zhao, Roncoli & Yildirimoglu 2025), the PUBLISHED and
    # citable model. Parity-tested to 5.8e-13 against the numpy original by
    # validation/test_bounded_parity.py. 210 s prediction R^2: density +0.460,
    # speed +0.645 - better than expA on speed (+0.565), which is the metric the
    # acceleration-based cost functions depend on.
    "bm":        dict(preset="bm"),
    # The actuator fix. Under the shipped cap min(V_fd, 1.1*u) the model thinks
    # drivers do 1.10x the sign; measured in SUMO they do 0.89x. With v_free
    # calibrated to 74.0 (bounded) or 81.5 (expA) against legal limits of
    # 84.6-99, the FD term always wins and the sign is inert on >99% of
    # segment-steps. vsl_scaled replaces the cap with (u/u_legal)*v_free, which
    # binds on 9/9 segments with a ZERO dead zone.
    "bm_scaled":  dict(preset="bm",  model=dict(vsl_scaled=True)),
    "s14_scaled": dict(preset="s14", model=dict(vsl_scaled=True)),
}


def _model_for(fix, geom):
    """Build the prediction model this fix wants. Cached, since the NLP build
    dominates: a second model costs ~1.2 s of CasADi graph construction."""
    kw = dict(FIXES[fix].get("model", {}))
    preset = FIXES[fix].get("preset")
    if preset == "s14":
        kw = dict(PARAMS_S14, **kw)
    elif preset == "bm":
        kw = dict(PARAMS_BM, **kw)
    if kw.get("vsl_mode") == "rate":
        coef = json.load(open(os.path.join(_ROOT, "data", "vsl_fd_coef.json")))
        if FIXES[fix].get("coef") == "rho_only":
            # the superseded fit, kept so old rows remain reproducible
            kw["vsl_A"] = coef.get("A_rho_only", coef["A"])
            kw["vsl_E"] = coef.get("E_rho_only", coef["E"])
        else:
            kw["vsl_A"], kw["vsl_E"] = coef["A"], coef["E"]
    key = tuple(sorted(kw.items()))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = MetanetModel(geom, MetanetParams(**kw))
    return _MODEL_CACHE[key]


_MODEL_CACHE = {}


def run(fix, seed, model, geom, cost=None):
    import traci
    kw = dict(FIXES[fix])
    kink_init = kw.pop("kink_init", False)
    cost = kw.pop("cost", cost or COST)
    kw.pop("model", None)
    kw.pop("coef", None)
    kw.pop("preset", None)
    model = _model_for(fix, geom)
    rate_mode = model.p.vsl_mode == "rate"
    ctrl = NmpcVslController(model, cost=cost, **kw)
    p = model.p

    def kink(rho):
        """VSL below which the control actually changes the equilibrium speed."""
        return (p.v_free_main * np.exp(-(1 / p.a) *
                (np.maximum(rho, 0) / p.rho_crit) ** p.a)) / (1 + p.vsl_alpha)

    edges = geom.edge_ids
    Lkm = np.array(geom.lengths_m) / 1000.0
    lanes = np.array(geom.lanes, float)
    limits = np.array(geom.limits_ms) * MS_TO_KMH
    applied = limits[: model.n_ctrl].copy()
    if kw.get("v_max_kmh"):
        applied = np.minimum(applied, kw["v_max_kmh"])

    # SUMO flags must match control/phase_d.py exactly. The no-control rows this
    # is paired against were produced there, and --collision.action changes the
    # physics (SUMO's default teleports on collision), so a mismatch would show
    # up as a control effect.
    traci.start(["sumo", "-c", CONFIG, "--end", str(STEPS), "--no-warnings",
                 "--seed", str(seed), "--time-to-teleport", "-1",
                 "--collision.action", "warn"])
    spd, dens, on_road, pend, bind, vsls = [], [], [], [], [], []
    arrived = nsolve = nfail = 0
    started = False
    for step in range(STEPS):
        traci.simulationStep()
        t = step + 1
        c = np.array([traci.edge.getLastStepVehicleNumber(e) for e in edges])
        vraw = np.array([traci.edge.getLastStepMeanSpeed(e) for e in edges]) * MS_TO_KMH
        v_edges = np.where(c > 0, vraw, np.append(applied, limits[-1]))
        spd.append(np.where(c > 0, vraw, np.nan))
        dens.append(c / (Lkm * lanes))
        on_road.append(c.sum())
        pend.append(len(traci.simulation.getPendingVehicles()))
        arrived += traci.simulation.getArrivedNumber()
        if t % 30:
            continue
        rho_seg = dens[-1][model.seg_edge_index]
        if rho_seg.min() <= 1.0:
            continue
        k = kink(dens[-1][: model.n_ctrl])
        if kink_init and not started:
            # Jump straight to the binding point instead of paying 5 moves of
            # rate-limited descent for no gradient.
            applied = np.clip(k, 20.0, limits[: model.n_ctrl])
            for e, vv in zip(edges[: model.n_ctrl], applied):
                traci.edge.setMaxSpeed(e, float(vv) / MS_TO_KMH)
        started = True
        d_hat = np.where(t + np.arange(ctrl.Np * ctrl.M) * p.T_sec < 1000.0,
                         4000.0, 0.0)
        u, info = ctrl.solve(rho_seg, v_edges[model.seg_edge_index],
                             float(pend[-1]), d_hat, applied)
        nsolve += 1
        if info["success"]:
            applied = np.asarray(u, float)
            for e, vv in zip(edges[: model.n_ctrl], applied):
                traci.edge.setMaxSpeed(e, float(vv) / MS_TO_KMH)
        else:
            nfail += 1
        # "Is the sign doing anything?" needs a different test per formulation.
        # Under the cap form the control is inert unless it is below the kink.
        # Under the rate form there is no kink: any b < 1 reshapes the curve, so
        # the honest test is whether the control has moved off the legal limit.
        bind.append(applied < (0.97 * limits[: model.n_ctrl] if rate_mode else k))
        vsls.append(applied.copy())
    traci.close()

    on_road = np.array(on_road, float)
    pend = np.array(pend, float)
    out = {"fix": fix, "seed": seed, "cost": cost,
           "TTS_h": float((on_road + pend).sum() / 3600.0),
           "throughput": float(arrived),
           "bind_pct": float(100 * np.mean(bind)) if bind else float("nan"),
           "mean_vsl": float(np.mean(vsls)) if vsls else float("nan"),
           "n_solves": nsolve, "n_failed": nfail}
    out.update(shockwave_metrics(np.array(spd), np.array(dens), 1.0, Lkm, lanes))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fixes", type=str, default="",
                    help="comma-separated subset, e.g. baseline,carlson")
    ap.add_argument("--cost", type=str, default=COST,
                    help="objective to run every fix with. The default "
                         "speedvar_tts won Phase D on the SHIPPED model; on the "
                         "ported model the winner is accel_delay, so a "
                         "cross-model comparison must state which it used.")
    a = ap.parse_args()
    wanted = [f for f in a.fixes.split(",") if f] or list(FIXES)
    geom = load_geometry()
    model = MetanetModel(geom, MetanetParams())
    rows = json.load(open(OUTFILE)) if os.path.exists(OUTFILE) else []
    # Rows predating the --cost option carry no "cost" key; they were all
    # speedvar_tts, so default them to it rather than silently re-running them.
    done = {(r["fix"], r["seed"], r.get("cost", COST)) for r in rows}

    # no-control reference, reused from the Phase D results
    pd = json.load(open(os.path.join(_ROOT, "data", "phase_d.json")))
    base = {r["seed"]: r for r in pd if r["cost"] == "nocontrol"}

    for fix in wanted:
        for seed in range(1, a.seeds + 1):
            if (fix, seed, a.cost) in done:
                continue
            t0 = time.time()
            try:
                r = run(fix, seed, None, geom, cost=a.cost)
            except Exception as e:
                print("  FAIL %-10s seed %d: %s" % (fix, seed, str(e)[:60]),
                      flush=True)
                continue
            b = base.get(seed)
            r["d_accel"] = (100 * (r["accel_rms"] / b["accel_rms"] - 1)
                            if b else float("nan"))
            r["d_TTS"] = (100 * (r["TTS_h"] / b["TTS_h"] - 1)
                          if b else float("nan"))
            rows.append(r)
            json.dump(rows, open(OUTFILE, "w"), indent=1)
            print("  %-10s seed %d  bind %5.1f%%  d_accel %+6.1f%%  "
                  "d_TTS %+6.1f%%  meanVSL %5.1f  fails %d/%d  [%.0fs]"
                  % (fix, seed, r["bind_pct"], r["d_accel"], r["d_TTS"],
                     r["mean_vsl"], r["n_failed"], r["n_solves"],
                     time.time() - t0), flush=True)

    def wilcoxon(x):
        """Two-sided signed-rank p, or None below the 6 pairs it needs to ever
        reach 0.05. Same convention as control/analyse_phase_d.py."""
        x = np.asarray([v for v in x if v != 0 and np.isfinite(v)])
        if len(x) < 6:
            return None
        try:
            from scipy.stats import wilcoxon as w
            return float(w(x).pvalue)
        except Exception:
            return None

    print("\n%-11s %7s %9s %9s %8s %8s %8s" %
          ("fix", "active", "d_accel", "p(accel)", "d_TTS", "meanVSL", "fails"))
    print("-" * 68)
    for fix in wanted:
        rs = [r for r in rows if r["fix"] == fix
              and r.get("cost", COST) == a.cost]
        if not rs:
            continue
        da = [r["d_accel"] for r in rs]
        pv = wilcoxon(da)
        fails = sum(r["n_failed"] for r in rs)
        solves = sum(r["n_solves"] for r in rs)
        print("%-11s %6.1f%% %8.1f%% %9s %7.1f%% %8.1f %4d/%d" %
              (fix, np.mean([r["bind_pct"] for r in rs]), np.median(da),
               "n/a" if pv is None else "%.3f" % pv,
               np.median([r["d_TTS"] for r in rs]),
               np.mean([r["mean_vsl"] for r in rs]), fails, solves))
    print("\n'active' is the share of segment-steps on which the sign is doing "
          "anything at all.\nNegative d_accel is less shockwave; negative d_TTS "
          "is less delay. The fixed-sign\nreference to beat is -5.7% d_accel at "
          "+2.2% d_TTS (experiments/fixed_vsl.py, b=0.95).")


if __name__ == "__main__":
    main()
