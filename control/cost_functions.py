# -*- coding: utf-8 -*-
"""
Phase D: the candidate MPC cost functions.

Each entry builds a CasADi expression from the NLP symbols the controller
exposes. Everything must be smooth for IPOPT, so: no `max`, no `abs`, no
indicator functions. Squares instead of absolute values, and variance instead
of standard deviation (the square root has an infinite gradient at zero, and
control drives it exactly there).

Symbols available, all CasADi SX:
    rho      (n_seg, N+1)   density,  veh/km/lane
    v        (n_seg, N+1)   speed,    km/h
    w        (1, N+1)       origin queue, veh
    v_ctrl   (n_ctrl, n_mv) the decision variables, km/h
    dv       (n_ctrl, n_mv) change in v_ctrl between moves, incl. the first
                            move against the currently applied limit
    L, lam, T, p            geometry and METANET parameters

Terms are normalised so weights are interpretable and roughly commensurate
across cost functions.

References for the individual terms:
  TTS                   Hegyi, De Schutter & Hellendoorn 2005, TR-C 13:185-209
  total travel delay    Han 2017 (TU Delft thesis, Tab. 3.2)
  vehicle-wtd accel     Zegeye et al. 2013, TR-C 31:158-171, eqs. 16-19
  squared densities     Alessandri et al. 1999, via Hegyi 2004 sec. 4.2
  control variation     Hegyi 2004 sec. 6.1.1
"""

import casadi as cs
import numpy as np


def _q(rho, v, lam):
    """Flow per cell, veh/h."""
    return rho * v * lam


def _softpos(x, eps=1.0):
    """Smooth max(x, 0). Used for a vehicle count that must not go negative."""
    return 0.5 * (x + cs.sqrt(x * x + eps * eps))


# Smoothing width of the bottleneck objective's one-sided hinge, in units of
# rho_crit. The hinge is centred exactly at rho = rho_crit, which is where a
# bottleneck controller lives, so the ledger's twice-learned lesson about kinks
# on the operating point (the VSL cap's fmin, the origin's fmin) applies in
# principle. Measured, though, it is NOT a performance factor here: solve time
# at a congested state is 0.6-0.7 s across eps = 0.05, 0.15, 0.25 and 0.50 with
# no failures at any of them. (The 80x slowdown that first prompted this check
# turned out to be the machine suspending mid-run, not the objective.) So eps is
# chosen on modelling grounds instead: small enough that the term really only
# penalises EXCEEDANCE rather than charging for merely approaching capacity.
BNECK_EPS = 0.10


# ---------------------------------------------------------------------------
# individual terms, each normalised to be O(1)
# ---------------------------------------------------------------------------
def term_tts(S):
    """Total time spent, including the origin queue. Hegyi 2005 eq. 13."""
    veh = cs.sum1(S["rho"] * S["Lkm"] * S["lam"]) + S["w"]
    return cs.sum2(veh) * S["T_h"] / S["tts_scale"]


def term_ttd(S):
    """Total travel distance, veh-km. Subtracted to form delay."""
    return cs.sum2(cs.sum1(S["rho"] * S["v"] * S["Lkm"] * S["lam"])) * S["T_h"]


def term_delay(S):
    """Total travel delay = TTS - TTD/v_free. Han 2017.

    Penalises time spent in excess of the free-flow journey, so unlike raw TTS
    it does not reward simply removing vehicles from the road.
    """
    tts = (cs.sum2(cs.sum1(S["rho"] * S["Lkm"] * S["lam"]) + S["w"]) * S["T_h"])
    return (tts - term_ttd(S) / S["v_free"]) / S["tts_scale"]


def term_accel(S):
    """Vehicle-weighted squared acceleration. Zegeye et al. 2013, eqs. 16-19.

    Two populations per step: vehicles that STAY in a cell experience
    (v_i(k+1) - v_i(k))/T, and vehicles that CROSS from i to i+1 experience
    (v_{i+1}(k+1) - v_i(k))/T. Each is weighted by how many vehicles actually
    experience it, which is the whole point: an acceleration on an empty
    segment should not count the same as one in heavy traffic.

    This is the shockwave term, and the only one measured to discriminate in
    both the plant and the model.
    """
    rho, v, T_h, T_s = S["rho"], S["v"], S["T_h"], S["T_s"]
    q = _q(rho, v, S["lam"])
    a_t = (v[:, 1:] - v[:, :-1]) / T_s / 3.6                       # m/s^2
    n_t = _softpos(S["Lkm"] * S["lam"] * rho[:, :-1] - T_h * q[:, :-1])
    a_s = (v[1:, 1:] - v[:-1, :-1]) / T_s / 3.6
    n_s = T_h * q[:-1, :-1]
    # NOTE: normalise by a CONSTANT, never by the vehicle total.
    # Dividing by sum(n) was tried and is wrong twice over. It makes the cost a
    # ratio, so the optimiser can lower it by putting MORE vehicles on the road
    # to inflate the denominator. And because that denominator depends on every
    # state, it turns an otherwise sparse Hessian dense across ~900 variables:
    # the NLP then took minutes to build instead of 1.2 s.
    return (cs.sum2(cs.sum1(n_t * a_t ** 2))
            + cs.sum2(cs.sum1(n_s * a_s ** 2))) / S["accel_scale"]


def term_accel_dens(S):
    """Acceleration weighted by DENSITY rather than by flow.

    Diagnostic for why the flow-weighted Zegeye form underperforms here.
    Measured on the uncontrolled baseline, 90% of that objective sits in the
    crossing term whose weight is flow, and its cost splits 32/31/37 across
    free-flow, transition and jammed conditions. So roughly a third of the
    objective is free-flow acceleration, which a speed limit can only reduce by
    slowing healthy traffic: the term partly fights itself.

    Density weighting moves the emphasis to where vehicles are packed, which is
    where the shockwave is.
    """
    v, T_s = S["v"], S["T_s"]
    rho = S["rho"]
    a_t = (v[:, 1:] - v[:, :-1]) / T_s / 3.6
    a_s = (v[1:, 1:] - v[:-1, :-1]) / T_s / 3.6
    w_t = S["Lkm"] * S["lam"] * rho[:, :-1]
    w_s = S["Lkm"][:-1] * S["lam"][:-1] * rho[:-1, :-1]
    return (cs.sum2(cs.sum1(w_t * a_t ** 2))
            + cs.sum2(cs.sum1(w_s * a_s ** 2))) / S["accel_scale"]


def term_speed_var(S):
    """Spatial speed variance, normalised. The variance form, never the std dev.

    A shockwave IS a spread in speed across the road, so this is the most
    direct macroscopic surrogate for it, and it is what CVS is built on.
    """
    v = S["v"] / S["v_free"]
    mean = cs.sum1(v) / v.shape[0]
    return cs.sum2(cs.sum1((v - cs.repmat(mean, v.shape[0], 1)) ** 2)) / v.numel()


def term_speed_grad(S):
    """Squared speed DROP between adjacent segments -- the safety objective.

    Distinct from `term_speed_var`, and the distinction is the point. Speed
    variance measures spread about the road-average speed; it is the standard
    macroscopic crash-risk surrogate, and every one of the seven variants built
    on it in this project made both the shockwave and the road WORSE. It also
    cannot tell apart two roads that differ in exactly the way that matters:

        90/90/90/30/30/30   variance high, ONE 60 km/h step
        90/75/60/45/30      variance similar, no step anywhere

    Rear-end conflicts come from the step, not from the spread. What a driver
    experiences is the speed of the segment they are entering relative to the
    one they are leaving, which is the adjacent difference, not the deviation
    from a road-wide mean no individual driver ever encounters.

    Only DECELERATION is penalised. Speeding up into faster traffic downstream
    is not a conflict, and penalising it symmetrically would push the optimiser
    to flatten the recovery behind a queue, which is neither dangerous nor
    desirable to suppress.
    """
    v = S["v"] / S["v_free"]
    drop = v[:-1, :] - v[1:, :]                 # positive where speed falls
    drop = cs.fmax(drop, 0.0)                   # deceleration only
    return cs.sum2(cs.sum1(drop ** 2)) / max(drop.numel(), 1)


def term_conflict_risk(S):
    """Speed drop weighted by how many vehicles are exposed to it.

    A 60 km/h step on empty road endangers nobody; the same step with a full
    lane behind it is the conflict. Crash-risk models in the safety literature
    scale with both the speed differential and the exposure, so this multiplies
    the squared drop by the density of the segment doing the decelerating.

    This is the exposure-weighted companion to `term_speed_grad`: that one asks
    how sharp the step is, this one asks how many vehicles hit it.
    """
    v = S["v"] / S["v_free"]
    drop = cs.fmax(v[:-1, :] - v[1:, :], 0.0)
    rho_up = S["rho"][:-1, :] / S["rho_crit"]   # exposure, normalised
    return cs.sum2(cs.sum1(rho_up * drop ** 2)) / max(drop.numel(), 1)


def term_accel_temporal(S):
    """Zegeye's TEMPORAL acceleration term ONLY -- the oscillation, not the step.

    THIS IS THE ONE THAT TARGETS THE WAVE. `term_accel` sums two populations:
    vehicles that stay in a cell and experience (v_i(k+1) - v_i(k))/T, and
    vehicles that CROSS from i to i+1 and experience (v_{i+1}(k+1) - v_i(k))/T.
    Measured on the uncontrolled long road, the crossing (spatial) term is
    **93% of the mean square** and the temporal term only 7%.

    Those two are not the same phenomenon. The spatial term is dominated by the
    speed step a vehicle drives INTO at the tail of a standing queue -- that is
    congestion. The temporal term is a vehicle speeding up and slowing down in
    place, which is what a stop-and-go wave does to you. So the headline
    objective has been 93% aimed at the wrong thing.

    It also matters that the two respond differently to control. Measured across
    the league table, the MPC objectives bite roughly three times harder on the
    temporal term than the spatial one (`tts`: -8.2% temporal against -2.7%
    spatial), but because the spatial term dominates the total, that shows up
    barely at all in the headline.

    Same normalisation constant as `term_accel` so the weights stay comparable.
    """
    v, T_s = S["v"], S["T_s"]
    rho = S["rho"]
    q = _q(rho, v, S["lam"])
    a_t = (v[:, 1:] - v[:, :-1]) / T_s / 3.6
    n_t = _softpos(S["Lkm"] * S["lam"] * rho[:, :-1] - S["T_h"] * q[:, :-1])
    return cs.sum2(cs.sum1(n_t * a_t ** 2)) / S["accel_scale"]


def term_upstream_queue(S):
    """Congestion penalised in proportion to how far UPSTREAM it has spread.

    Targets the front REACH directly. Measured on the uncontrolled long road the
    congestion front propagates upstream at about -9.6 km/h and reaches 2.7 km
    before a recovery wave sweeps it forward again at +10.2 km/h. Reach is the
    most threshold-robust of the wave measures -- identical to 0.01 km across
    v_c = 20, 30 and 40 km/h -- so it is the one worth writing an objective
    against.

    Every existing density term (`density_track`, `density_sq`, `bneck_density`)
    weights all segments equally, so a queue is equally bad wherever it sits. It
    is not: a queue that stays at the merge is a bottleneck doing its job, while
    a queue that has propagated a kilometre upstream is the wave. The linear
    weight makes upstream congestion cost more than the same congestion at the
    merge.

    One-sided, via the same smooth hinge as `term_bottleneck_density`: only
    EXCEEDANCE of critical density counts. Free-flowing upstream road is not a
    problem to be corrected.
    """
    rho = S["rho"]
    n = rho.shape[0]
    # weight 1 at the downstream end rising linearly upstream; segment 0 is the
    # most upstream cell the model spans
    w = np.linspace(2.0, 1.0, n).reshape(-1, 1)
    over = _softpos(rho / S["rho_crit"] - 1.0, eps=BNECK_EPS)
    return cs.sum2(cs.sum1(cs.DM(w) * over ** 2)) / max(rho.numel(), 1)


def term_dens_rate(S):
    """Rate of change of density -- how fast congestion BUILDS in each cell.

    Targets the front GROWTH RATE. A congestion front propagating upstream is,
    cell by cell, density rising sharply as the front arrives. Penalising
    |d rho / dt| discourages the sharp arrival, which is the front itself,
    rather than the level of density, which is the queue.

    Squared rather than absolute (the ledger's standing rule: no `abs` in an
    IPOPT objective, square it instead), and normalised by rho_crit so the
    weight is comparable with the other density terms.
    """
    rho = S["rho"] / S["rho_crit"]
    d = (rho[:, 1:] - rho[:, :-1]) / S["T_s"]
    return cs.sum2(cs.sum1(d ** 2)) / max(d.numel(), 1)


def term_density_sq(S):
    """Sum of squared densities: an inhomogeneity measure.

    Alessandri et al. 1999. Equalises density over space and time, which
    indirectly discourages the sharp density fronts a shockwave consists of.
    """
    return cs.sumsqr(S["rho"] / S["p"].rho_max) / S["rho"].numel()


def term_density_track(S):
    """Track a target density. The Assessment-2 cost, kept as the incumbent."""
    return cs.sumsqr(S["rho"] / S["p"].rho_max - S["rho_target_frac"]) \
        / S["rho"].numel()


def term_dvsl(S):
    """Control variation. Hegyi 2004 sec. 6.1.1, including the /v_free scaling.

    Not a performance term. Without it the controller chatters, which is
    undeployable and also unfair to the other terms.
    """
    return cs.sumsqr(S["dv"] / S["v_scale"]) / S["dv"].numel()


def term_restrict(S):
    """How much road is restricted, and by how much. The TARGETING term.

    Added 2026-09-20. Every other term here scores the TRAFFIC; this one scores
    the CONTROLLER's own footprint -- the area of the space-time plane over
    which a limit below the legal one is posted, normalised so 1.0 means every
    sign is held at a full stop for the whole horizon.

    Why it exists. Measured on the straight road at 4600 veh/h, the rule-based
    comparator removes as much of the jam as the MPC (-66 % vs -56 %) while
    holding a mean posted limit of 82.1 km/h against the legal 84.6 -- it
    restricts almost nothing on average and 40 km/h exactly where the jam is.
    `tts` holds 74.2 km/h across the whole road for the whole episode and
    charges +14 s/veh for it. Both clear the jam; only one taxes every driver.
    So the MPC's deficit is not optimisation quality, it is that nothing in its
    objective ever asked it to restrict LESS road.

    This is deliberately NOT `dvsl`. `dvsl` penalises CHANGING the sign, which
    is what stops the controller acting fast enough to catch a wave at all
    (that penalty suppresses three unrelated objectives from ~-55 % to ~0).
    This penalises the LEVEL instead: change as fast as you like, but justify
    every km/h you take away, and give it back as soon as it stops paying.
    """
    return cs.sumsqr(1.0 - S["v_ctrl"] / S["v_scale"]) / S["v_ctrl"].numel()


def term_restrict_l1(S):
    """`restrict`, but linear rather than squared.

    The squared form is cheap to restrict a lot of road slightly; the linear
    form prices every km/h the same wherever it is taken, which is the one that
    actually pushes toward a SPARSE control -- few signs, deep, briefly -- by
    analogy with L1 sparsity. Both are offered because which one matches the
    rule's behaviour is exactly the open question.
    """
    return cs.sum1(cs.sum2(cs.fabs(1.0 - S["v_ctrl"] / S["v_scale"])))         / S["v_ctrl"].numel()


def term_queue(S):
    """Origin queue, normalised. Guards against metering the road by starving it."""
    return cs.sumsqr(S["w"] / 100.0) / S["w"].numel()


def term_tts_road(S):
    """TTS counting ONLY vehicles on the road -- the origin queue excluded.

    A DIAGNOSTIC, not a candidate policy, and it is expected to behave badly.

    The metering result rests on a correlation across objectives plus a weight
    sweep. Neither is a controlled test of the mechanism. This is: `term_tts`
    includes the origin queue `w`, so an objective built on it cannot gain by
    pushing vehicles into that queue -- it pays for them either way. That is the
    most likely reason `tts` is the one objective on the short road that does not
    meter. Delete the `w` term and the same objective should meter hard.

    Predicted: on the SHORT road this reproduces the metering signature
    (large origin-queue growth, apparent smoothing). On the LONG road, where the
    queue has nowhere to hide, it should collapse back towards plain `tts`.
    Confirming both directions turns the metering finding from an association
    into a demonstrated mechanism.
    """
    veh = cs.sum1(S["rho"] * S["Lkm"] * S["lam"])
    return cs.sum2(veh) * S["T_h"] / S["tts_scale"]


def _bneck_hinge(S, rho_ref):
    """Shared body of the bottleneck-density family. See term_bottleneck_density."""
    lam = np.asarray(S["lam"]).ravel()
    idx = [i for i, l in enumerate(lam) if l < 1.5]
    if not idx:
        raise ValueError("no single-lane segment found; lanes=%s" % lam)
    rho_b = S["rho"][idx, :] / rho_ref
    over = _softpos(rho_b - 1.0, eps=BNECK_EPS)
    return cs.sum2(cs.sum1(over ** 2)) / max(rho_b.numel(), 1)


def term_bneck_legacy(S):
    """THE ORIGINAL, BUGGY bottleneck term. Kept ONLY for reproducibility.

    It normalises the BOTTLENECK's density by the MAINLINE critical density,
    because `nmpc_metanet.py` put `p.rho_crit` into the symbol bundle and the
    bottleneck link's own `rho_crit_bneck` was never exposed. Under PARAMS_BM
    that is 35.14 against the bottleneck's own 49.40, so the hinge starts
    charging at 71% of the density that maximises discharge in the model -- the
    classic "nonnecessary underload of the freeway" (ALINEA).

    Every `bneck*` row in `data/sweep/` predating 2026-09-01 was produced by
    this, including the FINDINGS.md sec. 7d claim that the canonical VSL law is
    significantly worse than no control. Keep it so that claim can be re-derived
    and the A/B against the fixed term is like-for-like. DO NOT use it for new
    conclusions.
    """
    return _bneck_hinge(S, S["rho_crit"])


def term_bneck_hi(S):
    """Bottleneck hinge at 1.15x the bottleneck's own critical density.

    Frejo & De Schutter measure the density that actually maximises bottleneck
    outflow in METANET at ~15% above the fundamental diagram's rho_crit
    parameter (36.78 against 32); Carlson et al. use a setpoint ~6% above. So
    the FD parameter is a lower bound on the right setpoint, not the setpoint
    itself. This brackets it from the other side.
    """
    return _bneck_hinge(S, 1.15 * S["rho_crit_bneck"])


def term_bottleneck_density(S):
    """Hold the BOTTLENECK near critical density -- the canonical VSL objective.

    FIXED 2026-09-01: normalises by the BOTTLENECK's own critical density
    (`rho_crit_bneck`, 49.40 under PARAMS_BM) rather than the mainline's
    (35.14). See `term_bneck_legacy` for what this used to do and why every
    stored `bneck*` result is affected.

    This is the control law the VSL literature is actually built on (Papageorgiou
    et al.'s mainstream traffic flow control; Carlson et al. 2010): a merge
    bottleneck discharges most when it is held at capacity, so the controller's
    job is to meter the approach just enough to keep the bottleneck at rho_crit
    and no fuller. Every objective in this set so far sums over ALL segments, so
    none of them expresses it, which is a real gap in the comparison.

    Only EXCEEDANCE is penalised, via a smooth one-sided hinge. Falling below
    critical is not a failure to correct -- it is what the free-flow period looks
    like -- and penalising it symmetrically would push the controller to keep the
    bottleneck artificially full.
    """
    # The bottleneck is the single-lane stretch; take it from the lane counts
    # rather than a hard-coded index so this stays correct if the geometry
    # changes (it already has once, when the approach was extended).
    return _bneck_hinge(S, S["rho_crit_bneck"])


def term_ttd_reward(S):
    """Anti-holding-back reward, proportional to total travel distance.

    Han, Hegyi, Yuan, Hoogendoorn, Papageorgiou & Roncoli 2017 (TR-C 77:405-420)
    sec. 3.2: their objective carries a term `G*U(j)` where "G is a vector of the
    cell lengths multiplied by a small negative value. The term G*U(j) aims to
    maximize the flows, so as to address the so-called HOLDING BACK PROBLEM.
    Holding back may occur for a congested state, where the VSLs cannot have an
    influence... this second term mitigates this undesirable occurrence by
    rewarding the solutions where the flows take higher values." They note it is
    proportional to total travel distance, so their objective is a weighted
    combination of travel-TIME minimisation and travel-DISTANCE maximisation.
    Credited there to Papageorgiou 1995, TR-C 3(1):19-30.

    This is exactly the pathology measured in this project: every acceleration
    objective buys its smoothing by holding traffic back, at a fixed exchange
    rate. The literature's own remedy has never been applied here as a tunable
    term.

    Note the identity worth stating in the write-up: `term_delay` is
    `tts - ttd/v_free`, so `tts:1|ttd_reward:1` is ALGEBRAICALLY `delay:1`. The
    remedy is therefore already present in the objective set, at exactly one
    weight, as `delay_dvsl` -- which is one of only two objectives that reduce
    total time spent on the long road. Sweeping this weight is what tests it.
    """
    return -term_ttd(S) / (S["v_free"] * S["tts_scale"])


def term_bneck_flow(S):
    """Maximise bottleneck discharge, weighted so it is TTS-consistent.

    With n(k) vehicles in the network and n(k+1) = n(k) + T*(q_in - q_out),
    total time spent telescopes to a constant plus a LINEARLY TIME-DISCOUNTED
    sum of the outflow:  min TTS  <=>  max sum_j (N-1-j) * q_out(j).
    So a FLAT `-sum_k q_bneck(k)` is a different and weaker objective: it is
    indifferent between discharging a vehicle now and at the end of the horizon.
    Schmitt & Lygeros (arXiv:1710.09216) prove the equivalence for the discounted
    form; Gomes & Horowitz (TR-C 14(4):244-262) show the undiscounted form is
    degenerate, because a vehicle already inside a bottleneck queue "may choose
    to slow down temporarily, allow a gap to open, and then catch up without
    impacting the total travel time".

    So the weights here descend linearly to zero at the horizon end, which is
    what makes this a reweighting of TTS that concentrates on the bottleneck
    rather than an independent and ill-posed throughput objective.
    """
    lam = np.asarray(S["lam"]).ravel()
    idx = [i for i, l in enumerate(lam) if l < 1.5]
    if not idx:
        raise ValueError("no single-lane segment found; lanes=%s" % lam)
    q_b = _q(S["rho"][idx, :], S["v"][idx, :], S["lam"][idx])
    n = q_b.shape[1]
    wts = cs.DM(np.arange(n - 1, -1, -1.0).reshape(1, -1))    # N-1 ... 0
    return -cs.sum2(cs.sum1(q_b) * wts) * S["T_h"] / S["tts_scale"]


def term_terminal(S):
    """Vehicles still in the network at the END of the prediction horizon.

    A finite-horizon TTS is blind past its own edge. From the identity in
    `term_bneck_flow`, minimising TTS over Np moves weights the discharge of
    move j by (Np - j), so at Np=7 the weights are 7,6,5,4,3,2,1 (verified
    numerically against a direct sum). The last move therefore carries 3.6% of
    the total weight and the final 60 s of a 210 s horizon carries 10.7%, while
    ANY delay pushed beyond move Np-1 carries exactly zero. So late discharge is
    heavily discounted and deferral past the horizon edge is free -- one
    mechanism by which an aggressive objective can look good.

    Charging the terminal state `c` times is exactly equivalent to extending the
    horizon by `c` steps with the state frozen, so it restores the missing tail
    weight. Precedent: AMOC carries a terminal term theta[x(Kp)] (Carlson et al.);
    Gomes & Horowitz and Papamichail et al. instead append a zero-demand
    cool-down so the network empties by the horizon end, which is the open-loop
    equivalent. This project's `term_queue` guard is an ad-hoc, one-sided
    special case of this.
    """
    veh_T = cs.sum1(S["rho"][:, -1] * S["Lkm"] * S["lam"]) + S["w"][:, -1]
    return veh_T * S["T_h"] / S["tts_scale"]


# ---------------------------------------------------------------------------
# the candidates
# ---------------------------------------------------------------------------
# Weights are chosen so each term contributes O(1) on the uncontrolled
# trajectory, so a coefficient of 1.0 means "this matters as much as that".
COSTS = {
    "density_track": dict(
        doc="Incumbent. Track 1.2x critical density, plus VSL smoothness. "
            "This is the Assessment-2 cost and the thing to beat.",
        terms={"density_track": 1.0, "dvsl": 0.1}),

    "tts": dict(
        doc="Total time spent alone. The field standard, and the one measured "
            "to be control-invariant here: expected to do nothing.",
        terms={"tts": 1.0}),

    "tts_dvsl": dict(
        doc="TTS + control variation. The literature-standard VSL objective "
            "(Hegyi 2005). The proper baseline to compare against.",
        terms={"tts": 1.0, "dvsl": 0.1}),

    "delay_dvsl": dict(
        doc="Total travel delay (Han 2017) + control variation. Delay rather "
            "than raw TTS, so removing vehicles is not itself rewarded.",
        terms={"delay": 1.0, "dvsl": 0.1}),

    "accel": dict(
        doc="Vehicle-weighted acceleration alone. Expected to be DEGENERATE: "
            "uniformly slow traffic has no acceleration, so this should "
            "collapse to slowing everything down. Included to demonstrate it.",
        terms={"accel": 1.0}),

    "accel_dvsl": dict(
        doc="Acceleration + control variation, still with no mobility term. "
            "Tests whether smoothing the control alone prevents the collapse.",
        terms={"accel": 1.0, "dvsl": 0.1}),

    "accel_tts": dict(
        doc="THE HYPOTHESIS. Shockwave term traded against delay. This is the "
            "combination the whole project argues for.",
        terms={"accel": 1.0, "tts": 1.0, "dvsl": 0.1}),

    "accel_tts_heavy": dict(
        doc="As accel_tts but weighted 3:1 towards smoothness, to trace the "
            "trade-off curve rather than a single point.",
        terms={"accel": 3.0, "tts": 1.0, "dvsl": 0.1}),

    "accel_delay": dict(
        doc="Shockwave term traded against DELAY rather than TTS.",
        terms={"accel": 1.0, "delay": 1.0, "dvsl": 0.1}),

    "speedvar_tts": dict(
        doc="Speed variance as the shockwave surrogate, traded against delay. "
            "The cheaper, more common alternative to the acceleration term.",
        terms={"speed_var": 1.0, "tts": 1.0, "dvsl": 0.1}),

    "speedvar_tts_nodvsl": dict(
        doc="speedvar_tts with the control-variation term removed. The VSL is "
            "inert on ~99% of segment-steps because it sits well above the "
            "kink in min(V_fd(rho), (1+alpha)*v_ctrl), and descending to the "
            "kink costs dvsl the whole way for no gradient. This takes that "
            "toll off, so the descent is free.",
        terms={"speed_var": 1.0, "tts": 1.0}),

    # -- ramp-metering guard sweep -------------------------------------------
    # Phase D on the working configuration showed EVERY acceleration-based
    # objective buys its smoothness by metering the entrance rather than by
    # smoothing the road: `accel_dens_tts` adds 128 s per vehicle of origin
    # queue against only 18 s of extra on-road time. That is not variable speed
    # limit control, it is relocating the bottleneck upstream where the metric
    # cannot see it.
    #
    # `term_queue` already guards against this and demonstrably works -
    # `accel_tts_queue` (weight 0.5) has the smallest queue growth of the family
    # at +37 s. It is simply outbid by the acceleration term. This sweep raises
    # the guard until the metering stops, to find whether smoothing survives
    # once the controller is forbidden from starving the road.
    "accel_tts_q2": dict(
        doc="accel_tts_queue with the metering guard at 4x. See the sweep note.",
        terms={"accel": 1.0, "tts": 1.0, "queue": 2.0, "dvsl": 0.1}),

    "accel_tts_q5": dict(
        doc="accel_tts_queue with the metering guard at 10x.",
        terms={"accel": 1.0, "tts": 1.0, "queue": 5.0, "dvsl": 0.1}),

    "accel_tts_q20": dict(
        doc="accel_tts_queue with the metering guard at 40x. Expected to "
            "suppress metering entirely; the question is what smoothing is "
            "left when it does.",
        terms={"accel": 1.0, "tts": 1.0, "queue": 20.0, "dvsl": 0.1}),

    # -- safety objectives ----------------------------------------------------
    # The thesis asks which cost function best increases SAFETY, and until now
    # nothing in this set was trying to. The seven `speedvar` objectives are the
    # standard macroscopic crash-risk surrogate and all seven made both the
    # shockwave and the road worse; the one that was safety-instrumented
    # (`speedvar_dvsl`) produced the WORST TET of the six measured, +10.2%.
    #
    # These target the speed DROP between adjacent segments instead of the
    # spread about the road mean - the step a driver actually drives into,
    # which is what produces a rear-end conflict. See term_speed_grad.
    "grad_tts": dict(
        doc="Spatial speed-drop penalty traded against delay. The safety "
            "objective the project was missing: penalises the step a driver "
            "drives into, not the road-wide variance.",
        terms={"speed_grad": 1.0, "tts": 1.0, "dvsl": 0.1}),

    "grad_dvsl": dict(
        doc="Speed-drop penalty with no mobility term, to isolate whether the "
            "delay term is doing the work (the lesson from accel vs accel_tts).",
        terms={"speed_grad": 1.0, "dvsl": 0.1}),

    "risk_tts": dict(
        doc="Speed drop weighted by the density exposed to it, traded against "
            "delay. Closer to a crash-risk model: conflict rate scales with "
            "both the differential and the number of vehicles meeting it.",
        terms={"conflict_risk": 1.0, "tts": 1.0, "dvsl": 0.1}),

    "risk_tts_heavy": dict(
        doc="As risk_tts but weighted 3:1 towards safety, to trace the "
            "trade-off rather than a single point.",
        terms={"conflict_risk": 3.0, "tts": 1.0, "dvsl": 0.1}),

    "density_sq_dvsl": dict(
        doc="Inhomogeneity (Alessandri 1999). Equalise density rather than "
            "target a level. Reported at -3.8% for VSL-only in the literature.",
        terms={"density_sq": 1.0, "dvsl": 0.1}),

    "speedvar_dvsl": dict(
        doc="Speed variance + control smoothness, NO mobility term. Isolates "
            "whether the TTS term is doing the work in speedvar_tts, since "
            "accel_dvsl (also no TTS) scored the same as speedvar_tts.",
        terms={"speed_var": 1.0, "dvsl": 0.1}),

    "speedvar_heavy": dict(
        doc="Speed variance weighted 3:1 against TTS. The acceleration term got "
            "WORSE when weighted up (accel_tts_heavy); does variance behave the "
            "same way, or is that specific to the acceleration form?",
        terms={"speed_var": 3.0, "tts": 1.0, "dvsl": 0.1}),

    "accel_dens_tts": dict(
        doc="Acceleration weighted by DENSITY not flow, traded against TTS. "
            "Direct test of the theory that flow weighting spreads the "
            "objective across free-flow traffic instead of concentrating it "
            "on the jam.",
        terms={"accel_dens": 1.0, "tts": 1.0, "dvsl": 0.1}),

    "speedvar_tts_d002": dict(
        doc="speedvar_tts with the control-variation weight at 0.02. Traces the "
            "aggressiveness curve directly: same objective, only the restraint "
            "term moves. Outcome looks inverted-U in mean VSL, best near 80 "
            "km/h and worse both sides, so this locates the turning point.",
        terms={"speed_var": 1.0, "tts": 1.0, "dvsl": 0.02}),

    "speedvar_tts_d05": dict(
        doc="speedvar_tts with the control-variation weight at 0.5. Traces the "
            "aggressiveness curve directly: same objective, only the restraint "
            "term moves. Outcome looks inverted-U in mean VSL, best near 80 "
            "km/h and worse both sides, so this locates the turning point.",
        terms={"speed_var": 1.0, "tts": 1.0, "dvsl": 0.5}),

    "speedvar_tts_d20": dict(
        doc="speedvar_tts with the control-variation weight at 2.0. Traces the "
            "aggressiveness curve directly: same objective, only the restraint "
            "term moves. Outcome looks inverted-U in mean VSL, best near 80 "
            "km/h and worse both sides, so this locates the turning point.",
        terms={"speed_var": 1.0, "tts": 1.0, "dvsl": 2.0}),

    "accel_tts_queue": dict(
        doc="accel_tts plus an origin-queue penalty, so the controller cannot "
            "buy smoothness on the road by starving it at the entrance.",
        terms={"accel": 1.0, "tts": 1.0, "queue": 0.5, "dvsl": 0.1}),

    # -- the canonical VSL objective, previously missing ----------------------
    "bneck": dict(
        doc="Hold the bottleneck at critical density. The control law the VSL "
            "literature is built on, and absent from this set until now: every "
            "other objective sums over all segments, so none of them can "
            "express 'keep the merge at capacity'.",
        terms={"bneck_density": 1.0, "dvsl": 0.1}),

    "bneck_tts": dict(
        doc="Bottleneck-density control traded against delay -- the deployable "
            "form, and the direct competitor to tts.",
        terms={"bneck_density": 1.0, "tts": 1.0, "dvsl": 0.1}),

    "bneck_tts_heavy": dict(
        doc="As bneck_tts but 3:1 towards holding the bottleneck, to trace the "
            "trade-off rather than a single point.",
        terms={"bneck_density": 3.0, "tts": 1.0, "dvsl": 0.1}),

    # -- best mobility term + best-behaved safety term ------------------------
    "tts_grad": dict(
        doc="The two things that actually worked, combined: tts is the only "
            "objective that improves the road, and the adjacent speed-drop "
            "penalty is the only safety term that does not meter or degrade "
            "mobility. Never tested together.",
        terms={"tts": 1.0, "speed_grad": 1.0, "dvsl": 0.1}),

    # -- mechanism test for the metering finding ------------------------------
    "tts_road": dict(
        doc="DIAGNOSTIC. tts with the origin queue removed from the objective. "
            "Tests whether tts avoids metering precisely BECAUSE it counts "
            "queued vehicles. Expected to meter hard on the short road and to "
            "collapse back to tts on the long one. See term_tts_road.",
        terms={"tts_road": 1.0, "dvsl": 0.1}),
}

JAM_V_C = 30.0      # the project's v_c: the primary metric counts cells below it
JAM_EPS = 3.0       # km/h width of the smooth step, so IPOPT sees a gradient


def term_jam(S):
    """Predicted JAM vehicle-hours: vehicles in cells moving below 30 km/h.

    The objective written to match the PRIMARY METRIC (`jam_vc30_veh_h`,
    validation/wave_metrics.jam_metrics) rather than a proxy for it. Added
    2026-10-05 with PARAMS_BM_JAM: on the straight road shrinking the jam COSTS
    total travel time (SUMO: every jam-reducing policy adds delay; the
    jam-calibrated model agrees), so a `tts` objective correctly refuses to
    restrict, and an objective that values the jam itself is needed for the
    controller to act on purpose. The jam-calibrated model predicts a
    time-limited 40 km/h restriction cuts an established jam 33-44 %, so this
    term has a real channel to work through.

    NOT a retry of `upstream_queue` (ledger 2026-09-06): that penalised density
    EXCEEDANCE weighted by distance, on the merge, under PARAMS_BM, which
    believed the whole signed stretch was congested. This counts slow
    VEHICLES (speed-based, like the metric) on a model calibrated to clear a
    jam the way SUMO does.

    Smooth step: weight 0.5*(1 + tanh((v_c - v)/eps)) per vehicle. Normalised
    like `term_tts` so weights are comparable.
    """
    veh = S["rho"] * S["Lkm"] * S["lam"]
    slow = 0.5 * (1.0 + cs.tanh((JAM_V_C - S["v"]) / JAM_EPS))
    return cs.sum2(cs.sum1(veh * slow)) * S["T_h"] / S["tts_scale"]


_TERMS = {
    "tts": term_tts, "ttd": term_ttd, "delay": term_delay, "accel": term_accel,
    "speed_var": term_speed_var, "speed_grad": term_speed_grad,
    "conflict_risk": term_conflict_risk, "accel_dens": term_accel_dens, "density_sq": term_density_sq,
    "density_track": term_density_track, "dvsl": term_dvsl, "queue": term_queue,
    "tts_road": term_tts_road, "bneck_density": term_bottleneck_density,
    # added 2026-09-01, see docs/research/2026-09-01-why-the-frontier-is-one-dimensional.md
    "bneck_legacy": term_bneck_legacy, "bneck_hi": term_bneck_hi,
    "ttd_reward": term_ttd_reward, "bneck_flow": term_bneck_flow,
    "terminal": term_terminal,
    # added 2026-09-05 -- the first objectives in this project aimed at
    # WAVE PROPAGATION rather than at acceleration magnitude
    "accel_temporal": term_accel_temporal,
    "upstream_queue": term_upstream_queue,
    "dens_rate": term_dens_rate,
    # added 2026-09-20 -- the first term that scores the CONTROLLER's own
    # footprint rather than the traffic. See term_restrict.
    "restrict": term_restrict, "restrict_l1": term_restrict_l1,
    # added 2026-10-05 -- the primary metric itself; see term_jam
    "jam": term_jam,
}


def parse_spec(name):
    """Weights for a cost, either a name from COSTS or an inline spec.

    An inline spec is `term:weight` pairs joined by `|`, e.g.
    `accel:2|tts:1|dvsl:0.1`. It exists so a weight sweep does not require 30
    near-duplicate entries in COSTS.

    This matters more than convenience. The single biggest stated limitation of
    the comparison is that weights were SCREENED, not optimised: every objective
    was run at one arbitrary setting, so the league table compares particular
    points rather than families at their best, and a reviewer can fairly say the
    acceleration family was never given a fair chance. Tracing a proper
    trade-off frontier per family needs arbitrary weights to be cheap.
    """
    if "|" in name or ":" in name:
        terms = {}
        for part in name.split("|"):
            k, _, w = part.partition(":")
            k = k.strip()
            if k not in _TERMS:
                raise KeyError("unknown term %r in spec %r; have %s"
                               % (k, name, sorted(_TERMS)))
            terms[k] = float(w)
        if not terms:
            raise KeyError("empty cost spec %r" % name)
        return terms
    if name not in COSTS:
        raise KeyError("unknown cost %r; have %s" % (name, sorted(COSTS)))
    return COSTS[name]["terms"]


def build_cost(name, S):
    """Assemble the named cost from the symbol bundle S."""
    expr = 0
    for term, weight in parse_spec(name).items():
        expr = expr + weight * _TERMS[term](S)
    return expr
