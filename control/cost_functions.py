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


def _q(rho, v, lam):
    """Flow per cell, veh/h."""
    return rho * v * lam


def _softpos(x, eps=1.0):
    """Smooth max(x, 0). Used for a vehicle count that must not go negative."""
    return 0.5 * (x + cs.sqrt(x * x + eps * eps))


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


def term_queue(S):
    """Origin queue, normalised. Guards against metering the road by starving it."""
    return cs.sumsqr(S["w"] / 100.0) / S["w"].numel()


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
}

_TERMS = {
    "tts": term_tts, "ttd": term_ttd, "delay": term_delay, "accel": term_accel,
    "speed_var": term_speed_var, "accel_dens": term_accel_dens, "density_sq": term_density_sq,
    "density_track": term_density_track, "dvsl": term_dvsl, "queue": term_queue,
}


def build_cost(name, S):
    """Assemble the named cost from the symbol bundle S."""
    if name not in COSTS:
        raise KeyError("unknown cost %r; have %s" % (name, sorted(COSTS)))
    expr = 0
    for term, weight in COSTS[name]["terms"].items():
        expr = expr + weight * _TERMS[term](S)
    return expr
