# -*- coding: utf-8 -*-
"""
expA -- asymmetric anticipation, made VALID and calibrated.

Fork of `experiments/exp4_metanet_variants/core.py` (parity-tested against the
project's sym-metanet model to ~1e-13 with every structural switch off;
`test_parity.py` re-asserts that here). Four things are added on top:

  1. `antic="smooth"` -- the asymmetry as a sigmoid in the density gradient
     rather than a hard switch. Differentiable everywhere, so it stays
     IPOPT-friendly for the NMPC, and it avoids a non-physical kink at
     grad == 0.
  2. `rho_crit_bneck` -- a separate critical density (hence capacity) for the
     1-lane bottleneck link. Discharge is a bench validity criterion and it is
     set by the bottleneck's capacity, so it needs its own degree of freedom.
  3. BATCHED STATE. Every state array may carry leading axes, so ~250
     prediction rollouts run as one vectorised sweep.
  4. BATCHED PARAMETERS. `Model([p1, p2, ...])` runs a whole differential-
     evolution population in one sweep. This matters because the cost of a step
     is dominated by numpy CALL overhead on 11-element arrays, not by element
     count -- so an entire population costs about as much as one individual.
     `test_batch.py` asserts batched == serial to ~1e-12.

WHY THIS ARM EXISTS
-------------------
exp4 found that weighting only the ACCELERATING half of METANET's anticipation
term (antic="asym", chi=0) lifts 210 s prediction correlation from 0.035 to
0.518 -- but discharge collapses from 2325 to 567 veh/h, so it is INVALID.

The mechanism of the collapse, worked out here and the thing this arm is built
around: METANET's downstream boundary is `rho_down = min(rho_N, rho_crit)`, so
at an oversaturated exit the density gradient is strongly NEGATIVE and stock
METANET's anticipation term hands the exit cell a large positive speed
increment. That increment is what makes the exit discharge at capacity. Set
chi=0 and it disappears, so the exit cell relaxes onto the FD's congested branch
(v = Veq(rho), which falls faster than 1/rho), flow falls, density rises, flow
falls further -- a runaway. The queue-smearing fix and the discharge are served
by the same term.

So chi<1 needs something else to hold the exit at capacity. Two candidates are
implemented, both standard first-order theory rather than invention:

  * `cap_demand`  -- a cell cannot send more than its capacity. An UPPER bound,
    so it cures the 4x-capacity acceleration but cannot cure the collapse.
  * `veq_mode="godunov"` -- the equilibrium speed becomes the supply/demand
    speed  Veq = min(v_free, Q/rho, S_down/(lam*rho))  instead of a
    single-valued FD. On the congested branch this is Q/rho, i.e. flow == Q, so
    an exit with open road ahead discharges at capacity no matter how dense it
    is, and the queue instead forms where the SUPPLY term binds. Speed is still
    an independent state relaxed over tau, so the model stays second order and
    acceleration/jerk remain meaningful.

CONVENTIONS
    density veh/km/lane, speed km/h, length km, time h.
    State arrays are (..., NSEG); the origin queue `w` is (..., 1) so it
    broadcasts against them; demand `d` is a scalar or (..., 1).
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# geometry (constants of the fixed SUMO baseline; see experiments/bench.py)
# --------------------------------------------------------------------------
N_MAIN = 9
N_BNECK = 2
L_MAIN = 1.796 / 9          # km, mean of the nine 2-lane edge lengths (199.6 m)
L_BNECK = 0.496 / 2         # km
LAM = np.array([2.0] * N_MAIN + [1.0] * N_BNECK)
LSEG = np.array([L_MAIN] * N_MAIN + [L_BNECK] * N_BNECK)
NSEG = N_MAIN + N_BNECK
SEG_EDGE = np.array(list(range(N_MAIN)) + [N_MAIN] * N_BNECK)
IS_MAIN = np.arange(NSEG) < N_MAIN

_GEOM = {}


def geometry():
    """Read the SUMO geometry once and cross-check the hard-coded constants.

    Returns (v_free_main, v_free_bneck, posted_limits_kmh_of_the_9_main_edges).
    """
    if not _GEOM:
        from model.metanet_model import MS_TO_KMH, load_geometry
        g = load_geometry()
        assert g.n_main == N_MAIN
        assert abs(float(np.mean(g.lengths_m[:N_MAIN])) / 1000.0 - L_MAIN) < 1e-9
        assert abs(g.lengths_m[-1] / 1000.0 / N_BNECK - L_BNECK) < 1e-9
        _GEOM["vm"] = float(np.mean(g.main_limits_kmh))
        _GEOM["vb"] = float(g.limits_ms[-1] * MS_TO_KMH)
        _GEOM["ub"] = np.array(g.main_limits_kmh, float)
    return _GEOM["vm"], _GEOM["vb"], _GEOM["ub"]


# --------------------------------------------------------------------------
@dataclass
class P:
    """Model parameters. Defaults reproduce the project's current METANET."""

    # --- stock METANET -----------------------------------------------------
    T_sec: float = 5.0
    rho_crit: float = 40.5
    a: float = 1.867
    tau_sec: float = 18.0
    eta: float = 30.0
    kappa: float = 40.0
    v_free_main: Optional[float] = None
    v_free_bneck: Optional[float] = None
    rho_max: float = 1000.0 / 5.5      # 181.8, used by the FD/origin only
    vsl_alpha: float = 0.1
    phi: Optional[float] = None        # lane-drop speed penalty, last L1 segment

    # --- expA addition: the bottleneck gets its own capacity ---------------
    rho_crit_bneck: Optional[float] = None   # None -> same as rho_crit

    # --- structural switches ----------------------------------------------
    fd: str = "exp"                    # 'exp' | 'exp2' | 'quad'
    a_cong: Optional[float] = None     # 'exp2': congested-branch exponent
    quad_width: float = 0.0            # 'quad': plateau half-width, fraction of rho_crit

    antic: str = "std"                 # 'std' | 'asym' | 'smooth' | 'limited'
    chi: float = 1.0                   # weight on the accelerating half
    grad_scale: float = 5.0            # 'smooth': sigmoid width, veh/km/lane
    antic_cap: float = 1e9             # 'limited': cap on |anticipation| (km/h/step)

    supply: bool = False               # CTM supply bound on the inter-cell flux
    w_cong: float = 18.0               # backward wave speed of the supply curve (km/h)
    rho_jam: float = 181.8             # supply vanishes here (veh/km/lane)
    cap_demand: bool = False           # CTM demand bound: a cell cannot discharge
                                       # above its capacity, whatever v says
    veq_mode: str = "metanet"          # 'metanet' | 'godunov' (supply-demand target)

    substeps: int = 1                  # sub-steps per T (numerical stabiliser)

    @property
    def T(self):
        return self.T_sec / 3600.0

    @property
    def tau(self):
        return self.tau_sec / 3600.0


# structural fields that must be identical across a parameter batch
_STRUCT = ("T_sec", "fd", "antic", "supply", "cap_demand", "veq_mode",
           "substeps", "vsl_alpha", "rho_max")


class Model:
    """Numpy METANET with structural switches. See module docstring.

    `p` is a single P, or a list of P sharing every structural field (a whole
    optimiser population). `pad` is the number of state axes BETWEEN the
    parameter-batch axis and the segment axis:
        pad=0 -> state (B, NSEG)          [open loop, one trajectory each]
        pad=1 -> state (B, R, NSEG)       [B parameter sets x R rollouts]
    """

    def __init__(self, p=None, pad=0):
        if p is None:
            p = P()
        self.batched = isinstance(p, (list, tuple))
        plist = list(p) if self.batched else [p]

        vm, vb, ub = geometry()
        plist = [replace(q,
                         v_free_main=vm if q.v_free_main is None else q.v_free_main,
                         v_free_bneck=vb if q.v_free_bneck is None else q.v_free_bneck)
                 for q in plist]
        p0 = plist[0]
        for q in plist[1:]:
            for f in _STRUCT:
                assert getattr(q, f) == getattr(p0, f), \
                    f"batched parameters must share structural field {f!r}"
        assert (p0.phi is None) == all(q.phi is None for q in plist), \
            "phi must be None for all or none of a parameter batch"
        self.p = p0
        self.plist = plist
        self.B = len(plist)
        self._pad = pad

        def pv(name, fallback=None):
            """Per-parameter-set scalar -> float, or array broadcasting against
            (..., NSEG) with the parameter batch on the leading axis.
            `fallback` names another field to use where this one is None."""
            vals = []
            for q in plist:
                x = getattr(q, name)
                vals.append(getattr(q, fallback) if x is None else x)
            if not self.batched:
                return float(vals[0])
            return np.asarray(vals, float).reshape((self.B,) + (1,) * (pad + 1))

        self.v_ctrl_ub = ub
        # --- parameters, as floats or (B,1[,1]) arrays ----------------------
        self.k_a = pv("a")
        self.k_a_cong = pv("a_cong", fallback="a")
        self.k_quad_w = pv("quad_width")
        self.k_kappa = pv("kappa")
        self.k_chi = pv("chi")
        self.k_gscale = pv("grad_scale")
        self.k_acap = pv("antic_cap")
        self.k_wcong = pv("w_cong")
        self.k_rhojam = pv("rho_jam")
        self.k_vfm = pv("v_free_main")
        self.k_rc_main = pv("rho_crit")
        self.k_phi = None if p0.phi is None else pv("phi")
        rc_b = pv("rho_crit_bneck", fallback="rho_crit")
        k_vfb = pv("v_free_bneck")

        # --- per-segment arrays --------------------------------------------
        self.rc = np.where(IS_MAIN, self.k_rc_main, rc_b)
        self.v_free = np.where(IS_MAIN, self.k_vfm, k_vfb)
        self.V_crit_main = self.k_vfm * np.exp(-1.0 / self.k_a)
        self.seg_edge = SEG_EDGE
        # 11 segments -> 10 SUMO edges; ref_metanet reports segments 0..9, i.e.
        # the *first* bottleneck segment stands for edge 9. Matched for
        # comparability.
        self.pick = list(range(10))

        # --- hoisted constants ---------------------------------------------
        self._ns = max(1, int(p0.substeps))
        Ts = (p0.T_sec / 3600.0) / self._ns
        self._Ts = Ts
        tau = pv("tau_sec") / 3600.0
        self._T_over_tau = Ts / tau
        self._T_over_lamL = Ts / (LAM * LSEG)       # (NSEG,), T_sec is fixed
        self._T_over_L = Ts / LSEG                  # (NSEG,)
        self._antic_num = pv("eta") * Ts / tau
        self._q_cap = 2.0 * self.V_crit_main * self.k_rc_main
        self._inv_a = 1.0 / self.k_a
        self._u_cap = (1.0 + p0.vsl_alpha) * ub     # (N_MAIN,)
        if p0.veq_mode == "godunov":
            # triangular sending function: D = min(rho*v_free, Q), kink at rho_crit
            self._Q_lane = self.rc * self.v_free
        else:
            self._Q_lane = self.rc * self.v_free * np.exp(-1.0 / self.k_a)
        self._use_flux = bool(p0.supply or p0.veq_mode == "godunov")
        self._fd = p0.fd
        self._antic = p0.antic
        self._cap_demand = bool(p0.cap_demand)
        self._godunov = p0.veq_mode == "godunov"

    # -- fundamental diagram ----------------------------------------------
    def _veq(self, rho):
        a, rc = self.k_a, self.rc
        r0 = np.maximum(rho, 0.0)
        if self._fd == "exp":
            return self.v_free * np.exp((-1.0 / a) * np.power(r0 / rc, a))

        if self._fd == "exp2":
            # Two-branch exponential FD (Dominguez Frejo, Camacho & Horowitz,
            # 2012, IEEE CDC): a separate exponent below and above rho_crit,
            # the congested branch rescaled so the two meet at rho_crit and
            # capacity is unchanged.
            ac = self.k_a_cong
            r = r0 / rc
            free = self.v_free * np.exp((-1.0 / a) * np.power(r, a))
            cong = self.v_free * np.exp(-1.0 / a) * np.exp(
                (-1.0 / ac) * (np.power(r, ac) - 1.0))
            return np.where(rho <= rc, free, cong)

        if self._fd == "quad":
            # Quadrangular FD (Han thesis Eq. 4.7): a capacity plateau around
            # rho_crit, implemented by shifting the density towards rho_crit
            # inside the plateau band.
            hw = self.k_quad_w * rc
            shifted = np.where(r0 < rc - hw, r0,
                               np.where(r0 > rc + hw, r0 - 2.0 * hw, rc))
            return self.v_free * np.exp((-1.0 / a) * np.power(shifted / rc, a))

        raise ValueError(f"unknown fd {self._fd!r}")

    # -- state helpers ----------------------------------------------------
    def _shape(self, extra=()):
        lead = ((self.B,) if self.batched else ()) + tuple(extra)
        return lead

    def x0_empty(self, extra=()):
        lead = self._shape(extra)
        rho = np.zeros(lead + (NSEG,))
        v = np.broadcast_to(self.v_free, lead + (NSEG,)).copy()
        w = np.zeros(lead + (1,))
        return rho, v, w

    def from_edges(self, rho0, v0, w0):
        """(...,10) SUMO-edge state -> (...,NSEG) segment state."""
        rho0 = np.asarray(rho0, float)
        v0 = np.asarray(v0, float)
        rho = rho0[..., SEG_EDGE].copy()
        v = v0[..., SEG_EDGE].copy()
        w = np.asarray(w0, float).reshape(np.shape(rho0)[:-1] + (1,))
        return rho, v, w

    # -- the one-step update ----------------------------------------------
    def step(self, rho, v, w, d, v_ctrl=None):
        for _ in range(self._ns):
            rho, v, w = self._sub(rho, v, w, d, v_ctrl)
        return rho, v, w

    def _sub(self, rho, v, w, d, v_ctrl):
        lam, L, T = LAM, LSEG, self._Ts
        rc = self.rc

        # ---- flows (cell demand) ----------------------------------------
        q = rho * v * lam

        # ---- origin ------------------------------------------------------
        # SmoothMainstreamOrigin.get_flow with v_ctrl = 2*v_free (never binds).
        v_lim = v[..., 0:1]
        ratio = np.clip(v_lim / self.k_vfm, 0.05, 0.999)
        q_lim = np.where(
            v_lim < self.V_crit_main,
            2.0 * v_lim * self.k_rc_main
            * np.power(-self.k_a * np.log(ratio), self._inv_a),
            self._q_cap)
        q_o = np.minimum(d + w / T, q_lim)

        # ---- boundary values --------------------------------------------
        q_up = np.empty_like(q)
        v_up = np.empty_like(v)
        rho_dn = np.empty_like(rho)

        q_up[..., 0:1] = q_o
        q_up[..., 1:] = q[..., :-1]
        v_up[..., 0:1] = v[..., 0:1]        # Origin.get_speed = v of first seg
        v_up[..., 1:] = v[..., :-1]
        rho_dn[..., :-1] = rho[..., 1:]
        rho_dn[..., -1:] = np.minimum(rho[..., -1:], rc[..., -1:])  # free dest.

        # ---- STRUCTURAL SWITCH: CTM/Godunov supply bound -----------------
        # Plain METANET has no receiving-capacity constraint: a cell will happily
        # push rho*v*lam into a full downstream cell, which is why lowering eta
        # (the only thing damping that) produces densities of 200-550. Here the
        # damping job is given to an explicit supply term instead, so eta is free
        # to be small.
        #     S_i = lam_i * w_cong * (rho_jam - rho_i)      (CTM receiving flow)
        # Bounding the SPEED by S_down/(rho*lam) rather than clipping the flux
        # keeps q = rho*v*lam exactly consistent -- important because the bench
        # computes discharge from rho*v, not from an internal flux variable.
        S_dn = None
        if self._use_flux:
            S = lam * self.k_wcong * np.maximum(self.k_rhojam - rho, 0.0)
            S_dn = np.empty_like(S)
            S_dn[..., :-1] = S[..., 1:]
            S_dn[..., -1] = np.inf          # free outflow at the destination
            v = np.minimum(v, S_dn / np.maximum(rho * lam, 1e-9))
            q_o = np.minimum(q_o, S[..., 0:1])

        if self._cap_demand:
            # CTM/Godunov *demand* (sending) bound. Without it the anticipation
            # term hands a cell at rho=180 sitting next to an empty cell +28 km/h
            # in one 5 s step -- a discharge ~4x capacity, which is what
            # evaporates a queue into a uniform smear. The physical statement is
            # simply that a queue discharges at capacity, no faster.
            v = np.minimum(v, self._Q_lane / np.maximum(rho, 1e-9))

        if self._cap_demand or self._use_flux:
            q = rho * v * lam
            q_up[..., 0:1] = q_o
            q_up[..., 1:] = q[..., :-1]
            v_up[..., 1:] = v[..., :-1]

        # ---- density -----------------------------------------------------
        rho_next = rho + self._T_over_lamL * (q_up - q)

        # ---- equilibrium / target speed ---------------------------------
        if self._godunov:
            # Replace METANET's density-only equilibrium speed with the speed
            # implied by the Godunov flux, i.e. by the supply-demand pair rather
            # than by a single-valued FD:
            #     Veq_i = min( v_free_i , Q_i/rho_i , S_dn/(lam_i*rho_i) )
            # An oversaturated exit then discharges at capacity however dense it
            # is (Q/rho => flow == Q), and the queue forms wherever the SUPPLY
            # term binds -- the CTM answer, and what SUMO does -- instead of the
            # whole road sliding onto the congested branch at one uniform
            # (rho, v), which is METANET's flattening. Speed stays a state with
            # tau relaxation, so the model is still second order.
            r = np.maximum(rho, 1e-9)
            Veq = np.minimum(self.v_free, self._Q_lane / r)
            Veq = np.minimum(Veq, S_dn / (lam * r))
        else:
            Veq = self._veq(rho)
        u = self._u_cap if v_ctrl is None else (1.0 + self.p.vsl_alpha) \
            * np.asarray(v_ctrl, float)
        # Veq always has rho's full shape (rho appears in every branch above),
        # so the two halves concatenate cleanly.
        Veq = np.concatenate(
            [np.minimum(Veq[..., :N_MAIN], u), Veq[..., N_MAIN:]], axis=-1)

        # ---- speed -------------------------------------------------------
        relaxation = self._T_over_tau * (Veq - v)
        convection = self._T_over_L * v * (v_up - v)
        grad = rho_dn - rho
        base = self._antic_num / (L * (rho + self.k_kappa))

        if self._antic == "std":
            antic = base * grad
        elif self._antic == "asym":
            # The anticipation term is a diffusion. Its DECELERATING half
            # (rho_down > rho: drivers brake for the queue ahead) is the physical
            # mechanism and is what keeps the upstream front sharp. Its
            # ACCELERATING half (rho_down < rho) is what lets a dense cell at the
            # head of a queue leap to an unphysical speed -- rho=180 next to
            # rho=30 gives +28 km/h in ONE 5 s step, an outflow ~4x capacity --
            # and that is what dissolves the queue into a uniform smear. `chi`
            # weights that half only; chi=1 is stock METANET, chi=0 is
            # deceleration-only.
            antic = base * np.where(grad >= 0.0, grad, self.k_chi * grad)
        elif self._antic == "smooth":
            # Same idea, but the switch is a sigmoid in the gradient instead of a
            # discontinuity at grad == 0. weight -> 1 for grad >> 0
            # (decelerating), -> chi for grad << 0 (accelerating), blending over
            # `grad_scale` veh/km/lane. C-infinity, so IPOPT sees a smooth
            # Jacobian.
            wgt = self.k_chi + (1.0 - self.k_chi) * 0.5 * (
                1.0 + np.tanh(grad / self.k_gscale))
            antic = base * grad * wgt
        elif self._antic == "limited":
            # Slope limiter: left alone for small gradients (where it does its
            # stabilising job), saturating for large ones (where it would smear a
            # shock). Smooth, so IPOPT-friendly.
            raw = base * grad
            c = self.k_acap
            antic = c * np.tanh(raw / c)
        else:
            raise ValueError(f"unknown antic {self._antic!r}")

        v_next = v + relaxation + convection - antic

        if self.k_phi is not None:
            # METANET lane-drop term on the last 2-lane segment (2 -> 1 lanes)
            i = N_MAIN - 1
            v_next[..., i:i + 1] -= (
                self.k_phi * T * 1.0 * rho[..., i:i + 1] * v[..., i:i + 1] ** 2
            ) / (L[i] * lam[i] * rc[..., i:i + 1])

        rho_next = np.maximum(rho_next, 0.0)
        v_next = np.maximum(v_next, 0.0)
        w_next = np.maximum(w + T * (d - q_o), 0.0)
        return rho_next, v_next, w_next

    # -- runs --------------------------------------------------------------
    def run(self, demand, x0=None, v_ctrl=None):
        """Open loop. Returns (RHO, V, W) with time on the FIRST axis."""
        rho, v, w = self.x0_empty() if x0 is None else x0
        n = len(demand)
        RHO = np.empty((n,) + np.shape(rho))
        V = np.empty((n,) + np.shape(v))
        W = np.empty((n,) + np.shape(w))
        for k, d in enumerate(demand):
            rho, v, w = self.step(rho, v, w, float(d), v_ctrl)
            RHO[k], V[k], W[k] = rho, v, w
        return RHO, V, W


# --------------------------------------------------------------------------
# bench adapter
# --------------------------------------------------------------------------
class BenchModel:
    """Wraps `Model` in the experiments/bench.py contract."""

    def __init__(self, p: Optional[P] = None):
        self.m = Model(p)
        self.T = self.m.plist[0].T_sec

    def _resample(self, inflow_1s, n):
        idx = np.clip((np.arange(n) * self.T).astype(int), 0, len(inflow_1s) - 1)
        return np.asarray(inflow_1s, float)[idx]

    def open_loop(self, inflow_vph):
        n_sec = len(inflow_vph)
        n = int(n_sec / self.T)
        RHO, V, _ = self.m.run(self._resample(inflow_vph, n))
        up = np.clip((np.arange(n_sec) / self.T).astype(int), 0, n - 1)
        return {"v": V[:, self.m.pick][up], "rho": RHO[:, self.m.pick][up]}

    def predict(self, rho0, v0, w0, inflow_vph, horizon_s):
        rho, v, w = self.m.from_edges(rho0, v0, w0)
        n = int(horizon_s / self.T)
        d = self._resample(inflow_vph, n)
        for k in range(n):
            rho, v, w = self.m.step(rho, v, w, float(d[k]))
        return {"v": v[self.m.pick], "rho": rho[self.m.pick]}


# --------------------------------------------------------------------------
def predict_batch(model, rho0, v0, w0, dem, n_steps):
    """Roll many predictions forward together.

    rho0, v0 : (..., R, 10) SUMO-edge states at R start times
    w0       : (..., R)     insertion queue at each start time
    dem      : (n_steps, ..., R, 1) demand each rollout sees at each model step
    Returns  (rho, v), each (..., R, 10).
    """
    rho, v, w = model.from_edges(rho0, v0, w0)
    for k in range(n_steps):
        rho, v, w = model.step(rho, v, w, dem[k])
    return rho[..., model.pick], v[..., model.pick]
