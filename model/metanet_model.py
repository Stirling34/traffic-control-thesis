# -*- coding: utf-8 -*-
"""
METANET prediction model of the SUMO lane-merge scenario, built with the
`sym-metanet` library (CasADi engine).

This replaces the hand-rolled LWR model (lwr_v1/lwr_v2) as the MPC prediction
model. See docs/decisionmaking.md for the full rationale of every choice here.

Network structure (mirrors the SUMO net 1:1):

    O1 (MainstreamOrigin)
      -> N1 -> L1 = LinkWithVsl, 9 segments x 0.2 km, 2 lanes   (SUMO edges
               start1, n1n2 ... n8n9 — each segment has a VSL sign)
      -> N2 -> L2 = Link, 1 segment x 0.5 km, 1 lane            (SUMO edge n9end)
      -> N3 -> D1 (uncongested Destination = free outflow)

Units follow the METANET convention used by sym-metanet's examples:
  density veh/km/lane, speed km/h, length km, time hours (T = seconds/3600).
"""

# -- project root on sys.path so `from model...` etc. resolve when this
# -- file is run directly (python validation/foo.py)
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from dataclasses import dataclass, field

import casadi as cs
import numpy as np

import sym_metanet as metanet
from sym_metanet import (
    Destination,
    Link,
    LinkWithVsl,
    MainstreamOrigin,
    Network,
    Node,
    engines,
)

from sym_metanet.engines.casadi import Engine as _CasadiEngine
from sym_metanet.engines.casadi import LinksEngine as _LinksEngine

from model.ExtraData import ExtractData

import os as _o
PROJECT_ROOT = _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__)))
def _r(*p):
    """Path relative to the project root, so scripts run from anywhere."""
    return _o.path.join(PROJECT_ROOT, *p)

MS_TO_KMH = 3.6


class SmoothMainstreamOrigin(MainstreamOrigin[cs.SX]):
    """MainstreamOrigin with a numerically safe flow expression.

    The library clips the speed ratio v_lim/v_free to at most 1.0, but the
    flow term (-a*log(ratio))^(1/a) has an infinite derivative at ratio = 1
    (exponent 1/a - 1 < 0), so the NLP constraint Jacobian becomes NaN
    whenever a (predicted) first-segment speed equals free-flow speed. That
    is exactly the failure the old sitecustomize.py patch on the codex branch
    was fighting. Clipping at 0.999 instead changes nothing numerically —
    below V_crit (where this branch is actually selected) the ratio is ~0.6 —
    but keeps the gradient finite everywhere.

    Only `get_flow` is overridden; the METANET equations themselves are
    untouched (same formula as sym_metanet.engines.casadi.OriginsEngine).
    """

    def get_flow(self, net, T, engine=None, **_):
        from sym_metanet.engines.casadi import LinksEngine
        from sym_metanet.engines.core import get_current_engine

        if engine is None:
            engine = get_current_engine()
        link_down = self._get_exiting_link(net)
        d = self.disturbances["d"]
        w = self.states["w"]
        v_ctrl = self.actions["v_ctrl"]
        v_first = link_down.states["v"][0]
        rho_crit, a = link_down.rho_crit, link_down.a
        v_free, lanes = link_down.v_free, link_down.lam

        V_crit = LinksEngine.Veq(rho_crit, v_free, rho_crit, a)
        v_lim = cs.fmin(v_ctrl, v_first)
        ratio = cs.fmax(0.05, cs.fmin(0.999, v_lim / v_free))  # was fmin(1.0, .)
        q_speed = lanes * v_lim * rho_crit * cs.power(-a * cs.log(ratio), 1 / a)
        q_cap = lanes * V_crit * rho_crit
        q_lim = cs.if_else(v_lim < V_crit, q_speed, q_cap)
        return cs.fmin(d + w / T, q_lim)


class BoundedOrigin(MainstreamOrigin[cs.SX]):
    """Bounded-METANET's origin, their Eq. (9).

        q_o = min[ d + w/T,  C_o,  C_o (rho_max - rho_1) / (rho_max - rho_crit) ]

    The third branch is the one that matters and it is NOT optional. METANET's
    origin limits inflow by the first segment's SPEED; Bounded-METANET limits it
    by the first segment's DENSITY, throttling to zero as that segment fills.
    Bounded-METANET also calibrates a much lower rho_max (98.3 here against
    181.8), so without this branch the origin keeps injecting into a full
    segment: measured, the port ran away to a density of 3254 veh/km/lane
    against the original's 156 until this was added.

    C_o is calibrated separately in the paper rather than derived from the FD,
    because deriving it would tie inflow throttling to the free-flow speed.
    """

    __slots__ = ("C_o",)

    def __init__(self, *args, C_o=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.C_o = C_o

    def get_flow(self, net, T, engine=None, **_):
        link_down = self._get_exiting_link(net)
        d = self.disturbances["d"]
        w = self.states["w"]
        rho_1 = link_down.states["rho"][0]
        rho_max, rho_crit = link_down.rho_max, link_down.rho_crit
        C_o = self.C_o
        if C_o is None:
            C_o = float(link_down.lam) * rho_crit * link_down.v_free                 * float(np.exp(-1.0 / link_down.a))
        throttle = C_o * (rho_max - rho_1) / (rho_max - rho_crit)
        q = cs.fmin(d + w / T, C_o)
        return cs.fmax(cs.fmin(q, throttle), 0.0)


# ---------------------------------------------------------------------------
# Capacity drop (Han-style) grafted onto METANET
# ---------------------------------------------------------------------------
# Why this exists
# ---------------
# Plain METANET has no sustained capacity drop at a stationary bottleneck: after a
# forced breakdown its steady-state discharge returns to exactly the FD capacity
# (verified 2026-07-27). Consequences, all measured:
#   * a jam cannot detach from the bottleneck and march upstream leaving free road
#     behind, which SUMO plainly does -> ~47 km/h free-flow prediction error;
#   * preventing breakdown earns the optimiser nothing, so TTS-shaped cost
#     functions are control-invariant (control_authority.py).
#
# Han et al. (2016 TRR 2560:108-118; 2017 TR-C 77:405-420) fix this in a *first-
# order* model by making the maximum discharge of a cell a linear function of the
# density of the cell UPSTREAM of it:
#
#     Q_i = c_i * ( 1 - alpha * (rho_{i-1} - rho_crit) / (rho_jam - rho_crit) )
#
# Driving it from the upstream cell is the key idea: it lets the drop follow a
# *moving* jam instead of being welded to one bottleneck cell. (Roncoli et al.
# 2015 used the same cell's density; Han shows that under-estimates the drop.)
#
# We do NOT port Han's model. It is first order, so speed is slaved to density and
# acceleration is a property of the discretisation rather than of the traffic --
# which would make acceleration/jerk cost functions meaningless, and those are
# central to this project. Instead we transplant only the mechanism.
#
# In METANET, capacity is rho_crit * V(rho_crit), so scaling the equilibrium speed
# scales capacity by the same factor. Hence:
#
#     z_up  = clamp( (rho_up - rho_crit) / (rho_cd_ref - rho_crit), 0, 1 )
#     Veq_i = min( V_fd(rho_i) * (1 - alpha_cd * z_up) , (1 + alpha) * v_ctrl_i )
#
# The min() mirrors Han's S_i = min(rho*v, Q_i, rho*V_i): the capacity drop and the
# VSL are *independent* upper bounds, not multiplied. Applying the drop to the FD
# term only (before the VSL cap) keeps the VSL's control authority intact.
#
# Crucially the speed *relaxation* equation is untouched, so v remains an
# independent state with its tau dynamics and acceleration stays physically
# meaningful. That is the whole reason for doing it this way.
#
# alpha_cd = 0 disables the term and must reproduce the plain model bit-identically
# (see test_capacity_drop.py).

def smooth_min(x, y, eps):
    """C-infinity approximation of min(x, y); sits at most eps/2 BELOW the true
    minimum, and equals it wherever |x - y| >> eps.

    Needed wherever a min() lands on the optimiser's own operating point. The
    scaled VSL cap does exactly that: at the legal limit it equals v_free, and
    at low density V_fd is within a hair of v_free too, so a hard fmin gives
    IPOPT a kink precisely where it starts every solve. Measured: 27-52% solver
    failures with fmin, against 7-12% without the cap at all.
    """
    if eps <= 0.0:
        return cs.fmin(x, y)
    return 0.5 * (x + y - cs.sqrt((x - y) ** 2 + eps ** 2))


def smooth_clamp01(x, eps):
    """C-infinity approximation of clip(x, 0, 1); error <= eps/2, always interior.

    Built from the standard smooth max/min, max(a,b) ~ (a+b+sqrt((a-b)^2+eps^2))/2.
    A hard clip would give IPOPT a discontinuous gradient exactly at the onset of
    the capacity drop, which is precisely where the solver needs to be able to
    trade off control actions.
    """
    lo = 0.5 * (x + cs.sqrt(x * x + eps * eps))                     # ~ max(x, 0)
    return 0.5 * (lo + 1.0 - cs.sqrt((lo - 1.0) ** 2 + eps * eps))  # ~ min(., 1)


def _shifted_upstream_density(link, rho):
    """Density of the segment immediately UPSTREAM of each segment of `link`.

    Within a link that is just rho shifted by one. For the first segment the
    upstream neighbour lives in another link: `link.cd_upstream` supplies it (the
    last segment of the link feeding this one). If there is none -- i.e. this is
    the first link, fed by the mainstream origin, which has no density -- we fall
    back to the segment's own density. That makes exactly one segment behave like
    Roncoli's same-cell variant; it is documented rather than hidden.
    """
    up = getattr(link, "cd_upstream", None)
    rho_ext = up.states["rho"][-1] if up is not None else rho[0]
    return cs.vertcat(rho_ext, rho[:-1]) if link.N > 1 else rho_ext


def veq_triangular(rho, v_free, rho_jam, beta, eps_rho=1.0, eps=1.0):
    """Equilibrium speed of a TRIANGULAR fundamental diagram, smoothed.

        q(rho) = min( v_free * rho ,  beta * (rho_jam - rho) )
        v(rho) = q / rho = min( v_free , beta * (rho_jam - rho) / rho )

    Why bother: the exponential FD's congested branch falls off far too fast.
    It carries the ~2390 veh/h discharge only at rho ~ 77, whereas SUMO holds
    150-185 veh/km/lane in its queue. Since jam density sets how many vehicles a
    kilometre of road can store, that is why METANET's congestion has to spread
    over ~3x more road at high load (measured 2026-08-11) -- the dominant source
    of the free-flow prediction error. A triangular FD parameterises jam density,
    capacity and wave speed INDEPENDENTLY, so it can be dense and still flowing.

    Smoothing: `eps_rho` keeps the 1/rho finite at an empty road (where the
    min picks v_free anyway, so it changes nothing physical); `eps` smooths the
    min so IPOPT sees a continuous gradient at the FD kink.
    """
    cong = beta * (rho_jam - rho) / (rho + eps_rho)
    # smooth min(v_free, cong), then smooth max(., 0) for rho > rho_jam
    m = 0.5 * (v_free + cong - cs.sqrt((v_free - cong) ** 2 + eps * eps))
    return 0.5 * (m + cs.sqrt(m * m + eps * eps))


def _veq(link, engine, rho):
    """Equilibrium speed for `link`, dispatching on its fundamental-diagram type."""
    if getattr(link, "fd_type", "exponential") == "triangular":
        return veq_triangular(rho, link.v_free, link.fd_rho_jam, link.fd_beta)
    return engine.links.Veq(rho, link.v_free, link.rho_crit, link.a)


def _capacity_drop_scale(link, rho):
    """Multiplicative factor on the equilibrium speed, in (1 - alpha_cd, 1]."""
    rho_up = _shifted_upstream_density(link, rho)
    span = link.cd_rho_ref - link.rho_crit
    z = smooth_clamp01((rho_up - link.rho_crit) / span, link.cd_eps)
    return 1.0 - link.cd_alpha * z


class CapacityDropLink(Link[cs.SX]):
    """Plain (uncontrolled) METANET link with the Han-style capacity drop."""

    __slots__ = ("cd_alpha", "cd_rho_ref", "cd_eps", "cd_upstream",
                 "fd_type", "fd_rho_jam", "fd_beta")

    def __init__(self, *args, cd_alpha=0.0, cd_rho_ref=None, cd_eps=0.02,
                 cd_upstream=None, fd_type="exponential", fd_rho_jam=None,
                 fd_beta=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cd_alpha = float(cd_alpha)
        self.cd_rho_ref = float(cd_rho_ref) if cd_rho_ref is not None else self.rho_max
        self.cd_eps = float(cd_eps)
        self.cd_upstream = cd_upstream
        self.fd_type = fd_type
        self.fd_rho_jam = float(fd_rho_jam) if fd_rho_jam is not None else self.rho_max
        self.fd_beta = float(fd_beta) if fd_beta is not None else 15.0

    def _get_equilibrium_speed(self, engine, rho):
        Veq = _veq(self, engine, rho)
        if self.cd_alpha > 0.0:
            Veq = Veq * _capacity_drop_scale(self, rho)
        return Veq


class CapacityDropLinkWithVsl(LinkWithVsl[cs.SX]):
    """`LinkWithVsl` with the Han-style capacity drop applied before the VSL cap."""

    __slots__ = ("cd_alpha", "cd_rho_ref", "cd_eps", "cd_upstream",
                 "fd_type", "fd_rho_jam", "fd_beta", "vsl_scaled",
                 "vsl_legal_kmh", "vsl_clamp_eps")

    def __init__(self, *args, cd_alpha=0.0, cd_rho_ref=None, cd_eps=0.02,
                 cd_upstream=None, fd_type="exponential", fd_rho_jam=None,
                 fd_beta=None, vsl_scaled=False, vsl_legal_kmh=None,
                 vsl_clamp_eps=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.cd_alpha = float(cd_alpha)
        self.cd_rho_ref = float(cd_rho_ref) if cd_rho_ref is not None else self.rho_max
        self.cd_eps = float(cd_eps)
        self.cd_upstream = cd_upstream
        self.fd_type = fd_type
        self.fd_rho_jam = float(fd_rho_jam) if fd_rho_jam is not None else self.rho_max
        self.fd_beta = float(fd_beta) if fd_beta is not None else 15.0
        self.vsl_scaled = bool(vsl_scaled)
        self.vsl_clamp_eps = float(vsl_clamp_eps)
        nv = len(self.vsl)
        self.vsl_legal_kmh = (np.full(nv, self.v_free) if vsl_legal_kmh is None
                              else np.asarray(vsl_legal_kmh, float).reshape(nv))

    def _vsl_cap(self):
        """Speed ceiling the posted limit imposes. See MetanetParams.vsl_scaled."""
        u = self.actions["v_ctrl"]
        if getattr(self, "vsl_scaled", False):
            return (u / cs.DM(self.vsl_legal_kmh)) * self.v_free
        return (1.0 + self.alpha) * u

    def _get_equilibrium_speed(self, engine, rho):
        if (self.cd_alpha <= 0.0 and self.fd_type == "exponential"
                and not getattr(self, "vsl_scaled", False)):
            # identical expression graph to the stock class -> bit-identical results
            return super()._get_equilibrium_speed(engine, rho)
        Veq = _veq(self, engine, rho)
        if self.cd_alpha > 0.0:
            Veq = Veq * _capacity_drop_scale(self, rho)
        # VSL is an independent upper bound, exactly as in Han's min(..., rho*V_i)
        Veq[self.vsl] = smooth_min(Veq[self.vsl], self._vsl_cap(),
                                   self.vsl_clamp_eps if self.vsl_scaled else 0.0)
        return Veq


# ---------------------------------------------------------------------------
# Asymmetric anticipation (the bake-off winner), as a pluggable engine
# ---------------------------------------------------------------------------
# Why this exists
# ---------------
# METANET's anticipation term is a diffusion:
#
#     anticipation = (eta*T/tau) * (rho_down - rho) / (L * (rho + kappa))
#
# Its DECELERATING half (rho_down > rho, drivers braking for a queue ahead) is
# the physical mechanism and is what keeps the upstream front of a queue sharp.
# Its ACCELERATING half (rho_down < rho) is what lets the dense cell at the HEAD
# of a queue leap to an unphysical speed: measured in `experiments/exp4`, a cell
# at rho=180 sitting next to one at rho=30 gains +28 km/h in a single 5 s step,
# an outflow about 4x capacity. That is what dissolves a queue into a uniform
# smear, and the smear is why stock METANET's 210 s prediction correlates 0.035
# with SUMO while simply assuming nothing changes correlates 0.324.
#
# `chi` weights the accelerating half only. chi = 1 is stock METANET; chi = 0 is
# deceleration-only. The switch is a sigmoid in the density gradient rather than
# a hard `where(grad >= 0, ...)`, so the Jacobian IPOPT sees stays smooth - the
# hard version has a kink at grad == 0 sitting right in the operating region.
#
# This is the `s14_smooth_tau18` model from `experiments/expA_asym_antic`, which
# scored held-out 210 s correlation 0.863 against stock METANET's 0.046 on the
# shared benchmark. Engines are sym-metanet's own extension point, so this
# subclasses rather than patches; `MetanetParams.antic="smooth"` selects it.

class SmoothAnticLinksEngine(_LinksEngine):
    """`LinksEngine` whose anticipation term is asymmetric and smooth."""

    chi: float = 1.0          # weight on the accelerating half
    grad_scale: float = 5.0   # sigmoid width, veh/km/lane

    @staticmethod
    def step_speed(v, v_up, rho, rho_down, Veq, lanes, L, tau, eta, kappa, T,
                   q_ramp=None, delta=None, lanes_drop=None, phi=None,
                   rho_crit=None):
        cls = SmoothAnticLinksEngine
        relaxation = (T / tau) * (Veq - v)
        convection = T * v / L * (v_up - v)
        grad = rho_down - rho
        # weight -> 1 for grad >> 0 (decelerating, keep it), -> chi for
        # grad << 0 (accelerating, damp it), blending over `grad_scale`
        wgt = cls.chi + (1.0 - cls.chi) * 0.5 * (
            1.0 + cs.tanh(grad / cls.grad_scale))
        anticipation = (eta * T / tau) * grad * wgt / (L * (rho + kappa))
        v_next = v + relaxation + convection - anticipation
        if q_ramp is not None and delta is not None:
            v_next[0] -= (delta * T * q_ramp * v[0]) / (L * lanes * (rho[0] + kappa))
        if lanes_drop is not None and phi is not None and rho_crit is not None:
            v_next[-1] -= (phi * T * lanes_drop * rho[-1] * v[-1] ** 2) / (
                L * lanes * rho_crit)
        return v_next


class SmoothAnticEngine(_CasadiEngine):
    """CasADi engine that uses the asymmetric-anticipation link equations."""

    @property
    def links(self):
        return SmoothAnticLinksEngine


# ---------------------------------------------------------------------------
# Bounded-METANET (Zhao, Roncoli & Yildirimoglu 2025, TR-C; arXiv:2503.17680)
# ---------------------------------------------------------------------------
# The published alternative to METANET's speed equation. METANET computes
#
#     v+ = v + (T/tau)(V[rho] - v) + convection - anticipation
#
# and the anticipation term is UNBOUNDED: with an empty road ahead it can push a
# segment's speed far above any equilibrium, which is the defect that makes the
# shipped model's 210 s forecast worthless. Bounded-METANET deletes convection,
# anticipation and the separate merge/lane-drop speed penalties outright, and
# folds the look-ahead into a VIRTUAL DENSITY fed to the equilibrium-speed
# function instead (their Eqs. 13-14):
#
#     rho_hat_i = rho_i + eta_t * (kap_t / (rho_i + kap_t)) * (rho_{i+1} - rho_i)
#     v_i+      = v_i + (T/tau) * ( V[rho_hat_i] - v_i )
#
# rho_hat is a convex combination of rho_i and rho_{i+1}, so the speed target can
# never exceed V[min(rho_i, rho_{i+1})] and the speed is bounded by construction.
# That is the whole point of the paper and the reason the model is stable where
# METANET is not.
#
# This is the published, citable model. It scored 0.784 held-out at 210 s in the
# project's own bake-off against the shipped METANET's 0.046. The asymmetric
# anticipation variant (SmoothAnticLinksEngine above) scored higher, 0.863, but
# it is a bespoke modification; Bounded-METANET can be cited.
#
# Parameters below are `pub_pred` from experiments/expB_bounded, the best of that
# arm. Note phi_t = 0: the lane-drop term of their Eq. (16) calibrated to zero on
# this network, which is unsurprising - the paper's own two case studies are
# on-ramp merges and it lists lane drops as untested.

class BoundedMetanetLink(LinkWithVsl[cs.SX]):
    """Link whose speed follows Bounded-METANET Eqs. (13)-(14).

    `step_dynamics` is overridden rather than the engine's `step_speed`, because
    the change is structural: the terms sym-metanet's `step_speed` expects
    (convection, anticipation) do not exist in this model at all.
    """

    __slots__ = ("eta_t", "kap_t", "down_bc", "vsl_scaled", "vsl_legal_kmh",
                 "is_last_link", "vsl_clamp_eps",
                 "cd_alpha", "cd_rho_ref", "cd_eps", "cd_upstream")

    def __init__(self, *args, eta_t=0.7, kap_t=1733.0, down_bc="zero",
                 vsl_scaled=False, vsl_legal_kmh=None, is_last_link=False,
                 vsl_clamp_eps=0.25, cd_alpha=0.0, cd_rho_ref=None,
                 cd_eps=0.02, cd_upstream=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_last_link = bool(is_last_link)
        # capacity drop; alpha 0 makes `_capacity_drop_scale` exactly 1.0 and
        # the term is skipped entirely in `_bounded_step`, so the default
        # reproduces the pre-2026-09-01 model bit-identically.
        self.cd_alpha = float(cd_alpha)
        self.cd_rho_ref = (float(cd_rho_ref) if cd_rho_ref is not None
                           else 2.0 * float(self.rho_crit))
        self.cd_eps = float(cd_eps)
        self.cd_upstream = cd_upstream
        self.vsl_clamp_eps = float(vsl_clamp_eps)
        self.eta_t = float(eta_t)
        self.kap_t = float(kap_t)
        self.down_bc = str(down_bc)
        self.vsl_scaled = bool(vsl_scaled)
        n = len(self.vsl)
        self.vsl_legal_kmh = (np.full(n, self.v_free) if vsl_legal_kmh is None
                              else np.asarray(vsl_legal_kmh, float).reshape(n))

    def _vsl_cap(self):
        """Speed ceiling the posted limit imposes. See MetanetParams.vsl_scaled."""
        u = self.actions["v_ctrl"]
        if self.vsl_scaled:
            return (u / cs.DM(self.vsl_legal_kmh)) * self.v_free
        return (1.0 + self.alpha) * u

    step_dynamics = None   # set below to _bounded_step; see that function


def _bounded_step(self, net, tau, eta, kappa, T, delta=None, phi=None,
                  engine=None, positive_next_speed=True,
                  positive_next_density=False, **_):
        """Bounded-METANET Eqs. (4), (13), (14). Shared by both link classes."""
        from sym_metanet.engines.core import get_current_engine
        if engine is None:
            engine = get_current_engine()

        node_up, node_down = net.nodes_by_link[self]
        rho = self.states["rho"]
        v = self.states["v"]
        q = self.get_flow(engine)
        v0, q0 = node_up.get_upstream_speed_and_flow(net, self, engine, T=T)
        rhoN_1 = node_down.get_downstream_density(net, engine)

        if self.N > 1:
            q_up = engine.vcat(q0, q[:-1])
            rho_down = engine.vcat(rho[1:], rhoN_1)
        else:
            q_up = q0
            rho_down = rhoN_1
        if self.down_bc == "zero" and getattr(self, "is_last_link", False):
            # Their Eq. (14) needs rho_{N+1} and the paper never states it. The
            # bake-off swept the three candidates and this arm calibrated to a
            # free exit, which is also what our uncongested Destination means.
            #
            # It applies ONLY to the final segment of the whole road. The numpy
            # original models the road as one 11-segment array so this is
            # automatic there; here it is two links, and applying it to L1's
            # last segment as well tells that segment the road ahead is empty.
            # It then accelerates and dumps flow into L2: measured, density ran
            # away to 3367 veh/km/lane against the original's 156.
            rho_down = (engine.vcat(rho[1:], cs.SX.zeros(1)) if self.N > 1
                        else cs.SX.zeros(1))

        # (4) conservation, unchanged from METANET
        rho_next = rho + (T / (self.L * self.lam)) * (q_up - q)

        # (14) virtual density: a convex combination of this segment and the next
        rho_hat = rho + self.eta_t * (self.kap_t / (rho + self.kap_t)) \
            * (rho_down - rho)

        # (13) relaxation onto the equilibrium speed of the VIRTUAL density.
        # No convection, no anticipation, no lane-drop speed penalty.
        Veq = engine.links.Veq(cs.fmax(rho_hat, 0.0), self.v_free,
                               self.rho_crit, self.a)
        # CAPACITY DROP. Added 2026-09-01. Before this, `_capacity_drop_scale`
        # was called only from CapacityDropLink / CapacityDropLinkWithVsl /
        # VslRateLink -- i.e. the shipped and expA model families -- and NEVER
        # from here, so on Bounded-METANET the `alpha_cd` parameter was a silent
        # no-op: rollouts at alpha_cd = 0 / 0.118 / 0.25 were bit-identical.
        # Every Bounded-METANET result in the project was therefore produced
        # through a model with no capacity-drop mechanism at all, which matters
        # because METANET's own drop is EMERGENT FROM THE ANTICIPATION TERM
        # (Hegyi 2004 sec. 3.3; Zhao, Roncoli & Yildirimoglu 2025 sec. 2.2:
        # "the anticipation term is a primary contributor to the capacity drop
        # phenomenon within the METANET model") and Bounded-METANET deletes
        # anticipation. Its authors concede it "underestimates the severity of
        # the initial flow drop".
        #
        # Hegyi (2004) sec. 5.1.1 makes this a necessary condition for the
        # controller to work at all: "the controller model should be able to
        # reproduce the capacity drop... Otherwise, the controller will not find
        # the control signals that can eliminate the capacity drop."
        #
        # `alpha_cd` defaults to 0.0 and `_capacity_drop_scale` returns exactly
        # 1.0 there, so this line is a no-op unless deliberately enabled --
        # verified by validation/test_bounded_parity.py, which must stay at
        # ~1e-13 against the numpy original.
        if getattr(self, "cd_alpha", 0.0):
            Veq = Veq * _capacity_drop_scale(self, rho)
        if getattr(self, "vsl", None):
            Veq[self.vsl] = smooth_min(Veq[self.vsl], self._vsl_cap(),
                                       self.vsl_clamp_eps
                                       if self.vsl_scaled else 0.0)
        v_next = v + (T / tau) * (Veq - v)

        if positive_next_speed:
            v_next = cs.fmax(v_next, 0.0)
        if positive_next_density:
            rho_next = cs.fmax(rho_next, 0.0)
        return {"rho": rho_next, "v": v_next}


BoundedMetanetLink.step_dynamics = _bounded_step


class BoundedMetanetPlainLink(Link[cs.SX]):
    """Bounded-METANET link with no VSL signs. Used for the bottleneck link."""

    __slots__ = ("eta_t", "kap_t", "down_bc", "is_last_link",
                 "cd_alpha", "cd_rho_ref", "cd_eps", "cd_upstream")

    def __init__(self, *args, eta_t=0.7, kap_t=1733.0, down_bc="zero",
                 is_last_link=True, cd_alpha=0.0, cd_rho_ref=None,
                 cd_eps=0.02, cd_upstream=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_last_link = bool(is_last_link)
        self.eta_t = float(eta_t)
        self.kap_t = float(kap_t)
        self.down_bc = str(down_bc)
        # This is the BOTTLENECK link, so this is where the merge capacity drop
        # lives: `cd_upstream` is L1, whose last segment supplies the upstream
        # density that triggers the drop. alpha 0 = disabled = bit-identical to
        # the pre-2026-09-01 model.
        self.cd_alpha = float(cd_alpha)
        self.cd_rho_ref = (float(cd_rho_ref) if cd_rho_ref is not None
                           else 2.0 * float(self.rho_crit))
        self.cd_eps = float(cd_eps)
        self.cd_upstream = cd_upstream

    step_dynamics = _bounded_step


class VslRateLink(CapacityDropLinkWithVsl):
    """VSL as a *rate* that reshapes the fundamental diagram, not as a cap.

    Carlson, Papamichail and Papageorgiou's formulation, as written out in
    Papamichail et al., IFAC World Congress 2008, eqs. (4)-(7). The control is
    the VSL rate b = displayed limit / legal limit, in [b_min, 1]:

        V(rho, b) = v_f(b) * exp{ -(1/a(b)) * ( rho / rho_cr(b) )^a(b) }
        v_f(b)    = b * v_f*
        rho_cr(b) = rho_cr* * ( 1 + A * (1 - b) )
        a(b)      = a*      * ( E - (E - 1) * b )

    Two things follow, and both matter here.

    First, b appears in every term rather than behind a min(), so dV/db is
    non-zero at every density. The flat region that makes the cap formulation
    inert on ~99% of segment-steps simply does not exist in this form.

    Second, with A > 0 a lower limit moves the critical density to the right.
    That is the mechanism by which a VSL is actually supposed to help: more
    vehicles get through before breakdown. The cap formulation leaves the
    critical density where it was, so it cannot represent that mechanism at all,
    only the loss of speed. A is therefore doing real work and must be measured
    for this network, not borrowed - calibration/vsl_fd.py fits it from SUMO.

    The action is still carried in km/h so the rate limit, the bounds and the
    logging all keep their existing meaning; b is formed internally by dividing
    by the legal limit of each segment.
    """

    __slots__ = ("vsl_A", "vsl_E", "vsl_legal_kmh", "vsl_one_way",
                 "vsl_clamp_eps")

    def __init__(self, *args, vsl_A=0.0, vsl_E=1.0, vsl_legal_kmh=None,
                 vsl_one_way=True, vsl_clamp_eps=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.vsl_A = float(vsl_A)
        self.vsl_E = float(vsl_E)
        self.vsl_one_way = bool(vsl_one_way)
        self.vsl_clamp_eps = float(vsl_clamp_eps)
        n = len(self.vsl)
        if vsl_legal_kmh is None:
            vsl_legal_kmh = np.full(n, self.v_free)
        self.vsl_legal_kmh = np.asarray(vsl_legal_kmh, dtype=float).reshape(n)

    def _get_equilibrium_speed(self, engine, rho):
        Veq = _veq(self, engine, rho)
        if self.cd_alpha > 0.0:
            Veq = Veq * _capacity_drop_scale(self, rho)

        b = self.actions["v_ctrl"] / cs.DM(self.vsl_legal_kmh)
        v_f = b * self.v_free
        rho_cr = self.rho_crit * (1.0 + self.vsl_A * (1.0 - b))
        a = self.a * (self.vsl_E - (self.vsl_E - 1.0) * b)
        # rho is clamped away from zero only inside the power: 0^a is fine for
        # a > 0 but its derivative is not, and IPOPT will evaluate it there.
        r = cs.fmax(rho[self.vsl], 1e-6) / rho_cr
        Veq_vsl = v_f * cs.exp(-(1.0 / a) * cs.power(r, a))

        if self.vsl_one_way and self.vsl_clamp_eps > 0.0:
            # Smooth minimum, not cs.fmin. The two curves TOUCH at b = 1, which
            # is where the controller starts every solve, so a hard min puts a
            # non-differentiable kink exactly on the operating point: measured
            # 98 solver failures in 482 solves against 30 in 459 without the
            # clamp. This form is differentiable everywhere and converges to the
            # true min as eps -> 0.
            #
            # It is not exactly one-way. The smoothing pulls hardest where the
            # two curves are closest, which is at b = 1, so a lowered sign can
            # come out fractionally faster than no sign at all. That residual is
            # bounded by eps/2 and measured at **+0.07 km/h** over the whole
            # (rho, u) grid at the default eps = 0.25, against **+6.0 km/h** for
            # the unclamped curve. Treat it as a rounding error on the physics,
            # not as a mechanism: it is far below the ~0.5 km/h step the rate
            # limit moves the sign in, so it cannot pay for an action.
            x, y = Veq_vsl, Veq[self.vsl]
            e = self.vsl_clamp_eps
            Veq_vsl = 0.5 * (x + y - cs.sqrt((x - y) ** 2 + e ** 2))
        elif self.vsl_one_way:
            # A sign can only ever slow traffic relative to no sign at all.
            # Without this clamp the raised rho_cr(b) lifts the curve ABOVE the
            # uncontrolled one in the congested branch, and the model concludes
            # that posting a lower number makes queued traffic move faster -
            # measured at 20.6 km/h under a 50 sign against 14.6 under an 88
            # sign at rho = 80. That is not a mechanism, it is the optimiser
            # being handed free speed. Clamping leaves the control live in the
            # undercritical branch, which is where a VSL genuinely acts: slow
            # the approach flow before the bottleneck, do not try to
            # re-accelerate a queue that has already formed.
            Veq_vsl = cs.fmin(Veq_vsl, Veq[self.vsl])

        if self.cd_alpha > 0.0:
            # the capacity drop is a property of the road, not of the sign, so
            # it multiplies the reshaped curve as well
            Veq_vsl = Veq_vsl * _capacity_drop_scale(self, rho)[self.vsl]
        Veq[self.vsl] = Veq_vsl
        return Veq


# ---------------------------------------------------------------------------
# Geometry parsed from the SUMO network
# ---------------------------------------------------------------------------
@dataclass
class Geometry:
    """SUMO geometry relevant to the METANET model (ordered upstream->downstream)."""

    edge_ids: list          # all 10 edge ids
    lengths_m: list         # per-edge length (m)
    lanes: list             # per-edge lane count
    limits_ms: list         # per-edge speed limit (m/s)
    edges_raw: list = field(repr=False, default=None)  # ExtractData dicts (for plotting)

    @property
    def n_main(self):
        """Number of 2-lane (controlled) edges — the METANET L1 segments."""
        return sum(1 for l in self.lanes if l == 2)

    @property
    def main_limits_kmh(self):
        return [v * MS_TO_KMH for v, l in zip(self.limits_ms, self.lanes) if l == 2]


def load_geometry(net_file=None, allow_straight=False):
    """`allow_straight=True` also accepts a road with the same lane count end to
    end and no bottleneck (sumo/build_straight_net.py). Such a geometry must be
    modelled with `MetanetModel(..., n_ctrl=...)`, never the default merge
    layout, which would silently treat its last edge as a lane drop."""
    if net_file is None:
        net_file = _r("sumo", "net", "plain.net2.xml")
    edges = ExtractData().parse_net_file(net_file)
    geom = Geometry(
        edge_ids=[e["edge_id"] for e in edges],
        lengths_m=[e["lanes"][0]["length"] for e in edges],
        lanes=[e["num_lanes"] for e in edges],
        limits_ms=[e["avg_speed"] for e in edges],
        edges_raw=edges,
    )
    straight = allow_straight and all(l == geom.lanes[0] for l in geom.lanes)
    # Sanity: this model hard-assumes "N two-lane edges then one single-lane edge"
    assert straight or (geom.lanes[-1] == 1 and all(l == 2 for l in geom.lanes[:-1])), (
        "Network layout changed — METANET model assumes 2-lane edges followed by "
        f"a single 1-lane bottleneck edge, got lanes={geom.lanes}"
    )
    return geom


def model_geometry(geom, n_ctrl=9):
    """The sub-stretch the prediction model spans: the LAST `n_ctrl` two-lane
    edges plus the one-lane bottleneck.

    On the extended-approach network (sumo/build_long_net.py) the measured road
    is 5.4 km but the controller's authority is still the nine segments in front
    of the bottleneck. Slicing the geometry here — rather than widening the
    model — is what keeps the two scenarios comparable: the NLP has the same
    11 segments and ~900 variables it always had, so the calibration, the parity
    tests and the validated 210 s prediction skill all carry over untouched, and
    the ONLY difference between the short and long runs is how much road is
    scored.

    Returns `(sub_geometry, offset)`, where `offset` is the index of the first
    modelled edge in the FULL edge list. On the original network offset is 0 and
    the returned geometry is the input unchanged, so nothing about the existing
    results moves.
    """
    n_main = geom.n_main
    if n_ctrl > n_main:
        raise ValueError("asked for %d controlled edges, geometry has %d"
                         % (n_ctrl, n_main))
    offset = n_main - n_ctrl
    if offset == 0:
        return geom, 0
    keep = slice(offset, len(geom.edge_ids))
    sub = Geometry(
        edge_ids=geom.edge_ids[keep],
        lengths_m=geom.lengths_m[keep],
        lanes=geom.lanes[keep],
        limits_ms=geom.limits_ms[keep],
        edges_raw=geom.edges_raw[keep] if geom.edges_raw else None,
    )
    return sub, offset


def straight_model_geometry(geom, n_up, n_seg=None):
    """The modelled stretch of the STRAIGHT jam-wave road: `n_seg` edges from
    edge `n_up` (default: everything to the end), i.e. the controlled segments
    plus the downstream segments where the disturbance happens. The first `n_up`
    edges are the measured but unmodelled approach.

    `n_seg` exists for the long-downstream variant (sumo/build_straight_net.py,
    N_DOWN_LONG), which adds road for a TRAVELLING disturbance to run into. That
    extra road is scored but deliberately NOT modelled, so the NLP, the
    calibration and the controller problem stay identical to every earlier
    straight-road round and the results remain comparable.

    Returns `(sub_geometry, offset)` like `model_geometry`; build the model with
    `MetanetModel(sub, params, n_ctrl=...)`."""
    end = len(geom.edge_ids) if n_seg is None else int(n_up) + int(n_seg)
    keep = slice(int(n_up), min(end, len(geom.edge_ids)))
    sub = Geometry(
        edge_ids=geom.edge_ids[keep],
        lengths_m=geom.lengths_m[keep],
        lanes=geom.lanes[keep],
        limits_ms=geom.limits_ms[keep],
        edges_raw=geom.edges_raw[keep] if geom.edges_raw else None,
    )
    return sub, int(n_up)


# ---------------------------------------------------------------------------
# METANET parameters
# ---------------------------------------------------------------------------
# The bake-off winner, calibrated: experiments/expA_asym_antic, tag
# s14_smooth_tau18, held-out 210 s correlation 0.863 against stock METANET's
# 0.046 on the shared benchmark (experiments/bench.py). Parity-tested against
# the original numpy implementation to 5e-13 by validation/test_antic_parity.py.
# Bounded-METANET (Zhao, Roncoli & Yildirimoglu 2025, TR-C), calibrated:
# experiments/expB_bounded tag `pub_pred`, the best of that arm at held-out 210 s
# correlation 0.784 against the shipped METANET's 0.046. This is the PUBLISHED
# and citable model. phi_t calibrated to 0, so their lane-drop term (Eq. 16) is
# off; the paper's own case studies are on-ramp merges and it lists lane drops
# as untested.
PARAMS_BM = dict(
    family="bounded",
    v_free_main=74.0, v_free_bneck=85.15822388597344,
    rho_crit=35.44689130754422, a=2.8947959929480875,
    tau_sec=48.81938730643131,
    eta_t=0.7006427004748599, kap_t=1733.3224564846118,
    rho_max=98.29531195944853, down_bc="zero",
    C_o=4074.7466695823164,
)

# Bounded-METANET refitted to the STRAIGHT jam-wave road
# (calibration/calibrate_straight.py, 2026-09-15; data/params_straight.json).
# PARAMS_BM was fitted on the merge, where the mainline never passed ~1750
# veh/h/lane and discharge is set by the 1-lane bottleneck; this road carries
# ~2350 veh/h/lane. Fitted on two runs (4000, 4600 veh/h), held out on two
# (4600 seed 2, 4800): held-out objective 1.92 -> 1.46, 210 s prediction RRMSE
# 0.25-0.32 -> 0.18-0.24, 2-lane capacity 3714 -> 5900 veh/h.
#
# TWO HONEST CAVEATS.
# 1. The fit is NOT identifiable in (rho_crit, a): three fits with rho_crit 70,
#    91 and 120 and a 1.42, 1.10 and 0.94 score 1.4647, 1.4636 and 1.4673 on
#    held-out data. This set is the one whose capacity sits inside the plant's
#    measured band, which is why it was chosen -- not because it fits best.
# 2. IT DOES NOT FIX JAM DISCHARGE, which is what motivated the recalibration.
#    810 -> 1092 veh/h against SUMO's ~4700, and the model still peaks at ~47
#    veh/km/lane where SUMO's jam reaches ~180. That gap is structural for this
#    model family (ledger 2026-08-11: the ~2x queue-density gap is irreducible),
#    so a controller on this road still believes jams drain ~4x too slowly.
PARAMS_BM_STRAIGHT = dict(
    family="bounded",
    v_free_main=80.0, v_free_bneck=80.0,
    rho_crit=91.2643, a=1.1043,
    tau_sec=46.5904,
    eta_t=0.3437, kap_t=1995.16,
    rho_max=229.532, down_bc="zero",
    C_o=4866.05,
)

# Bounded-METANET calibrated on JAM CLEARANCE (calibration/calibrate_jam.py,
# 2026-10-05, seed 0; data/params_jam_s0.json). Objective = calibrate_straight's
# (RRMSE + 210 s prediction + penalties) PLUS clearance of SUMO-seeded jams
# released at 840/900/960 s. Fit cases clear like SUMO (305 vs 282 s from
# 840 s on d4600_s1); held-out jam cell-seconds within 7-40 % but clearance
# still slower than SUMO (> 600 vs 468-554 s). General prediction is unchanged
# from PARAMS_BM_STRAIGHT (held-out 1.466 both).
# CAVEAT: it gets there with tau 78 s (bound 80) and capacity on its 4700 floor,
# i.e. a jam that drains slowly because capacity barely exceeds inflow; its
# jam discharge (3586 veh/h peak at the jam boundary) is still below SUMO's
# ~4700. Variant B (`_dis`) adds a discharge term to remove that compensation.
PARAMS_BM_JAM = dict(
    family="bounded",
    v_free_main=70.93440200848505, v_free_bneck=70.93440200848505,
    rho_crit=53.9535662978059, a=2.050126441663475,
    tau_sec=78.01997072924554,
    eta_t=0.7027773188323065, kap_t=1036.4586259396597,
    rho_max=169.14232342811985, down_bc="zero",
    C_o=6427.215768450603,
)

# Variant B of the jam-clearance fit (calibrate_jam.py with JAM_W_DIS=1, seed 0;
# data/params_jam_dis_s0.json): the same objective PLUS |log(peak jam discharge
# / 4700)|, so clearance cannot be bought with slow drivers and a capacity that
# barely exceeds inflow (fit A's compensation). Result is physically coherent:
# tau 24 s, discharge 4687 veh/h peak at the jam boundary (SUMO ~4700); held-out
# clearance error 0.336 (fit A 0.537, PARAMS_BM 1.09, PARAMS_BM_STRAIGHT 4.95)
# -- d4600_s2 clears in 400/375/340 s vs SUMO 468/408/348; 210 s prediction
# RRMSE 0.21-0.22 held out (fit A 0.24-0.25); whole-episode held-out objective
# 1.548 vs 1.466. Capacity 4631 veh/h sits just under the 4700 band (penalty
# 0.03), i.e. barely above the 4600 arriving -- note it with any result.
PARAMS_BM_JAM_DIS = dict(
    family="bounded",
    v_free_main=71.3415489963018, v_free_bneck=71.3415489963018,
    rho_crit=47.30898441981669, a=2.6541893027350927,
    tau_sec=24.117104352224107,
    eta_t=0.9601576990778027, kap_t=296.1253618590499,
    rho_max=118.7126809697613, down_bc="zero",
    C_o=4554.431780459563,
)

PARAMS_S14 = dict(
    antic="smooth",
    a=2.682064482545289, chi=0.2580522734629326, eta=16.052450492089307,
    grad_scale=1.845872344655433, kappa=36.21988278516994,
    rho_crit=35.144326917533064, rho_crit_bneck=49.400604510855196,
    v_free_bneck=113.26497556989747, v_free_main=81.4966930734148,
    tau_sec=18.0,
)


@dataclass
class MetanetParams:
    """METANET model parameters (literature-standard defaults, then calibrated).

    Defaults are the Hegyi (2004) / sym-metanet example values, except where
    tied to SUMO physically:
      - rho_max from SUMO vehicle length 5 m + minGap 0.5 m -> 1000/5.5 veh/km/lane
      - v_free per link from the SUMO edge speed limits
    """

    T_sec: float = 5.0            # model step (s); must satisfy T*v_free < 0.2 km
    rho_max: float = 1000.0 / 5.5  # jam density (veh/km/lane) ~ 182
    # rho_crit calibrated 2026-07-21 so the 1-lane link capacity matches the
    # SUMO merge discharge (~2390 veh/h): rho_crit = q / (v_free * e^(-1/a))
    rho_crit: float = 40.5        # critical density (veh/km/lane)
    a: float = 1.867              # FD exponent
    tau_sec: float = 18.0         # speed relaxation time (s)
    # eta lowered from the textbook 60: with 200 m segments the anticipation
    # term (~eta/L) is 2.5-5x stronger than in the ~500-1000 m segments it was
    # tuned for, and eta=60 produces severe numerical stop-go stripes here.
    eta: float = 30.0             # anticipation (km^2/h)
    kappa: float = 40.0           # anticipation regularisation (veh/km/lane)
    vsl_alpha: float = 0.1        # VSL non-compliance factor (SUMO speedFactor dev ~0.1)
    v_free_main: float = None     # free-flow speed of 2-lane link (km/h); from SUMO if None
    v_free_bneck: float = None    # free-flow speed of bottleneck link (km/h); from SUMO if None
    n_bneck_segs: int = 2         # segments in the 1-lane bottleneck link (>=2 keeps the
                                  # free-outflow boundary from masking the bottleneck)
    phi: float = None             # METANET lane-drop speed-penalty coefficient
                                  # (None = term disabled; applied to last L1 segment)
    # -- Han-style capacity drop (see the block above the Link subclasses) ------
    alpha_cd: float = 0.0         # max extent of the drop, 0 = disabled (= plain
                                  # METANET, bit-identical). Han's calibrated
                                  # values: 0.35 synthetic/A13, 0.66 A12 field.
                                  # Empirical: ~10-15% standing queue, ~30% jam wave.
    rho_cd_ref: float = None      # density at which the drop is maximal. None =
                                  # rho_max, which is Han-faithful. BUT our
                                  # congested branch only reaches ~77-80
                                  # veh/km/lane (the documented ~2x-low queue
                                  # density), so normalising by rho_max=182 means
                                  # z never exceeds ~0.26 and only a quarter of
                                  # alpha_cd is ever realised. Set this to the
                                  # density the congested branch actually reaches
                                  # if you want alpha_cd to mean what Han means.
    cd_eps: float = 0.02          # smoothing width of the clamp, in units of z
    # -- fundamental diagram family -------------------------------------------
    fd_type: str = "exponential"  # "exponential" (METANET standard) or
                                  # "triangular" (Han/CTM). Triangular decouples
                                  # jam density from capacity, so the queue can be
                                  # dense AND still flowing -- see veq_triangular.
    fd_rho_jam: float = None      # triangular only; None = rho_max
    fd_beta: float = None         # triangular only; congestion wave speed (km/h).
                                  # None -> derived from capacity and rho_jam.
    # -- how the VSL enters the model -----------------------------------------
    vsl_mode: str = "cap"         # "cap"  = Hegyi (2005): Veq = min(V_fd(rho),
                                  #          (1+alpha)*v_ctrl). The control is an
                                  #          upper bound, so above the kink it
                                  #          vanishes from the expression and
                                  #          dJ/du is exactly zero. Measured in
                                  #          closed loop, the limit is inert on
                                  #          ~99% of segment-steps.
                                  # "rate" = Carlson / Papamichail / Papageorgiou:
                                  #          the VSL rate b = displayed/legal
                                  #          reshapes the whole speed-density
                                  #          curve, so b appears in every term
                                  #          and the gradient is non-zero
                                  #          everywhere. See VslRateLink.
    vsl_A: float = 0.0            # "rate" only. How much a lower limit raises the
                                  # critical density: rho_cr(b) = rho_cr*(1+A(1-b)).
                                  # 0 = no shift. MEASURE it, do not import it:
                                  # calibration/vsl_fd.py fits it for this network.
    vsl_E: float = 1.0            # "rate" only. Curve sharpness: a(b) = a*(E-(E-1)b).
                                  # 1 = the exponent does not move with b.
    vsl_b_min: float = 0.25       # "rate" only. Lowest admissible rate.
    vsl_scaled: bool = False      # HOW THE POSTED LIMIT BECOMES A SPEED CEILING.
                                  # False: the shipped form, cap = (1+alpha)*u
                                  #   with alpha = +0.1, i.e. drivers do 1.10x the
                                  #   sign. MEASURED IN SUMO THEY DO 0.89x, so the
                                  #   cap is 24% too permissive and never binds:
                                  #   with v_free 81.5 (expA) or 74.0 (bounded)
                                  #   the FD term always wins and the sign is
                                  #   inert on >99% of segment-steps.
                                  # True:  cap = (u / u_legal) * v_free, i.e. the
                                  #   posted limit scales the free-flow speed in
                                  #   proportion. At u = the legal limit the cap
                                  #   equals v_free exactly, so no control means
                                  #   no change; below it the cap binds
                                  #   IMMEDIATELY, at every segment, with no dead
                                  #   zone. It is one-way by construction and
                                  #   stays inert in a queue, where V_fd is
                                  #   already below the cap and no sign can
                                  #   re-accelerate traffic. Unlike the Carlson
                                  #   rate form it has no free coefficient to fit,
                                  #   so none of that form's fitting problems.
    # -- model family ---------------------------------------------------------
    family: str = "metanet"       # "metanet" = METANET speed equation (with the
                                  #   `antic` variants above).
                                  # "bounded" = Bounded-METANET: convection,
                                  #   anticipation and the speed penalties are
                                  #   deleted and folded into a virtual density.
                                  #   See BoundedMetanetLink. Use PARAMS_BM.
    eta_t: float = 0.70           # "bounded": look-ahead weight, Eq. (14), [0,1]
    kap_t: float = 1733.0         # "bounded": look-ahead smoothing, veh/km/lane
    C_o: float = None             # "bounded": origin capacity, their Eq. (9).
                                  # None derives it from the mainline FD.
    down_bc: str = "zero"         # "bounded": the downstream density their
                                  # Eq. (14) needs and the paper never states.
                                  # "zero" = free exit; calibrated choice.
    # -- anticipation term ----------------------------------------------------
    antic: str = "std"            # "std"    = stock METANET.
                                  # "smooth" = asymmetric, sigmoid-weighted; the
                                  #            bake-off winner s14_smooth_tau18,
                                  #            held-out 210 s corr 0.863 against
                                  #            stock METANET's 0.046. See
                                  #            SmoothAnticLinksEngine.
    chi: float = 1.0              # "smooth": weight on the ACCELERATING half of
                                  # the anticipation diffusion. 1 = stock.
    grad_scale: float = 5.0       # "smooth": sigmoid width, veh/km/lane.
    rho_crit_bneck: float = None  # separate critical density (hence capacity)
                                  # for the 1-lane bottleneck link. None = same
                                  # as rho_crit. Discharge is set by the
                                  # bottleneck, so it needs its own freedom.
    vsl_one_way: bool = True      # "rate" only. A speed limit is a ONE-WAY
                                  # actuator: it can slow traffic down, it can
                                  # never make traffic go faster than it would
                                  # have without the sign. The raw Carlson curve
                                  # violates this above ~40 veh/km/lane, where
                                  # raising rho_cr(b) lifts the curve above the
                                  # uncontrolled one and the model starts
                                  # believing a lower sign speeds congested
                                  # traffic up. Clamping to the uncontrolled
                                  # curve restores the physics and leaves the
                                  # gradient live exactly where a sign really
                                  # does bite: undercritical traffic, upstream
                                  # of a bottleneck, before it breaks down.
                                  # Set False only to reproduce the unclamped
                                  # form for comparison.
    vsl_clamp_eps: float = 0.25   # "rate" + one_way only. Smoothing width of the
                                  # clamp, in km/h. The two curves touch at b=1,
                                  # which is the controller's starting point, so
                                  # a hard min puts a kink on the operating point
                                  # and IPOPT fails on it. 0 = hard cs.fmin.

    @property
    def T(self):
        return self.T_sec / 3600.0

    @property
    def tau(self):
        return self.tau_sec / 3600.0

    def capacity_per_lane(self, v_free):
        """METANET capacity per lane (veh/h): rho_crit * Veq(rho_crit)."""
        return self.rho_crit * v_free * np.exp(-1.0 / self.a)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
class MetanetModel:
    """Builds the sym-metanet network and compiles the one-step dynamics F.

    F (compact=2, more_out=True):
        F(x, u, d) -> (x+, q)
      x = [rho (n_seg); v (n_seg); w (1)]   (veh/km/lane, km/h, veh)
      u = v_ctrl (n_ctrl)                   VSL per L1 segment (km/h)
      d = demand (1)                        origin demand (veh/h)
      q = [link flows (n_seg); origin flow] (veh/h)
    """

    def __init__(self, geom: Geometry, params: MetanetParams = None,
                 n_ctrl: int = None):
        """`n_ctrl=None` builds the merge: every 2-lane edge carries a sign and
        the 1-lane last edge is split into `n_bneck_segs` bottleneck segments.

        `n_ctrl=int` builds a STRAIGHT road (sumo/build_straight_net.py): the
        first `n_ctrl` edges carry signs and the rest form an unsigned
        downstream link with the SAME lane count, free-flow speed and critical
        density, one segment per edge. It is the same road, so
        `v_free_bneck`/`rho_crit_bneck` are ignored there rather than letting
        merge-calibrated bottleneck values leak onto plain road."""
        self.geom = geom
        self.p = params if params is not None else MetanetParams()
        p = self.p
        self.straight = n_ctrl is not None

        if self.straight:
            n_main = int(n_ctrl)
            n_bneck = len(geom.edge_ids) - n_main
            if n_bneck < 1 or any(l != geom.lanes[0] for l in geom.lanes):
                raise ValueError("straight model needs >=1 downstream edge and "
                                 "one lane count; got n_ctrl=%d, lanes=%s"
                                 % (n_main, geom.lanes))
            if p.phi is not None:
                raise ValueError("phi is a lane-drop term; a straight road has none")
            L_main = float(np.mean(geom.lengths_m[:n_main])) / 1000.0
            L_bneck = float(np.mean(geom.lengths_m[n_main:])) / 1000.0
            lanes_main = lanes_down = int(geom.lanes[0])
            legal_kmh = np.array(geom.limits_ms[:n_main]) * MS_TO_KMH
        else:
            n_main = geom.n_main                     # 9 controlled segments
            # Uniform segment length for L1 = mean of the 2-lane edge lengths (196-200 m)
            L_main = float(np.mean(geom.lengths_m[:n_main])) / 1000.0
            n_bneck = max(1, int(p.n_bneck_segs))
            L_bneck = geom.lengths_m[-1] / 1000.0 / n_bneck
            lanes_main, lanes_down = 2, 1
            legal_kmh = np.array(geom.main_limits_kmh)

        if p.v_free_main is None:
            # scalar per link (library expects scalars — vectors break the origin
            # flow equations; see docs/decisionmaking.md): mean of the edge limits
            p.v_free_main = float(np.mean(legal_kmh))
        if self.straight:
            v_free_down, rho_crit_down = p.v_free_main, p.rho_crit
        else:
            if p.v_free_bneck is None:
                p.v_free_bneck = geom.limits_ms[-1] * MS_TO_KMH
            v_free_down = p.v_free_bneck
            rho_crit_down = (p.rho_crit if p.rho_crit_bneck is None
                             else p.rho_crit_bneck)

        # CFL-style stability check for the density update
        assert p.T * max(p.v_free_main, v_free_down) < min(L_main, L_bneck), (
            f"Model step T={p.T_sec}s too large for segment length "
            f"{min(L_main, L_bneck)*1000:.0f} m"
        )

        if p.family == "bounded":
            # Bounded-METANET overrides step_dynamics wholesale, so the engine's
            # step_speed is never called; the stock engine is correct here.
            engines.use("casadi", sym_type="SX")
        elif p.antic == "smooth":
            # Engines are sym-metanet's own extension point. Set the two
            # coefficients on the class before installing it, since step_speed
            # is a staticmethod and takes no extra arguments.
            SmoothAnticLinksEngine.chi = float(p.chi)
            SmoothAnticLinksEngine.grad_scale = float(p.grad_scale)
            engines.use(SmoothAnticEngine(sym_type="SX"))
        else:
            engines.use("casadi", sym_type="SX")

        N1, N2, N3 = Node(name="N1"), Node(name="N2"), Node(name="N3")
        O1 = (BoundedOrigin(name="O1", C_o=p.C_o) if p.family == "bounded"
              else SmoothMainstreamOrigin(name="O1"))
        D1 = Destination[cs.SX](name="D1")
        rho_jam = p.fd_rho_jam if p.fd_rho_jam is not None else p.rho_max
        cd = dict(cd_alpha=p.alpha_cd, cd_rho_ref=p.rho_cd_ref, cd_eps=p.cd_eps,
                  fd_type=p.fd_type, fd_rho_jam=rho_jam)
        if p.family == "bounded":
            vsl_cls = BoundedMetanetLink
            vsl_kw = dict(eta_t=p.eta_t, kap_t=p.kap_t, down_bc=p.down_bc,
                          is_last_link=False,   # L2 lies downstream of L1
                          # capacity drop, added 2026-09-01. L1 is fed by the
                          # mainstream origin, which carries no density, so its
                          # first segment falls back to its own (documented in
                          # `_shifted_upstream_density`). alpha_cd defaults to 0,
                          # so this changes nothing unless enabled.
                          cd_alpha=p.alpha_cd, cd_rho_ref=p.rho_cd_ref,
                          cd_eps=p.cd_eps, cd_upstream=None)
        else:
            vsl_cls = VslRateLink if p.vsl_mode == "rate" else CapacityDropLinkWithVsl
            vsl_kw = (dict(vsl_A=p.vsl_A, vsl_E=p.vsl_E,
                           vsl_one_way=p.vsl_one_way,
                           vsl_clamp_eps=p.vsl_clamp_eps)
                      if p.vsl_mode == "rate" else {})
        vsl_kw.update(vsl_scaled=p.vsl_scaled,
                      vsl_clamp_eps=p.vsl_clamp_eps,
                      vsl_legal_kmh=legal_kmh)
        if p.family != "bounded":
            # capacity-drop and FD-family kwargs exist only on the METANET links
            vsl_kw.update(
                cd_upstream=None,   # fed by the origin, which has no density
                fd_beta=(p.fd_beta if p.fd_beta is not None
                         else p.capacity_per_lane(p.v_free_main)
                         / (rho_jam - p.capacity_per_lane(p.v_free_main)
                            / p.v_free_main)),
                **cd)
        L1 = vsl_cls(
            n_main, lanes_main, L_main, p.rho_max, p.rho_crit, p.v_free_main, p.a,
            segments_with_vsl=set(range(n_main)), alpha=p.vsl_alpha, name="L1",
            **vsl_kw,
        )
        bneck_cls = (BoundedMetanetPlainLink if p.family == "bounded"
                     else CapacityDropLink)
        bneck_kw = (dict(eta_t=p.eta_t, kap_t=p.kap_t, down_bc=p.down_bc,
                         # THE MERGE CAPACITY DROP. `cd_upstream=L1` means the
                         # drop at the bottleneck is triggered by the density of
                         # L1's last segment, i.e. by the queue backing up into
                         # the merge -- which is the mechanism a VSL is supposed
                         # to prevent. This is the channel that was missing.
                         cd_alpha=p.alpha_cd, cd_rho_ref=p.rho_cd_ref,
                         cd_eps=p.cd_eps, cd_upstream=L1)
                    if p.family == "bounded" else dict(
                        cd_upstream=L1, fd_beta=(
                            p.fd_beta if p.fd_beta is not None
                            else p.capacity_per_lane(v_free_down)
                            / (rho_jam - p.capacity_per_lane(v_free_down)
                               / v_free_down)), **cd))
        # On the merge L2 is the 1-lane bottleneck; on a straight road it is the
        # unsigned downstream stretch of the same road (see __init__ docstring).
        L2 = bneck_cls(
            n_bneck, lanes_down, L_bneck, p.rho_max, rho_crit_down,
            v_free_down, p.a, name="L2", **bneck_kw
        )
        net = Network(name="merge").add_path(
            origin=O1, path=(N1, L1, N2, L2, N3), destination=D1
        )
        net.is_valid(raises=True)

        # Fix the origin's own speed sign to "no control" (2*v_free never binds),
        # so it disappears from F's inputs — the VSLs on L1 are the only controls.
        step_kwargs = dict(
            T=p.T,
            tau=p.tau,
            eta=p.eta,
            kappa=p.kappa,
            init_conditions={O1: {"v_ctrl": p.v_free_main * 2.0}},
            positive_next_speed=True,
            positive_next_density=True,
            positive_next_queue=True,
        )
        if p.phi is not None:
            # lane-drop speed penalty on the last L1 segment (2 lanes -> 1 lane)
            step_kwargs["phi"] = p.phi
        net.step(**step_kwargs)
        self.F: cs.Function = metanet.engine.to_function(
            net=net, more_out=True, compact=2, T=p.T
        )

        self.net = net
        self.L1, self.L2, self.O1 = L1, L2, O1
        self.n_seg = n_main + n_bneck
        self.n_ctrl = n_main
        self.seg_lengths_km = [L_main] * n_main + [L_bneck] * n_bneck
        self.seg_lanes = [lanes_main] * n_main + [lanes_down] * n_bneck
        # SUMO edge index each METANET segment maps to (bottleneck segments all
        # belong to the last SUMO edge; on a straight road one segment per edge)
        self.seg_edge_index = (list(range(n_main + n_bneck)) if self.straight
                               else list(range(n_main)) + [n_main] * n_bneck)
        # per-segment VSL upper bound = the SUMO edge's own legal limit (km/h)
        self.v_ctrl_ub = np.asarray(legal_kmh, float)
        # per-segment free-flow speed, for callers that must stay below it
        self.seg_v_free = [p.v_free_main] * n_main + [v_free_down] * n_bneck

    # -- helpers -----------------------------------------------------------
    @property
    def origin_flow_cap(self):
        """Hard ceiling on the flow the mainstream origin can ever inject (veh/h).

        sym-metanet's mainstream origin (see SmoothMainstreamOrigin.get_flow and
        `sym_metanet.engines.casadi.OriginsEngine.get_mainstream_flow`) computes

            q_origin = min(d + w/T,  q_lim)

        where `q_lim` is a function of the first link's state and parameters
        ONLY — it does not contain `d`. So once `d + w/T` exceeds `q_lim`, the
        min() is pinned to `q_lim` and the simulation becomes *completely*
        insensitive to demand: the road state is bit-identical for d = 5000 and
        d = 800000, and the entire surplus accumulates in the origin queue `w`.

        This returns the free-flow value of `q_lim` (its maximum, reached when
        the first segment is uncongested), i.e. the demand above which nothing
        on the road can change. Note the *instantaneous* cap is lower whenever
        the first segment is congested (v_lim < V_crit selects the `q_speed`
        branch instead).
        """
        lanes = self.seg_lanes[0]
        V_crit = self.p.v_free_main * np.exp(-1.0 / self.p.a)
        return lanes * V_crit * self.p.rho_crit * 1.0

    def x0_empty(self):
        """Initial state: empty road at free flow."""
        rho = np.zeros(self.n_seg)
        v = np.array(self.seg_v_free, float)
        w = np.zeros(1)
        return np.concatenate([rho, v, w])

    def pack_x(self, rho, v, w):
        return np.concatenate([np.asarray(rho).ravel(),
                               np.asarray(v).ravel(),
                               np.atleast_1d(np.asarray(w).ravel())])

    def unpack_x(self, x):
        x = np.asarray(x).ravel()
        n = self.n_seg
        return x[:n], x[n : 2 * n], x[2 * n :]

    def check_demand_sensitivity(self, demand_vph, verbose=True):
        """Warn if the requested demand is above the origin's flow ceiling.

        Above `origin_flow_cap` the model is provably insensitive to demand
        (see that property), which otherwise fails *silently* — the plots just
        stop responding to the demand you set. Returns the saturated fraction.
        """
        d = np.atleast_1d(np.asarray(demand_vph, dtype=float))
        cap = self.origin_flow_cap
        frac = float(np.mean(d > cap))
        if frac > 0 and verbose:
            print(
                f"\n  !! DEMAND SATURATES THE ORIGIN in {frac*100:.0f}% of steps.\n"
                f"     demand max = {d.max():.0f} veh/h  vs  origin cap = {cap:.0f} veh/h\n"
                f"     Above the cap the road state is IDENTICAL for any demand;\n"
                f"     only the origin queue w grows. Lower the demand below\n"
                f"     {cap:.0f} veh/h to get a demand-responsive simulation.\n"
            )
        return frac

    def simulate(self, demand_vph, x0=None, v_ctrl=None, check_demand=True):
        """Open-loop simulation.

        demand_vph : array of origin demand (veh/h), one entry per model step.
        v_ctrl     : fixed VSL vector (km/h) or None for "no control" (limits).
        check_demand : print a warning if demand exceeds `origin_flow_cap`.
        Returns dict of arrays over time: rho (T,n_seg), v, w, q.
        """
        if check_demand:
            self.check_demand_sensitivity(demand_vph)
        x = self.x0_empty() if x0 is None else np.asarray(x0, dtype=float)
        u = self.v_ctrl_ub if v_ctrl is None else np.asarray(v_ctrl, dtype=float)
        RHO, V, W, Q = [], [], [], []
        for d in demand_vph:
            x_next, q = self.F(x, u, float(d))
            x = np.asarray(x_next).ravel()
            rho, v, w = self.unpack_x(x)
            RHO.append(rho)
            V.append(v)
            W.append(w)
            Q.append(np.asarray(q).ravel())
        return {
            "rho": np.array(RHO), "v": np.array(V),
            "w": np.array(W), "q": np.array(Q),
        }


if __name__ == "__main__":
    # Smoke test: build the model and run 1000 s of the SUMO demand (4000 veh/h)
    geom = load_geometry()
    model = MetanetModel(geom)
    p = model.p
    print(f"Segments: {model.n_seg} (controlled: {model.n_ctrl})")
    print(f"v_free main/bneck: {p.v_free_main:.1f}/{p.v_free_bneck:.1f} km/h")
    print(f"Capacity per lane (main): {p.capacity_per_lane(p.v_free_main):.0f} veh/h")
    print(f"Bottleneck capacity (1 lane): {p.capacity_per_lane(p.v_free_bneck):.0f} veh/h")
    n_steps = int(1000 / p.T_sec)
    res = model.simulate(np.full(n_steps, 4000.0))
    print(f"\nAfter {n_steps} steps of T={p.T_sec}s:")
    print("  final rho (veh/km/lane):", np.round(res["rho"][-1], 1))
    print("  final v   (km/h)       :", np.round(res["v"][-1], 1))
    print("  final queue w (veh)    :", np.round(res["w"][-1], 1))
    print("  max rho over run       :", np.round(res["rho"].max(axis=0), 1))
