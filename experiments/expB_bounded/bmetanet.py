# -*- coding: utf-8 -*-
"""
Bounded-METANET for the SUMO lane-drop merge, implemented in plain numpy.

Zhao, W., Roncoli, C., Yildirimoglu, M. (2025), "Bounded-METANET: a new
discrete-time second-order macroscopic traffic flow model for bounded speed",
Transportation Research Part C.  arXiv:2503.17680.

--------------------------------------------------------------------------
EQUATIONS -- TRANSCRIBED FROM https://arxiv.org/html/2503.17680v1 (2026-08-16)
--------------------------------------------------------------------------
Standard METANET part that Bounded-METANET keeps:

  (4)  rho_{m,i}(k+1) = rho_{m,i}(k)
                        + T/(L_m lam_m) [ q_{m,i-1}(k) - q_{m,i}(k)
                                          + q_o(k) - q_s(k) ]
  (6)  q_{m,i}(k) = rho_{m,i}(k) v_{m,i}(k)
  (7)  V[rho_{m,i}(k)] = v_free,m exp{ -(1/a_m) (rho_{m,i}(k)/rho_crit,m)^{a_m} }
  (8)  w_o(k+1) = w_o(k) + T [ d_o(k) - q_o(k) ]
  (9)  q_o(k) = min[ d_o(k) + w_o(k)/T,
                     C_o,
                     C_o (rho_max - rho_{m,i}(k)) / (rho_max - rho_crit,m) ]

What Bounded-METANET REPLACES.  METANET's speed equation had relaxation +
convection + anticipation, plus separate merging (delta) and lane-drop (phi)
speed penalties.  All of convection, anticipation and the two penalties are
deleted and folded into a "virtual density" fed to the equilibrium-speed
function:

  (13) v_{m,i}(k+1) = v_{m,i}(k) + (T/tau) ( V[rho_hat_{m,i}(k)] - v_{m,i}(k) )

  (14) rho_hat_{m,i}(k) = rho_{m,i}(k)
        + eta_t ( kap_t / (rho_{m,i}(k) + kap_t) ) [ rho_{m,i+1}(k) - rho_{m,i}(k) ]

  (15) on-ramp merging (NOT USED HERE -- our network has no on-ramp):
       rho_til_{m,i} = rho_hat_{m,i}
         + del_t (kap_t/(rho_{m,i}+kap_t)) (v_{m,i}/v_max) (q_o/C_o)
           [ rho_max - rho_hat_{m,i} ]

  (16) LANE DROP  <-- our geometry:
       rho_til_{m,i}(k) = rho_hat_{m,i}(k)
         + phi_t (dlam/lam_m) ( rho_{m,N_m}(k)/rho_max ) ( v_{m,i}(k)/v_max )
           [ rho_max - rho_hat_{m,i}(k) ]

  (17) combined merge+lane-drop = half the sum of the two bracketed
       coefficients, sharing the common (v_{m,i}/v_max)[rho_max - rho_hat]
       factor.  (Not used: no on-ramp.)

Stated ranges / conditions (paper):
  0 <= alpha := T/tau <= 1  (so tau >= T);  0 <= eta_t <= 1;  kap_t > 0;
  0 <= del_t <= 1;  0 <= phi_t <= 1;  CFL: a vehicle must not cross more than
  one segment per step (T v_free <= L).
Calibration bounds used by the paper (their Table): eta_t [0,1],
kap_t [1,300] veh/km/lane, del_t [0,1], tau [5,80] s, v_free [110,200] km/h,
rho_crit [15,40], rho_max [110,150], a [0.5,2.5].  Their two calibrated sets:
eta_t 0.52 / 0.92, kap_t 149.9 / 299.4, tau 12.35 / 80.0, v_free 118.5 / 138.6,
rho_crit 34.4 / 22.2, rho_max 149.9 / 149.4, a 1.81 / 1.92.  phi_t is never
calibrated in the paper -- both their case studies are on-ramp merges, and
their own future-work list names lane-drop bottlenecks as untested.

--------------------------------------------------------------------------
READING DECISIONS WHERE THE PAPER IS SILENT OR AMBIGUOUS  (flagged honestly)
--------------------------------------------------------------------------
A. LANES IN THE FLOW EQUATION.  Eq. (6) as printed is q = rho v, with rho in
   veh/km/lane.  For a homogeneous link the lam_m in Eq. (4)'s denominator
   cancels whichever convention is used, so the paper never has to be explicit.
   It does NOT cancel across our 2-lane -> 1-lane node.  We therefore use total
   flow q = rho v lam and divide by (L lam), which is the standard METANET
   convention and the only one that conserves vehicles at the lane drop.
B. NODE BETWEEN THE TWO LINKS.  Total flow is conserved across the 2->1 node;
   the per-lane density of the first bottleneck segment is used as
   rho_{m,i+1} for the last mainline segment.  Standard METANET rule for a node
   with a single exiting link.
C. DOWNSTREAM BOUNDARY.  Eq. (14) needs rho_{N+1} for the final segment of the
   whole road; the paper never states it.  Three options are implemented and
   swept: "crit" (min(rho_N, rho_crit) -- the METANET uncongested-destination
   rule sym-metanet uses), "neumann" (rho_{N+1}=rho_N) and "zero".
D. SCOPE OF (16).  The free index i in Eq. (16) alongside the fixed index N_m
   implies the term applies to more than one segment.  `lanedrop_scope="link"`
   applies it to every segment of the link that loses a lane; "last" applies it
   only to the segment immediately before the drop (where METANET's own Eq. 11
   lane-drop penalty sits).  Both are calibrated.
E. BOUNDEDNESS GUARDS.  Eq. (16)'s coefficient is a convex weight only while
   rho_{m,N_m} <= rho_max and rho_hat <= rho_max.  The exponential FD has no
   finite jam density, so density CAN exceed rho_max.  We clip
   rho_{m,N_m}/rho_max and v/v_max to [0,1] and floor [rho_max - rho_hat] at 0,
   which preserves the paper's stated convexity without changing the equation
   anywhere the paper's own assumptions hold.

--------------------------------------------------------------------------
KNOWN STRUCTURAL BEHAVIOUR OF THIS MODEL ON AN OVERSATURATED LANE DROP
--------------------------------------------------------------------------
Bounded-METANET's only speed forcing is V[rho_hat], and rho_hat is a CONVEX
COMBINATION of rho_i and rho_{i+1}, so segment i's speed target can never
exceed V[min(rho_i, rho_{i+1})].  Past rho_crit the exponential FD's flow
rho*V(rho) FALLS steeply, so if the bottleneck segment's density overshoots
far enough its own discharge collapses and it ratchets to a dead stop that
nothing downstream can relieve.  METANET does not have this failure mode
because its anticipation term -(eta T/(tau L))(rho_down - rho)/(rho + kappa) is
UNBOUNDED and pushes speed far ABOVE the local equilibrium when the road ahead
is empty (the same unboundedness that exp4 diagnosed as METANET's own defect).
Two levers control the ratchet, and both are inside the model's own parameter
space: eta_t -> 1 with large kap_t (so an empty downstream is fully seen), and
an FD whose congested branch decays slowly.  Hence `fd=` below.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Geometry: identical to bench.py's, so the mapping to SUMO edges is exact.
# ---------------------------------------------------------------------------
EDGE_LEN_M = np.array([200., 200., 200., 200., 200., 200., 200., 200., 196., 496.])
EDGE_LANES = np.array([2., 2., 2., 2., 2., 2., 2., 2., 2., 1.])
N_MAIN_EDGES = 9


class BoundedMetanet:
    """Bounded-METANET on the 9x~200 m two-lane + 496 m one-lane lane-drop road.

    Parameters (all keyword):
      v_free_main, v_free_bneck : km/h, per link
      rho_crit, a               : exponential FD (Eq. 7)
      a_cong                    : congested-branch exponent for fd="exp2"
      rho_jam_fd                : jam density for fd="tri"
      fd                        : "exp"  -- Eq. (7), the published FD
                                  "exp2" -- two-branch exponential, separate
                                            exponent above/below rho_crit,
                                            continuous at the join (Dominguez
                                            Frejo, Camacho & Horowitz 2012)
                                  "tri"  -- triangular / Newell-Daganzo
      tau_sec                   : relaxation time, s   (Eq. 13); needs tau >= T
      eta_t, kap_t              : virtual-density anticipation (Eq. 14)
      phi_t                     : lane-drop weight (Eq. 16); 0 disables
      rho_max                   : jam density used by Eq. 16 and the origin Eq. 9
      T_sec                     : model step, s
      n_bneck                   : segments the 496 m bottleneck is split into
      lanedrop_scope            : "link" (all mainline segments) or "last"
      down_bc                   : downstream boundary for Eq. 14 at the last
                                  segment: "crit" | "neumann" | "zero"
      convection                : ABLATION ONLY. If >0, adds METANET's convection
                                  term (T/L) v_i (v_{i-1} - v_i) back on top of
                                  Eq. 13, scaled by this factor.  Not part of
                                  Bounded-METANET.
      origin_throttle           : include the third branch of Eq. 9
      alpha_vsl                 : VSL compliance factor (see `set_vsl`)
    """

    def __init__(self, v_free_main=95.0, v_free_bneck=105.0, rho_crit=40.0,
                 a=1.8, a_cong=1.8, rho_jam_fd=185.0, fd="exp",
                 tau_sec=18.0, eta_t=0.5, kap_t=100.0, phi_t=0.0,
                 rho_max=150.0, T_sec=5.0, n_bneck=2, lanedrop_scope="link",
                 convection=0.0, origin_throttle=True, down_bc="crit",
                 alpha_vsl=0.1, C_o=None, lengths_m=None, lanes=None):
        self.down_bc = down_bc
        self.T_sec = float(T_sec)
        self.T = self.T_sec / 3600.0
        self.n_bneck = int(n_bneck)

        # -- segment grid ---------------------------------------------------
        # `lengths_m`/`lanes` give an explicit geometry, one segment per SUMO
        # edge -- added 2026-09-15 for the straight jam-wave road, which is
        # homogeneous and has no lane drop. Omit both and the grid is exactly the
        # merge road it always was (parity test unchanged).
        self._one_to_one = lengths_m is not None
        if self._one_to_one:
            self.L = np.asarray(lengths_m, float) / 1000.0
            self.lam = np.asarray(lanes, float)
            self.n = len(self.L)
            self.seg_edge = np.arange(self.n)
            self.is_main = self.lam >= (self.lam.max() - 1e-9)
            self.i_last_main = int(np.nonzero(self.is_main)[0][-1])
        else:
            L_m = list(EDGE_LEN_M[:N_MAIN_EDGES] / 1000.0)
            L_b = [EDGE_LEN_M[-1] / 1000.0 / self.n_bneck] * self.n_bneck
            self.L = np.array(L_m + L_b)                   # km
            self.lam = np.array([2.0] * N_MAIN_EDGES + [1.0] * self.n_bneck)
            self.n = len(self.L)
            self.seg_edge = np.array(list(range(N_MAIN_EDGES))
                                     + [N_MAIN_EDGES] * self.n_bneck)
            self.is_main = self.seg_edge < N_MAIN_EDGES
            self.i_last_main = N_MAIN_EDGES - 1            # N_m of the 2-lane link

        # -- parameters -----------------------------------------------------
        self.v_free = np.where(self.is_main, v_free_main, v_free_bneck)
        self.v_max_link = self.v_free.copy()               # v_max in Eq. 16
        self.rho_crit = float(rho_crit)
        self.a = float(a)
        self.a_cong = float(a_cong)
        self.rho_jam_fd = float(rho_jam_fd)
        self.fd = fd
        self.tau_sec = float(tau_sec)
        self.alpha = self.T_sec / self.tau_sec             # T/tau, must be <= 1
        self.eta_t = float(eta_t)
        self.kap_t = float(kap_t)
        self.phi_t = float(phi_t)
        self.rho_max = float(rho_max)
        self.lanedrop_scope = lanedrop_scope
        self.convection = float(convection)
        self.origin_throttle = bool(origin_throttle)
        self.alpha_vsl = float(alpha_vsl)
        self.v_ctrl = None                                 # no VSL by default

        # lane-drop mask for Eq. 16
        if phi_t > 0.0:
            if lanedrop_scope == "last":
                m = np.zeros(self.n, bool)
                m[self.i_last_main] = True
            else:
                m = self.is_main.copy()
            self.ld_mask = m
        else:
            self.ld_mask = np.zeros(self.n, bool)
        # 1 lane dropped of 2 on the merge road; nothing dropped on a road with
        # a uniform lane count, which makes Eq. (16) inert there as it should be
        self.dlam_over_lam = 0.0 if (self._one_to_one
                                     and np.ptp(self.lam) == 0) else 1.0 / 2.0

        # origin capacity C_o (Eq. 9).  The paper treats C_o as a property of the
        # origin, calibrated separately; deriving it from the mainline FD ties it
        # to v_free_main and makes the optimiser trade free-flow speed against
        # inflow throttling.  Free by default here, derived if not given.
        self.C_o = (2.0 * self._capacity_per_lane(v_free_main)
                    if C_o is None else float(C_o))

        # CFL for the upwind conservation update (paper Sec. 2: T v_free <= L)
        self.cfl = self.T * self.v_free.max() / self.L.min()

    # -- FD ------------------------------------------------------------------
    def _capacity_per_lane(self, v_free):
        if self.fd == "tri":
            return self.rho_crit * v_free
        return self.rho_crit * v_free * np.exp(-1.0 / self.a)

    def Veq(self, rho, v_free=None):
        """Equilibrium speed.  `fd="exp"` is the paper's Eq. (7)."""
        vf = self.v_free if v_free is None else v_free
        r = np.maximum(rho, 0.0)
        if self.fd == "tri":
            # triangular / Newell-Daganzo, continuous at rho_crit
            rj, rc = self.rho_jam_fd, self.rho_crit
            w = rc / max(rj - rc, 1e-6)
            cong = vf * w * (rj - r) / np.maximum(r, 1e-9)
            return np.where(r <= rc, vf, np.maximum(np.minimum(cong, vf), 0.0))
        x = r / self.rho_crit
        if self.fd == "exp2":
            # free-flow branch: exponent a; congested branch: exponent a_cong,
            # matched in value at rho = rho_crit.
            free = vf * np.exp(-(1.0 / self.a) * x ** self.a)
            cong = (vf * np.exp(-1.0 / self.a)
                    * np.exp(-(1.0 / self.a_cong) * (x ** self.a_cong - 1.0)))
            return np.where(r <= self.rho_crit, free, cong)
        return vf * np.exp(-(1.0 / self.a) * x ** self.a)   # Eq. (7)

    # -- VSL -----------------------------------------------------------------
    def set_vsl(self, v_ctrl):
        """Apply a variable speed limit (km/h per model segment, or None).

        Enters exactly where METANET-with-VSL puts it -- as an upper bound on
        the equilibrium speed the relaxation Eq. (13) drives toward:

            V_eff = min( V[rho_til] , (1 + alpha_vsl) * v_ctrl )

        `alpha_vsl` is the usual non-compliance factor (drivers exceed the sign
        by alpha_vsl).  This is the same form used by the project's own
        `metanet_model.CapacityDropLinkWithVsl`, so a VSL policy transfers
        directly.  Because min() of two quantities that are each <= v_free
        keeps the bound, Eq. (13) stays a convex combination and speed remains
        in [0, v_free]: the model's boundedness guarantee survives the control
        input, which is the whole reason for preferring this insertion point.
        """
        if v_ctrl is None:
            self.v_ctrl = None
        else:
            u = np.asarray(v_ctrl, float)
            if u.size == 1:
                u = np.full(self.n, float(u))
            elif u.size == 10:                 # per SUMO edge
                u = u[self.seg_edge]
            assert u.size == self.n, "v_ctrl must be scalar, 10 (edges) or n (segments)"
            self.v_ctrl = u
        return self

    # -- one step ------------------------------------------------------------
    def step(self, rho, v, w, d_vph):
        """One Bounded-METANET step. rho per lane, v km/h, w veh, d veh/h."""
        T, L, lam = self.T, self.L, self.lam

        # (6) flows.  Total veh/h -- see reading decision A.
        q = rho * v * lam

        # (9) origin flow
        q_o = d_vph + w / T
        q_o = min(q_o, self.C_o)
        if self.origin_throttle:
            lim = self.C_o * (self.rho_max - rho[0]) / (self.rho_max - self.rho_crit)
            q_o = min(q_o, max(lim, 0.0))
        q_o = max(q_o, 0.0)

        # (4) conservation
        q_in = np.empty(self.n)
        q_in[0] = q_o
        q_in[1:] = q[:-1]
        rho_n = rho + (T / (L * lam)) * (q_in - q)
        np.maximum(rho_n, 0.0, out=rho_n)

        # (14) virtual density -- downstream boundary per reading decision C
        rho_dn = np.empty(self.n)
        rho_dn[:-1] = rho[1:]
        if self.down_bc == "neumann":
            rho_dn[-1] = rho[-1]
        elif self.down_bc == "zero":
            rho_dn[-1] = 0.0
        else:                                    # "crit": uncongested destination
            rho_dn[-1] = min(rho[-1], self.rho_crit)
        rho_hat = rho + self.eta_t * (self.kap_t / (rho + self.kap_t)) * (rho_dn - rho)

        # (16) lane drop
        if self.phi_t > 0.0:
            c = (self.phi_t * self.dlam_over_lam
                 * min(rho[self.i_last_main] / self.rho_max, 1.0)
                 * np.clip(v / self.v_max_link, 0.0, 1.0))
            rho_til = np.where(self.ld_mask,
                               rho_hat + c * np.maximum(self.rho_max - rho_hat, 0.0),
                               rho_hat)
        else:
            rho_til = rho_hat

        # (13) speed relaxation -- no convection, no separate anticipation
        V = self.Veq(rho_til)
        if self.v_ctrl is not None:
            V = np.minimum(V, (1.0 + self.alpha_vsl) * self.v_ctrl)
        v_n = v + self.alpha * (V - v)

        # ABLATION ONLY: put METANET's convection term back
        if self.convection > 0.0:
            v_up = np.empty(self.n)
            v_up[0] = v[0]
            v_up[1:] = v[:-1]
            v_n = v_n + self.convection * (T / L) * v * (v_up - v)

        np.maximum(v_n, 0.0, out=v_n)

        # (8) origin queue
        w_n = max(w + T * (d_vph - q_o), 0.0)
        return rho_n, v_n, w_n, q_o

    # -- initial state -------------------------------------------------------
    def x0_empty(self):
        return np.zeros(self.n), self.v_free.copy(), 0.0

    # -- seg -> SUMO edge aggregation ---------------------------------------
    def to_edges(self, rho, v):
        if self._one_to_one or self.n_bneck == 1:
            return rho.copy(), v.copy()
        re = np.empty(10)
        ve = np.empty(10)
        re[:N_MAIN_EDGES] = rho[:N_MAIN_EDGES]
        ve[:N_MAIN_EDGES] = v[:N_MAIN_EDGES]
        rb, vb, Lb = rho[N_MAIN_EDGES:], v[N_MAIN_EDGES:], self.L[N_MAIN_EDGES:]
        re[-1] = (rb * Lb).sum() / Lb.sum()
        m = (rb * Lb).sum()
        ve[-1] = (rb * Lb * vb).sum() / m if m > 1e-9 else vb.mean()
        return re, ve

    def from_edges(self, rho_e, v_e):
        return (np.asarray(rho_e, float)[self.seg_edge].copy(),
                np.asarray(v_e, float)[self.seg_edge].copy())

    # -- bench contract ------------------------------------------------------
    def open_loop(self, inflow_vph):
        n_sec = len(inflow_vph)
        n_steps = int(round(n_sec / self.T_sec))
        idx = np.clip((np.arange(n_steps) * self.T_sec).astype(int), 0, n_sec - 1)
        d = np.asarray(inflow_vph, float)[idx]
        rho, v, w = self.x0_empty()
        n_e = self.n if self._one_to_one else 10
        RH = np.empty((n_steps, n_e))
        VV = np.empty((n_steps, n_e))
        for k in range(n_steps):
            rho, v, w, _ = self.step(rho, v, w, float(d[k]))
            RH[k], VV[k] = self.to_edges(rho, v)
        up = np.clip((np.arange(n_sec) / self.T_sec).astype(int), 0, n_steps - 1)
        return {"v": VV[up], "rho": RH[up]}

    def open_loop_segments(self, inflow_vph):
        """Same run, but returns the raw per-segment fields (for diagnostics)."""
        n_sec = len(inflow_vph)
        n_steps = int(round(n_sec / self.T_sec))
        idx = np.clip((np.arange(n_steps) * self.T_sec).astype(int), 0, n_sec - 1)
        d = np.asarray(inflow_vph, float)[idx]
        rho, v, w = self.x0_empty()
        RH = np.empty((n_steps, self.n))
        VV = np.empty((n_steps, self.n))
        WW = np.empty(n_steps)
        for k in range(n_steps):
            rho, v, w, _ = self.step(rho, v, w, float(d[k]))
            RH[k], VV[k], WW[k] = rho, v, w
        return RH, VV, WW

    def predict(self, rho0, v0, w0, inflow_vph, horizon_s):
        rho, v = self.from_edges(rho0, v0)
        w = float(w0)
        n_steps = int(round(horizon_s / self.T_sec))
        inflow_vph = np.asarray(inflow_vph, float)
        for k in range(n_steps):
            j = min(int(k * self.T_sec), len(inflow_vph) - 1)
            rho, v, w, _ = self.step(rho, v, w, float(inflow_vph[j]))
        re, ve = self.to_edges(rho, v)
        return {"v": ve, "rho": re}
