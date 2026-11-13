"""Soft-association trackers: JPDA and the variational (VB) tracker, on the GNN tracker's track management.

Both keep everything the GNN tracker does for the life of a track (tentative tracks, M-of-N confirmation,
coast deletion, duplicate dropping, filter tier judged from observables). They differ only in how
CONFIRMED tracks use the reports of a radar look:

  GNN    one hard assignment per look (Hungarian), then a normal Kalman update.
  JPDA   exact assignment marginals per look (gated clusters; Fortmann, Bar-Shalom & Scheffe 1983), and
         the JPDA state update: the moment-matched mixture over "missed" and every gated report.
  VB     the variational tracker of Turner, Bottone & Avasarala (NIPS 2014), causal, over a sliding window:
           filter step   at each look, the same marginals as JPDA, but the variational state update: one
                         soft pseudo-measurement, information-weighted (R~^-1 = sum_j g_j R_j^-1)
           window step   at every feature-grid time, for every cluster of tracks that shared reports in
                         the last window_s seconds, coordinate ascent on q(X) q(A): assignment marginals
                         per look from the smoothed states (scores w + E_q[log N(z; Hx, R)]), then a soft
                         Kalman/RTS smoother per track, `iterations` times; each track restarts from its
                         smoothed state at the newest look. Nothing after the grid time is used, so the
                         estimates stay causal (fixed-lag re-estimation of the window, not hindsight).

Scores. For track i and report j of one look, the log-odds of "i produced j" against "j is clutter and i
was missed":  W_ij = ln(P_D / (1 - P_D)) - ln rho_j + ln N(z_j; H x_i, S_ij),  gated by the same
chi-square test as the GNN. rho_j is the clutter density at the report (per m^3), from the radar's
specification (clutter_rate per look, spread uniformly in range, azimuth sector and 0-3 deg elevation, as
the generator does); it is never estimated from truth.

Reports that are mostly clutter-or-unexplained after the soft step (P(clutter) >= 0.5) go on to the
tentative tracks and to track initiation exactly as with the GNN.

Track life. A confirmed track counts as updated at a look if its total association probability is at least
min_assoc_mass; the coast limit runs from the last such look.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .assoc import marginals
from .gnn import GNNTracker, _predict, _q_block

SOFT_DEFAULTS = {
    "p_detect": 0.9,            # assumed detection probability of a confirmed track per look
    "min_assoc_mass": 0.2,      # total association probability that counts as "updated" (coast clock)
    "exact_max_tracks": 12,     # exact marginals up to this many tracks in a cluster, else loopy BP
    "window_s": 20.0,           # VB: sliding window length
    "iterations": 5,            # VB: at most this many coordinate-ascent iterations per window solve ...
    "tol_m": 1.0,               # ... stopping early once no smoothed position moves more than this
    "solve_isolated": False,    # VB: also re-solve windows of tracks that shared no reports
    "clutter_floor": 1.0e-18,   # clutter density (per m^3) outside the clutter region
}
LOG2PI = np.log(2 * np.pi)


def soft_settings(cfg):
    s = {**SOFT_DEFAULTS, **((cfg.get("tracking") or {}).get("soft") or {})}
    unknown = set(s) - set(SOFT_DEFAULTS)
    if unknown:
        raise KeyError(f"tracking.soft: unknown keys {sorted(unknown)}")
    return s


def clutter_density(cfg, sensor_type, range_m, el_deg):
    """Specified clutter density at each report (false alarms per look per m^3)."""
    s = cfg["sensors"]; lam = float(s["clutter_rate"])
    out = np.full(len(range_m), np.nan)
    for st, rt in s["radar_types"].items():
        k = np.asarray(sensor_type) == st
        if not k.any():
            continue
        r_ref = rt["ref_range_km"] * 1000.0
        daz = np.deg2rad(2 * s["r_p_sector_halfwidth_deg"]) if rt["sector"] == "threat" else 2 * np.pi
        r = np.clip(range_m[k], 0.05 * r_ref, r_ref)
        out[k] = lam / (0.95 * r_ref * daz * np.deg2rad(3.0) * r ** 2)
        out[k] = np.where((range_m[k] > r_ref) | (el_deg[k] > 3.0), 0.0, out[k])
    return out


def _lognorm(r, S):
    """ln N(r; 0, S) for r (k, 3), S (k, 3, 3)."""
    sol = np.linalg.solve(S, r[..., None])[..., 0]
    return -0.5 * (np.einsum("ki,ki->k", r, sol) + np.linalg.slogdet(S)[1] + 3 * LOG2PI)


def _pseudo(beta, Z, R, Ri=None):
    """Information-weighted soft pseudo-measurement: R~^-1 = sum b_j R_j^-1, z~ = R~ sum b_j R_j^-1 z_j."""
    Ri = np.linalg.inv(R) if Ri is None else Ri
    I = np.einsum("j,jab->ab", beta, Ri)
    Rt = np.linalg.inv(I)
    zt = Rt @ np.einsum("j,jab,jb->a", beta, Ri, Z)
    return zt, (Rt + Rt.T) / 2


def _kf(x, P, z, R):
    S = P[:3, :3] + R
    K = np.linalg.solve(S, P[:3, :]).T
    xn = x + K @ (z - x[:3])
    Pn = P - K @ S @ K.T
    return xn, (Pn + Pn.T) / 2


def soft_rts(ta, xa, Pa, steps, q):
    """VB window step for one track: soft Kalman filter from the anchor, then an RTS smoother.

    ta, xa, Pa   anchor time and state (the track's filtered state at or before the window start)
    steps        [(t, beta, Z, R, Ri), ...] in time order, one per look: association probabilities beta (m,)
                 of this track to the look's m reports Z (m, 3) with noise R (m, 3, 3) and inverses Ri;
                 beta summing to ~0 means "missed": that look is a pure prediction
    q            white-acceleration process noise intensity of the track's tier
    Returns {t: (x_smoothed, P_smoothed)} for every look time.

    Each look enters as the information-weighted pseudo-measurement of _pseudo, so with one-hot beta this
    is exactly a standard Kalman filter + RTS smoother, and beta = b on one report is the same as that
    report with noise R / b (tests/test_vb_tracker.py checks both against a batch solution).
    """
    times = [s[0] for s in steps]
    xf, Pf, xpred, Ppred, F_ = [], [], [], [], []
    x, P, tp = xa, Pa, ta
    for t, beta, Z, R, Ri in steps:
        dt = t - tp
        F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
        xq = F @ x; Pq = F @ P @ F.T + _q_block(np.array([q]), np.array([dt]))[0]
        if beta.sum() > 1e-9:
            zt, Rt = _pseudo(beta, Z, R, Ri)
            x, P = _kf(xq, Pq, zt, Rt)
        else:
            x, P = xq, Pq
        xf.append(x); Pf.append(P); xpred.append(xq); Ppred.append(Pq); F_.append(F); tp = t
    xs, Ps = xf[-1], Pf[-1]
    new = {times[-1]: (xs, Ps)}
    for k in range(len(times) - 2, -1, -1):
        C = np.linalg.solve(Ppred[k + 1].T, (Pf[k] @ F_[k + 1].T).T).T
        xs = xf[k] + C @ (xs - xpred[k + 1])
        Ps = Pf[k] + C @ (Ps - Ppred[k + 1]) @ C.T
        Ps = (Ps + Ps.T) / 2
        new[times[k]] = (xs, Ps)
    return new


class SoftTracker(GNNTracker):
    def __init__(self, cfg, mode="jpda", debug=0, timing_log=None):
        super().__init__(cfg, debug, timing_log)
        if mode not in ("jpda", "vb"):
            raise ValueError(mode)
        self.mode = mode
        self.s = soft_settings(cfg)
        self.wlog = np.log(self.s["p_detect"] / (1 - self.s["p_detect"]))
        cap = len(self.x)
        self.t_hit = np.zeros(cap)                         # last look with association mass >= min
        self.hist = {}                                     # track -> deque of (t, x, P) after each soft look
        self.looks = deque()                               # VB window buffer, one entry per gated cluster
        self.stats = dict(looks_soft=0, clusters=0, max_cluster_tracks=0, bp_clusters=0,
                          window_solves=0, window_tracks=0)

    def _grow(self):
        super()._grow()
        self.t_hit = np.concatenate([self.t_hit, np.zeros_like(self.t_hit)])

    def _new_track(self, z, R, t, sensor_type, rcs, vr=np.nan):
        i = super()._new_track(z, R, t, sensor_type, rcs, vr)
        self.t_hit[i] = t
        return i

    def _update(self, pairs, idx_pos, xp, Pp, Z, R, t):        # hard (tentative-track) updates
        super()._update(pairs, idx_pos, xp, Pp, Z, R, t)
        for p, _ in pairs:
            self.t_hit[self._cur_idx[p]] = t

    # ------------------------------------------------------------------ coast clock = last real hit
    def _expire(self, t):
        idx = np.flatnonzero(self.active[:self.n])
        if not len(idx):
            return
        coast = np.array([self.tiers[k]["coast_s"] for k in self.tier[idx]])
        conf = self.confirmed[idx]
        dead = (conf & (t - self.t_hit[idx] > coast)) | (~conf & (t - self.t0[idx] > self.g["confirm_window_s"]))
        for i in idx[dead]:
            self.active[i] = False; self.rcs.pop(i, None); self.hist.pop(i, None)

    def snapshot(self, tg):
        self._t_now = tg
        if self.mode in ("vb", "vbm"):
            with self.T("snapshot/window"):
                self._window_solve(tg)
        with self.T("snapshot/output"):
            return self._snapshot_out(tg)

    def _snapshot_out(self, tg):
        idx = np.flatnonzero(self.active[:self.n] & self.confirmed[:self.n])
        if not len(idx):
            return []
        coast = np.array([self.tiers[k]["coast_s"] for k in self.tier[idx]])
        keep = tg - self.t_hit[idx] <= coast
        idx = idx[keep]
        if not len(idx):
            return []
        q = np.array([self.tiers[k]["q"] for k in self.tier[idx]])
        xs, Ps = _predict(self.x[idx], self.P[idx], q, np.maximum(tg - self.t[idx], 0.0))
        k2 = self._merge_duplicates(idx, xs, Ps)
        idx, xs, Ps = idx[k2], xs[k2], Ps[k2]
        age = tg - self.t_hit[idx]
        ps = np.sqrt(np.trace(Ps[:, :3, :3], axis1=1, axis2=2)); vs = np.sqrt(np.trace(Ps[:, 3:, 3:], axis1=1, axis2=2))
        if getattr(self, "cov_log", None) is not None:          # bench only: full position covariance
            for k, i in enumerate(idx):
                self.cov_log[(int(i) + self.g["track_id_base"], float(tg))] = Ps[k, :3, :3]
        return [(int(i), tg, *xs[k], float(ps[k]), float(vs[k]), float(age[k])) for k, i in enumerate(idx)]

    # ------------------------------------------------------------------ one radar look
    def process_look(self, t, Z, R, sensor_type, rcs, vr=None, rho=None):
        T = self.T
        out = np.full(len(Z), -1)
        with T("look/expire"):
            self._expire(t)
        idx = np.flatnonzero(self.active[:self.n])
        used = np.zeros(len(Z), bool)
        self._cur_rcs = rcs; self._cur_idx = idx
        rho = np.full(len(Z), self.s["clutter_floor"]) if rho is None else np.maximum(rho, self.s["clutter_floor"])
        if len(idx):
            with T("look/predict"):
                dt = np.maximum(t - self.t[idx], 0.0)
                q = np.array([self.tiers[k]["q"] for k in self.tier[idx]])
                xp, Pp = _predict(self.x[idx], self.P[idx], q, dt)
            ci = np.flatnonzero(self.confirmed[idx])
            if len(ci) and len(Z):
                with T("look/soft"):
                    self._soft_look(t, idx, ci, xp, Pp, Z, R, rho, rcs, out, used)
            with T("look/tentative"):
                ti = np.flatnonzero(~self.confirmed[idx])             # tentative tracks: hard GNN step
                pairs = self._assign(ti, xp[ti], Pp[ti], Z, R, used)
                pairs = [(ti[a], j) for a, j in pairs]
                self._update(pairs, None, xp, Pp, Z, R, t)
                for p, j in pairs:
                    used[j] = True; out[j] = idx[p]
        with T("look/births"):
            for j in np.flatnonzero(~used):
                out[j] = self._new_track(Z[j], R[j], t, sensor_type, rcs[j], np.nan if vr is None else vr[j])
        with T("look/confirm"):
            self._confirm_loop(t)
        return out

    def _confirm_loop(self, t):
        live = np.flatnonzero(self.active[:self.n])
        for i in live:
            if not self.tier_fixed[i] and t - self.t0[i] >= self.acq["window_s"] and self.hits[i] >= 2:
                self._fix_tier(i)
            if not self.confirmed[i] and self.hits[i] >= self.g["confirm_hits"] \
                    and t - self.t0[i] <= self.g["confirm_window_s"]:
                self.confirmed[i] = True; self.t_conf[i] = t
                self.hist[i] = deque([(t, self.x[i].copy(), self.P[i].copy())])

    def _soft_look(self, t, idx, ci, xp, Pp, Z, R, rho, rcs, out, used):
        tm = self.T
        T_gate = tm("look/soft/gate"); T_gate.__enter__()
        HP = Pp[ci, :3, :3]
        d = np.linalg.norm(xp[ci, None, :3] - Z[None], axis=2)
        bound = np.sqrt(self.g["gate_chi2"] * (np.trace(HP, axis1=1, axis2=2)[:, None]
                                               + np.trace(R, axis1=1, axis2=2)[None, :]))
        a, b = np.nonzero(d <= bound)
        if not len(a):
            return
        S = HP[a] + R[b]; r = Z[b] - xp[ci[a], :3]
        d2 = np.einsum("ki,ki->k", r, np.linalg.solve(S, r[..., None])[..., 0])
        ok = d2 <= self.g["gate_chi2"]
        a, b, S, r = a[ok], b[ok], S[ok], r[ok]
        if not len(a):
            T_gate.__exit__(None, None, None)
            return
        llr = self.wlog - np.log(rho[b]) + _lognorm(r, S)
        T_gate.__exit__(None, None, None)
        nT, nZ = len(ci), len(Z)
        with tm("look/soft/cluster"):
            adj = coo_matrix((np.ones(len(a)), (a, nT + b)), shape=(nT + nZ, nT + nZ))
            ncomp, lab = connected_components(adj, directed=False)
        self.stats["looks_soft"] += 1
        for c in np.unique(lab[a]):
            T_b = tm("look/soft/build"); T_b.__enter__()
            T = np.flatnonzero(lab[:nT] == c); J = np.flatnonzero(lab[nT:] == c)
            tpos = {k: n for n, k in enumerate(T)}; jpos = {k: n for n, k in enumerate(J)}
            W = np.full((len(T), len(J)), -1e3); gate = np.zeros((len(T), len(J)), bool)
            sel = (lab[a] == c)
            for aa, bb, ll in zip(a[sel], b[sel], llr[sel]):
                W[tpos[aa], jpos[bb]] = ll; gate[tpos[aa], jpos[bb]] = True
            T_b.__exit__(None, None, None)
            with tm("look/soft/marginals"):
                G = marginals(W, self.s["exact_max_tracks"])
            T_u = tm("look/soft/update"); T_u.__enter__()
            self.stats["clusters"] += 1
            self.stats["max_cluster_tracks"] = max(self.stats["max_cluster_tracks"], len(T))
            self.stats["bp_clusters"] += int(len(T) > self.s["exact_max_tracks"])
            for n, k in enumerate(T):
                p = ci[k]; i = idx[p]
                beta = G[n] * gate[n]
                mass = beta.sum()
                x, P = xp[p], Pp[p]
                if mass > 1e-9:
                    if self.mode == "jpda":
                        x, P = self._jpda_update(x, P, beta, Z[J], R[J])
                    else:
                        zt, Rt = _pseudo(beta, Z[J], R[J])
                        x, P = _kf(x, P, zt, Rt)
                self.x[i] = x; self.P[i] = P; self.t[i] = t
                if mass >= self.s["min_assoc_mass"]:
                    self.t_hit[i] = t; self.hits[i] += 1
                if i in self.rcs and mass >= 0.5:
                    self.rcs[i].append(float(np.dot(beta, rcs[J]) / mass))
                if i in self.hist:
                    self.hist[i].append((t, x.copy(), P.copy()))
            T_u.__exit__(None, None, None)
            T_l = tm("look/soft/labels+buffer"); T_l.__enter__()
            best = np.argmax(G[:-1], axis=0)
            for m, j in enumerate(J):
                if G[-1, m] < 0.5:
                    used[j] = True
                if G[best[m], m] >= 0.5:
                    out[j] = idx[ci[T[best[m]]]]
            self._mark_resolved(t, [idx[ci[T[best[m]]]] for m in range(len(J)) if G[best[m], m] >= 0.5])
            if self.mode == "vb":
                Rj = R[J].copy(); Ri = np.linalg.inv(Rj)
                self.looks.append(dict(t=t, tids=idx[ci[T]].copy(), Z=Z[J].copy(), R=Rj, Ri=Ri,
                                       c0=self.wlog - np.log(rho[J]) - 0.5 * (np.linalg.slogdet(Rj)[1] + 3 * LOG2PI),
                                       gate=gate))
            T_l.__exit__(None, None, None)

    @staticmethod
    def _jpda_update(x, P, beta, Zc, Rc):
        b0 = max(1.0 - beta.sum(), 0.0)
        xs, Ps = [x], [P]; w = [b0]
        for bj, z, Rj in zip(beta, Zc, Rc):
            if bj <= 0:
                continue
            xn, Pn = _kf(x, P, z, Rj)
            xs.append(xn); Ps.append(Pn); w.append(bj)
        w = np.array(w); w = w / w.sum()
        X = np.array(xs); xm = w @ X
        dX = X - xm
        Pm = np.einsum("k,kab->ab", w, np.array(Ps)) + np.einsum("k,ka,kb->ab", w, dX, dX)
        return xm, (Pm + Pm.T) / 2

    # ------------------------------------------------------------------ VB window
    def _window_solve(self, tg):
        W = self.s["window_s"]; t0 = tg - W
        while self.looks and self.looks[0]["t"] < t0:
            self.looks.popleft()
        for i, h in self.hist.items():
            while len(h) > 1 and h[1][0] <= t0:
                h.popleft()
        if not self.looks:
            return
        T_c = self.T("snapshot/window/clusters"); T_c.__enter__()
        # clusters of tracks that shared reports in the window
        parent = {}

        def find(u):
            while parent.setdefault(u, u) != u:
                parent[u] = parent[parent[u]]; u = parent[u]
            return u
        for L in self.looks:
            alive = [int(k) for k in L["tids"] if self.active[k]]
            shared = len(alive) > 1 and (L["gate"].sum(0) > 1).any()
            if shared or self.s["solve_isolated"]:
                for k in alive:
                    find(k)
                for k in alive[1:]:
                    parent[find(k)] = find(alive[0])
        groups = {}
        for k in parent:
            groups.setdefault(find(k), []).append(k)
        T_c.__exit__(None, None, None)
        for members in groups.values():
            if len(members) < 2 and not self.s["solve_isolated"]:
                continue
            with self.T("snapshot/window/solve"):
                self._solve_cluster(sorted(members), t0)

    def _solve_cluster(self, members, t0):
        tm = self.T
        T_s = tm("snapshot/window/solve/setup"); T_s.__enter__()
        mset = set(members)
        looks = [L for L in self.looks if mset.intersection(int(k) for k in L["tids"])]
        if not looks:
            T_s.__exit__(None, None, None)
            return
        anchors = {}
        for i in members:
            h = self.hist.get(i)
            if not h:
                T_s.__exit__(None, None, None)
                return
            anchors[i] = h[0]                                     # state at or before the window start
        # initial q(x): the filtered states at each look time
        means = {i: {} for i in members}
        for i in members:
            for (t, x, P) in self.hist[i]:
                means[i][t] = (x, P)
        q_of = {i: self.tiers[self.tier[i]]["q"] for i in members}
        T_s.__exit__(None, None, None)
        for it in range(self.s["iterations"]):
            # E-step for A: per look, marginals from the current smoothed states
            betas = {i: {} for i in members}
            T_a = tm("snapshot/window/solve/assoc_scores"); T_a.__enter__()
            for L in looks:
                tids = [int(k) for k in L["tids"]]
                rows = [n for n, k in enumerate(tids) if k in mset and L["t"] in means[k]]
                if not rows:
                    continue
                Wm = np.full((len(rows), len(L["Z"])), -1e3)
                for rr, n in enumerate(rows):
                    x, P = means[tids[n]][L["t"]]
                    g = L["gate"][n]
                    if not g.any():
                        continue
                    Ri = L["Ri"][g]; r = L["Z"][g] - x[:3]
                    # E_q[ln N(z; Hx, R)] = ln N(z; Hm, R) - tr(R^-1 P_pos) / 2
                    Wm[rr, g] = L["c0"][g] - 0.5 * (np.einsum("ka,kab,kb->k", r, Ri, r)
                                                    + np.einsum("kab,ba->k", Ri, P[:3, :3]))
                T_a.__exit__(None, None, None)
                with tm("snapshot/window/solve/assoc_marginals"):
                    G = marginals(Wm, self.s["exact_max_tracks"])
                T_a.__enter__()
                for rr, n in enumerate(rows):
                    betas[tids[n]][L["t"]] = (G[rr] * L["gate"][n], L)
            T_a.__exit__(None, None, None)
            # E-step for X: soft Kalman filter + RTS smoother per track over its looks
            change = 0.0
            T_x = tm("snapshot/window/solve/smoother"); T_x.__enter__()
            for i in members:
                # only looks AFTER the anchor: the anchor state already contains the look at its own time
                # (a look exactly at the window start used to be applied a second time, as a dt = 0 step)
                times = sorted(t for t in betas[i] if t > anchors[i][0])
                if not times:
                    continue
                steps = [(t, betas[i][t][0], betas[i][t][1]["Z"], betas[i][t][1]["R"], betas[i][t][1]["Ri"])
                         for t in times]
                new = soft_rts(*anchors[i], steps, q_of[i])
                if it:
                    common = [t for t in new if t in means[i]]
                    if common:
                        change = max(change, max(float(np.max(np.abs(new[t][0][:3] - means[i][t][0][:3]))) for t in common))
                means[i] = new
            T_x.__exit__(None, None, None)
            self.stats["window_iters"] = self.stats.get("window_iters", 0) + 1
            if it and change < self.s["tol_m"]:
                self.stats["window_early_stops"] = self.stats.get("window_early_stops", 0) + 1
                break
        self.stats["window_solves"] += 1; self.stats["window_tracks"] += len(members)
        for i in members:
            if means[i] and self.active[i]:
                tl = max(means[i])
                if tl >= self.t[i] - 1e-9:                         # newest look: restart the filter there
                    self.x[i], self.P[i] = means[i][tl]; self.t[i] = tl

    # ------------------------------------------------------------------ whole run (adds clutter density)
    def run(self, renu_all: pd.DataFrame, grid: np.ndarray):
        r = renu_all.sort_values(["time", "sensor_id"], kind="mergesort")
        rho = clutter_density(self.cfg, r.sensor_type.values, r.range_m.values.astype(float),
                              r.el_deg.values.astype(float))
        self._rho_sorted = rho
        return super().run(renu_all, grid)
