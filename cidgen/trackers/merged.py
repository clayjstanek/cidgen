"""Variational tracker with merged (unresolved) reports: method `vbm`.

The GNN, JPDA and VB trackers all assume that a report comes from at most one aircraft. When a radar
cannot resolve a group (a swarm 50 m apart in a 2 deg counter-UAS beam, a formation inside one 150 m x
1.5 deg surveillance cell) it returns ONE report for several aircraft, and every one of those trackers
must then let all but one member go unexplained, coast and eventually die.

Here a report may be explained by a SET of tracks:

  events      for report j: clutter; a single track i (as before); or a set S of 2..max_set tracks that
              the radar could not have resolved. A set is allowed only if every pair in it has
              probability >= p_unresolved_min of lying inside one resolution cell of THIS radar
              (|dr| < res_r_m, |daz| < res_az_deg, |del| < 1.5 res_az_deg, the generator's own rule, from
              the radar specification), computed from the predicted positions and their covariances.
  scores      single:  ln(PD/(1-PD)) - ln rho + ln N(z; Hx_i, R + HP_iH')
              set S:   |S| ln(PD/(1-PD)) - ln rho + ln N(z; mean_S Hx, R + C_S + sum HP_iH' / |S|^2)
                       + sum over pairs of ln p_unresolved
              C_S = merge_spread x the spread of the members' positions (a merged report lands at an
              echo-weighted centroid of the members, not exactly at their mean).
  marginals   exact over all valid joint assignments (each track used at most once per look):
              forward-backward over subsets of tracks, with "report -> set" steps (assoc.exact_set_marginals).
  update      mean-field (the variational update): a member i of set S sees the pseudo-measurement
              |S| z - sum_{k in S, k != i} H m_k with covariance |S|^2 (R + C_S) + sum_{k != i} H P_k H'.
              Members are held APART by this (each is told where the others are); JPDA's mixture
              update pulls them together instead. All of a track's events are combined information-
              weighted, then one Kalman update (filter step), or one soft RTS pass (window step).
  window      as `vb`: every grid time, clusters of tracks that shared reports or sets in the last
              window_s seconds are re-solved by coordinate ascent with the smoothed states.

Duplicate dropping (needed because of radar registration bias) must not delete real members of a
group: two tracks that took DIFFERENT reports in the same look within the last protect_resolved_s
seconds are two objects, and are never dropped as duplicates.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import ndtr

from .assoc import exact_set_marginals, marginals
from .gnn import _predict, _q_block
from .soft import LOG2PI, SoftTracker, _kf, _lognorm

MERGE_DEFAULTS = {
    "max_set": 4,                 # largest set of tracks one report may stand for
    "max_sets_per_report": 24,    # keep only the most likely sets per report
    "p_unresolved_min": 0.05,     # each pair in a set must be unresolved with at least this probability
    "merge_spread": 0.5,          # C_S = merge_spread x covariance of the members' positions
    "max_cluster_tracks": 8,      # larger clusters: single-track events only (exact or BP marginals)
    "protect_resolved_s": 30.0,   # never drop as duplicates two tracks seen resolved this recently
    "starved_miss": 0.3,          # sets are tried only when a track's miss probability exceeds this
    "window_min_setmass": 0.05,   # window re-solve only for clusters whose looks had merged hypotheses
}


def merge_settings(cfg):
    s = {**MERGE_DEFAULTS, **((cfg.get("tracking") or {}).get("merge") or {})}
    unknown = set(s) - set(MERGE_DEFAULTS)
    if unknown:
        raise KeyError(f"tracking.merge: unknown keys {sorted(unknown)}")
    return s


def p_unresolved(xa, Pa, xb, Pb, site, res_r, res_az_deg):
    """Probability that two targets lie in one resolution cell of a radar at `site` (Gaussian approx)."""
    d = xa[:3] - xb[:3]
    mid = (xa[:3] + xb[:3]) / 2 - site
    rng = max(np.linalg.norm(mid), 1.0); ur = mid / rng
    ground = max(np.hypot(mid[0], mid[1]), 1.0)
    uaz = np.array([mid[1], -mid[0], 0.0]) / ground
    uel = np.array([0.0, 0.0, 1.0]) - ur[2] * ur
    uel = uel / max(np.linalg.norm(uel), 1e-9)
    S = Pa[:3, :3] + Pb[:3, :3]
    out = 1.0
    for u, scale, half in ((ur, 1.0, res_r),
                           (uaz, np.rad2deg(1.0 / ground), res_az_deg),
                           (uel, np.rad2deg(1.0 / rng), 1.5 * res_az_deg)):
        mu = float(d @ u) * scale
        sd = max(np.sqrt(float(u @ S @ u)) * scale, 1e-9)
        out *= ndtr((half - mu) / sd) - ndtr((-half - mu) / sd)
    return float(out)


class MergedVBTracker(SoftTracker):
    def __init__(self, cfg, debug=0, timing_log=None):
        super().__init__(cfg, "vb", debug, timing_log)
        self.mode = "vbm"
        self.m = merge_settings(cfg)
        self.g["protect_resolved_s"] = self.m["protect_resolved_s"]   # the shared protection (gnn.py)
        self.stats.update(set_events=0, set_mass=0.0)

    # ------------------------------------------------------------------ events of one cluster
    def _events(self, Zc, Rc, rhoc, xs, Ps, site, res, gate_tr, starved=None):
        """Events per report for a cluster: list of (mask, logw, kind) with kind = (members tuple, C_S).
        xs, Ps: states of the cluster's tracks at the look time; gate_tr (n, m) singly-gated pairs."""
        n, m = len(xs), len(Zc)
        wlog = self.wlog; g2 = self.g["gate_chi2"]
        pu = np.zeros((n, n))
        if n > 1 and starved is not None and starved.any():
            for a, b in itertools.combinations(range(n), 2):
                if starved[a] or starved[b]:                 # only pairs that could form a useful set
                    pu[a, b] = pu[b, a] = p_unresolved(xs[a], Ps[a], xs[b], Ps[b], site, *res)
        ev = []
        for j in range(m):
            e = []
            for i in range(n):
                if gate_tr[i, j]:
                    S = Ps[i][:3, :3] + Rc[j]; r = Zc[j] - xs[i][:3]
                    e.append((1 << i, wlog - np.log(rhoc[j]) + float(_lognorm(r[None], S[None])[0]), ((i,), None)))
            # candidate members: near the report, within the merged gate
            cand = [i for i in range(n) if np.linalg.norm(Zc[j] - xs[i][:3]) <=
                    np.sqrt(g2 * (np.trace(Ps[i][:3, :3]) + np.trace(Rc[j]))) + res[0] + np.deg2rad(res[1]) *
                    np.linalg.norm(Zc[j] - site)]
            sets = []
            cliques = self._cliques(cand, pu) if (starved is not None and starved.any()) else []
            cliques = [c for c in cliques if any(starved[i] for i in c)]   # a set must rescue a starved track
            for Sset in cliques:
                k = len(Sset)
                lp = float(sum(np.log(pu[a, b]) for a, b in itertools.combinations(Sset, 2)))
                if True:
                    X = np.array([xs[i][:3] for i in Sset]); mean = X.mean(0)
                    C = self.m["merge_spread"] * np.cov(X.T, bias=True)
                    Pm = sum(Ps[i][:3, :3] for i in Sset) / k ** 2
                    S = Rc[j] + C + Pm; r = Zc[j] - mean
                    d2 = float(r @ np.linalg.solve(S, r))
                    if d2 > g2:
                        continue
                    lw = k * wlog - np.log(rhoc[j]) + float(_lognorm(r[None], S[None])[0]) + lp
                    sets.append((sum(1 << i for i in Sset), lw, (Sset, C)))
            sets.sort(key=lambda z: -z[1])
            e.extend(sets[: self.m["max_sets_per_report"]])      # the most likely sets only
            ev.append(e)
        return ev

    def _cliques(self, cand, pu):
        """Sets of 2..max_set candidates, every pair unresolved with probability >= p_unresolved_min
        (grown member by member, so hopeless branches are cut early)."""
        pmin = self.m["p_unresolved_min"]; kmax = self.m["max_set"]
        out = []

        def grow(cur, rest):
            if len(cur) >= 2:
                out.append(tuple(cur))
            if len(cur) == kmax:
                return
            for n, c in enumerate(rest):
                if all(pu[c, o] >= pmin for o in cur):
                    grow(cur + [c], rest[n + 1:])
        for n, c in enumerate(cand):
            grow([c], cand[n + 1:])
        return out

    def _pseudo_for(self, i, events_j, probs_j, Zj, Rj, xs, Ps):
        """Information-weighted pseudo-measurement for cluster track i from one report's events."""
        info = np.zeros((3, 3)); vec = np.zeros(3); mass = 0.0
        for (mask, _, (members, C)), p in zip(events_j, probs_j):
            if not (mask >> i) & 1 or p <= 1e-12:
                continue
            k = len(members)
            if k == 1:
                zi, Ri = Zj, Rj
            else:
                others = [o for o in members if o != i]
                zi = k * Zj - sum(xs[o][:3] for o in others)
                Ri = k ** 2 * (Rj + C) + sum(Ps[o][:3, :3] for o in others)
            Rinv = np.linalg.inv(Ri)
            info += p * Rinv; vec += p * Rinv @ zi; mass += p
        return info, vec, mass

    # ------------------------------------------------------------------ one look (filter step)
    def _soft_look(self, t, idx, ci, xp, Pp, Z, R, rho, rcs, out, used):
        a0, b0 = self._cur_slice
        site = self._site_sorted[a0]; res = self._res_sorted[a0]
        HP = Pp[ci, :3, :3]
        cell = res[0] + np.deg2rad(res[1]) * np.linalg.norm(Z - site, axis=1)      # one resolution cell
        d = np.linalg.norm(xp[ci, None, :3] - Z[None], axis=2)
        bound = np.sqrt(self.g["gate_chi2"] * (np.trace(HP, axis1=1, axis2=2)[:, None]
                                               + np.trace(R, axis1=1, axis2=2)[None, :]))
        a, b = np.nonzero(d <= bound + cell[None, :])
        if not len(a):
            return
        # single-track gate (as GNN/JPDA)
        S1 = HP[a] + R[b]; r1 = Z[b] - xp[ci[a], :3]
        d2 = np.einsum("ki,ki->k", r1, np.linalg.solve(S1, r1[..., None])[..., 0])
        single = d2 <= self.g["gate_chi2"]
        # clusters are built from single-gate links, plus a "rescue" link for a track with NO single gate
        # to its nearest report within a cell (it may be hidden inside that report); this keeps clusters
        # as small as JPDA's while still letting an unexplained member join a merged report
        has_single = np.zeros(len(ci), bool); has_single[a[single]] = True
        keep = single.copy()
        for k in np.flatnonzero(~has_single):
            sel = np.flatnonzero((a == k) & ~single)
            if len(sel):
                keep[sel[np.argmin(d[k, b[sel]])]] = True
        a, b, single = a[keep], b[keep], single[keep]
        nT, nZ = len(ci), len(Z)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        adj = coo_matrix((np.ones(len(a)), (a, nT + b)), shape=(nT + nZ, nT + nZ))
        _, lab = connected_components(adj, directed=False)
        self.stats["looks_soft"] += 1
        for c in np.unique(lab[a]):
            T = np.flatnonzero(lab[:nT] == c); J = np.flatnonzero(lab[nT:] == c)
            tpos = {k: n for n, k in enumerate(T)}; jpos = {k: n for n, k in enumerate(J)}
            gate = np.zeros((len(T), len(J)), bool)
            sel = lab[a] == c
            for aa, bb, sg in zip(a[sel], b[sel], single[sel]):
                if sg:
                    gate[tpos[aa], jpos[bb]] = True
            xs = [xp[ci[k]] for k in T]; Ps = [Pp[ci[k]] for k in T]
            self.stats["clusters"] += 1
            self.stats["max_cluster_tracks"] = max(self.stats["max_cluster_tracks"], len(T))
            starved = None
            if len(T) <= self.m["max_cluster_tracks"] and len(T) > 1:
                ev = self._events(Z[J], R[J], rho[J], xs, Ps, site, res, gate)          # single-track events
                P, Pc = exact_set_marginals([[(mk, lw) for mk, lw, _ in e] for e in ev], len(T), len(J))
                held = np.zeros(len(T))
                for e, p in zip(ev, P):
                    for (mk, _, (mem, _C)), pp in zip(e, p):
                        held[mem[0]] += pp
                starved = held < 1.0 - self.m["starved_miss"]
                if starved.any():                                # some track went unexplained: try sets
                    ev = self._events(Z[J], R[J], rho[J], xs, Ps, site, res, gate, starved)
                    P, Pc = exact_set_marginals([[(mk, lw) for mk, lw, _ in e] for e in ev], len(T), len(J))
            elif len(T) == 1:
                ev = self._events(Z[J], R[J], rho[J], xs, Ps, site, res, gate)
                P, Pc = exact_set_marginals([[(mk, lw) for mk, lw, _ in e] for e in ev], len(T), len(J))
            else:                                               # too big: single-track events only
                W = np.full((len(T), len(J)), -1e3)
                for n_ in range(len(T)):
                    for m_ in range(len(J)):
                        if gate[n_, m_]:
                            S = Ps[n_][:3, :3] + R[J[m_]]; r = Z[J[m_]] - xs[n_][:3]
                            W[n_, m_] = self.wlog - np.log(rho[J[m_]]) + float(_lognorm(r[None], S[None])[0])
                G = marginals(W, self.s["exact_max_tracks"])
                ev = [[(1 << n_, W[n_, m_], ((n_,), None)) for n_ in range(len(T)) if gate[n_, m_]] for m_ in range(len(J))]
                P = [np.array([G[n_, m_] for n_ in range(len(T)) if gate[n_, m_]]) for m_ in range(len(J))]
                Pc = G[-1]
            # per-track mass on each report (for labels and the resolved-pair record)
            tm = np.zeros((len(T), len(J)))
            for m_, (e, p) in enumerate(zip(ev, P)):
                for (mk, _, (members, _C)), pp in zip(e, p):
                    for n_ in members:
                        tm[n_, m_] += pp
                    if len(members) > 1:
                        self.stats["set_events"] += 1; self.stats["set_mass"] += float(pp)
            # mean-field Kalman update per track
            for n_, k in enumerate(T):
                p_ = ci[k]; i = idx[p_]
                info = np.zeros((3, 3)); vec = np.zeros(3); mass = 0.0
                for m_ in range(len(J)):
                    I_, v_, ms = self._pseudo_for(n_, ev[m_], P[m_], Z[J[m_]], R[J[m_]], xs, Ps)
                    info += I_; vec += v_; mass += ms
                x, Pk = xp[p_], Pp[p_]
                if mass > 1e-9:
                    Rt = np.linalg.inv(info); zt = Rt @ vec
                    x, Pk = _kf(x, Pk, zt, (Rt + Rt.T) / 2)
                self.x[i] = x; self.P[i] = Pk; self.t[i] = t
                if mass >= self.s["min_assoc_mass"]:
                    self.t_hit[i] = t; self.hits[i] += 1
                if i in self.rcs and mass >= 0.5:
                    self.rcs[i].append(float(np.dot(tm[n_], rcs[J]) / max(tm[n_].sum(), 1e-9)))
                if i in self.hist:
                    self.hist[i].append((t, x.copy(), Pk.copy()))
            best = np.argmax(tm, axis=0)
            winners = []
            for m_, j in enumerate(J):
                if Pc[m_] < 0.5:
                    used[j] = True
                if tm[best[m_], m_] >= 0.5:
                    out[j] = idx[ci[T[best[m_]]]]
                    winners.append((j, idx[ci[T[best[m_]]]]))
            self._mark_resolved(t, [w for _, w in winners])     # different reports -> different objects
            setmass = float(sum(pp for e, p in zip(ev, P) for (mk, _, (mem, _C)), pp in zip(e, p) if len(mem) > 1))
            self.looks.append(dict(t=t, tids=idx[ci[T]].copy(), Z=Z[J].copy(), R=R[J].copy(),
                                   rho=rho[J].copy(), gate=gate, site=site, res=res, setmass=setmass,
                                   starved=None if starved is None else starved.copy()))

    # ------------------------------------------------------------------ window step
    def _solve_cluster(self, members, t0):
        mset = set(members)
        looks = [L for L in self.looks if mset.intersection(int(k) for k in L["tids"])]
        if not looks:
            return
        anchors = {}
        for i in members:
            h = self.hist.get(i)
            if not h:
                return
            anchors[i] = h[0]
        means = {i: {t: (x, P) for (t, x, P) in self.hist[i]} for i in members}
        q_of = {i: self.tiers[self.tier[i]]["q"] for i in members}
        for it in range(self.s["iterations"]):
            obs = {i: {} for i in members}                      # track -> look time -> (info, vec, mass)
            for L in looks:
                tids = [int(k) for k in L["tids"]]
                rows = [n for n, k in enumerate(tids) if k in mset and L["t"] in means[k]]
                if not rows:
                    continue
                xs = [means[tids[n]][L["t"]][0] for n in rows]; Ps = [means[tids[n]][L["t"]][1] for n in rows]
                gate = L["gate"][rows]
                if len(rows) <= self.m["max_cluster_tracks"]:
                    # scores from the smoothed states q(x) (predictive form ln N(z; m, R + P); the strict
                    # E_q form ln N(z; m, R) - tr(R^-1 P)/2 differs only in how P enters)
                    st = None if L.get("starved") is None else L["starved"][rows]
                    ev = self._events(L["Z"], L["R"], L["rho"], xs, Ps, L["site"], L["res"], gate,
                                      st if L.get("setmass", 0.0) > 0.01 else None)
                    P, _ = exact_set_marginals([[(mk, lw) for mk, lw, _ in e] for e in ev], len(rows), len(L["Z"]))
                else:
                    continue
                for n_, nn in enumerate(rows):
                    info = np.zeros((3, 3)); vec = np.zeros(3); mass = 0.0
                    for m_ in range(len(L["Z"])):
                        I_, v_, ms = self._pseudo_for(n_, ev[m_], P[m_], L["Z"][m_], L["R"][m_], xs, Ps)
                        info += I_; vec += v_; mass += ms
                    if mass > 1e-9:
                        obs[tids[nn]][L["t"]] = (info, vec)
            change = 0.0
            for i in members:
                ta, xa, Pa = anchors[i]
                times = sorted(t for t in obs[i] if t > ta)
                if not times:
                    continue
                xf, Pf, xpred, Ppred, F_ = [], [], [], [], []
                x, P, tp = xa, Pa, ta
                for t in times:
                    dt = t - tp
                    F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
                    xq = F @ x; Pq = F @ P @ F.T + _q_block(np.array([q_of[i]]), np.array([dt]))[0]
                    info, vec = obs[i][t]
                    Rt = np.linalg.inv(info); zt = Rt @ vec
                    x, P = _kf(xq, Pq, zt, (Rt + Rt.T) / 2)
                    xf.append(x); Pf.append(P); xpred.append(xq); Ppred.append(Pq); F_.append(F); tp = t
                xs_, Ps_ = xf[-1], Pf[-1]
                new = {times[-1]: (xs_, Ps_)}
                for k in range(len(times) - 2, -1, -1):
                    C = np.linalg.solve(Ppred[k + 1].T, (Pf[k] @ F_[k + 1].T).T).T
                    xs_ = xf[k] + C @ (xs_ - xpred[k + 1])
                    Ps_ = Pf[k] + C @ (Ps_ - Ppred[k + 1]) @ C.T
                    new[times[k]] = (xs_, (Ps_ + Ps_.T) / 2)
                common = [t for t in new if t in means[i]]
                if it and common:
                    change = max(change, max(float(np.max(np.abs(new[t][0][:3] - means[i][t][0][:3]))) for t in common))
                means[i].update(new)
            if it and change < self.s["tol_m"]:
                break
        self.stats["window_solves"] += 1; self.stats["window_tracks"] += len(members)
        for i in members:
            if means[i] and self.active[i]:
                tl = max(means[i])
                if tl >= self.t[i] - 1e-9:
                    self.x[i], self.P[i] = means[i][tl]; self.t[i] = tl

    def _window_solve(self, tg):
        # clusters: tracks linked by any shared report OR candidate set in the window
        W = self.s["window_s"]; t0 = tg - W
        while self.looks and self.looks[0]["t"] < t0:
            self.looks.popleft()
        for i, h in self.hist.items():
            while len(h) > 1 and h[1][0] <= t0:
                h.popleft()
        parent = {}

        def find(u):
            while parent.setdefault(u, u) != u:
                parent[u] = parent[parent[u]]; u = parent[u]
            return u
        for L in self.looks:
            alive = [int(k) for k in L["tids"] if self.active[k]]
            if len(alive) > 1 and L.get("setmass", 0.0) >= self.m["window_min_setmass"]:
                for k in alive:
                    find(k)
                for k in alive[1:]:
                    parent[find(k)] = find(alive[0])
        groups = {}
        for k in parent:
            groups.setdefault(find(k), []).append(k)
        for mem in groups.values():
            if 2 <= len(mem) <= self.m["max_cluster_tracks"]:
                self._solve_cluster(sorted(mem), t0)

    # ------------------------------------------------------------------ whole run
    def run(self, renu_all, grid):
        r = renu_all.sort_values(["time", "sensor_id"], kind="mergesort")
        self._site_sorted = r[["site_e", "site_n", "site_u"]].to_numpy(float)
        rt = self.cfg["sensors"]["radar_types"]
        self._res_sorted = np.array([(rt[s]["res_r_m"], rt[s]["res_az_deg"]) for s in r.sensor_type.values], float)
        return super().run(renu_all, grid)
