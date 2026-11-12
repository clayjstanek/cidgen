"""Global nearest neighbour (GNN) tracker: the baseline multi-target tracker.

Per radar look (all reports from one sensor at one time):
  1. predict every active track to the look time (constant-velocity Kalman filter, per-track q);
  2. gate: Mahalanobis distance d2 = r' S^-1 r <= gate_chi2 (S = H P H' + R);
  3. assign reports one-to-one, confirmed tracks first, by minimum total cost d2 + ln|S|
     (scipy.optimize.linear_sum_assignment); then remaining reports to tentative tracks;
  4. update assigned tracks; every report left over starts a new tentative track;
  5. confirm a tentative track after confirm_hits updates within confirm_window_s of its first report;
     delete a tentative track not confirmed in time, and any track not updated for its coast time.
  6. at every feature-grid time, drop duplicate confirmed tracks: two tracks whose 6-D states agree
     (d2 <= merge_chi2 on P1 + P2) at merge_checks consecutive grid times are the same object seen through
     two radars' different registration biases; the younger one is deleted.

Filter settings (q, initial velocity sigma, coast) follow the observable tier, never the truth:
a first guess from the first report (altitude above tier3_min_alt_m, or measured radial speed at or
above the acquisition rule's Tier 3 speed -> Tier 3; low, small, counter-UAS
radar -> Tier 1; else Tier 2), then fixed by the acquisition rule (speed, altitude, median RCS) once
acquisition window_s has passed. Clutter reports are tracked like any other report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..tracking import ACQUISITION_DEFAULTS, tracker_settings
from .timing import Timing

GNN_DEFAULTS = {
    "gate_chi2": 16.0,          # 3-D chi-square gate (about 99.9 %)
    "confirm_hits": 3,          # updates needed to confirm a tentative track ...
    "confirm_window_s": 10.0,   # ... within this time of its first report
    "tier3_min_alt_m": 15000.0, # first-report guess: above this altitude, use Tier 3 filter settings
    "merge_chi2": 12.6,         # duplicate-track test: 6-D state difference d2 = dx' (P1 + P2)^-1 dx below this ...
    "merge_checks": 3,          # ... at this many consecutive grid times -> drop the younger track
    "protect_resolved_s": 0.0,  # never drop as duplicates two tracks that took DIFFERENT reports in one
                                # look within this many seconds (0 = off, the original behaviour)
    "track_id_base": 1_000_000, # track numbers start here, so they never collide with true object ids
}
H = np.hstack([np.eye(3), np.zeros((3, 3))])


def gnn_settings(cfg):
    g = {**GNN_DEFAULTS, **((cfg.get("tracking") or {}).get("gnn") or {})}
    unknown = set(g) - set(GNN_DEFAULTS)
    if unknown:
        raise KeyError(f"tracking.gnn: unknown keys {sorted(unknown)}")
    return g


def _q_block(q, dt):
    """Batched white-acceleration process noise, q (n,), dt (n,) -> (n, 6, 6)."""
    n = len(dt); Q = np.zeros((n, 6, 6))
    a, b, c = q * dt ** 3 / 3, q * dt ** 2 / 2, q * dt
    for k in range(3):
        Q[:, k, k] = a; Q[:, k, k + 3] = b; Q[:, k + 3, k] = b; Q[:, k + 3, k + 3] = c
    return Q


def _predict(x, P, q, dt):
    xp = x.copy(); xp[:, :3] += x[:, 3:] * dt[:, None]
    F = np.repeat(np.eye(6)[None], len(dt), 0); F[:, 0, 3] = F[:, 1, 4] = F[:, 2, 5] = dt
    Pp = F @ P @ F.transpose(0, 2, 1) + _q_block(q, dt)
    return xp, Pp


class GNNTracker:
    def __init__(self, cfg, debug=0, timing_log=None):
        self.cfg = cfg
        # --debug 1: section timers (see timing.py); anything else: timers off
        self.T = Timing(enabled=(debug == 1), log_path=timing_log if debug == 1 else None)
        self.g = gnn_settings(cfg)
        self.tiers = tracker_settings(cfg)
        self.acq = {**ACQUISITION_DEFAULTS, **((cfg.get("tracking") or {}).get("acquisition") or {})}
        cap = 4096
        self.x = np.zeros((cap, 6)); self.P = np.zeros((cap, 6, 6)); self.t = np.zeros(cap)
        self.tier = np.zeros(cap, int); self.tier_fixed = np.zeros(cap, bool)
        self.t0 = np.zeros(cap); self.hits = np.zeros(cap, int); self.confirmed = np.zeros(cap, bool)
        self.t_conf = np.full(cap, np.nan); self.active = np.zeros(cap, bool); self.n = 0
        self.rcs = {}                                      # track -> RCS samples during acquisition
        self.close = {}                                    # (older, younger) -> consecutive duplicate checks
        self.resolved = {}                                 # (a, b) -> last look they took different reports
        self.n_protected = 0
        self.n_merged = 0

    # ------------------------------------------------------------------ bookkeeping
    def _grow(self):
        for k in ("x", "P", "t", "tier", "tier_fixed", "t0", "hits", "confirmed", "t_conf", "active"):
            a = getattr(self, k)
            pad = np.full_like(a, np.nan) if k == "t_conf" else np.zeros_like(a)
            setattr(self, k, np.concatenate([a, pad]))

    def _first_tier(self, z, sensor_type, rcs, vr=np.nan):
        if z[2] >= self.g["tier3_min_alt_m"] or (np.isfinite(vr) and abs(vr) >= self.acq["tier3_min_speed_mps"]):
            return 3
        if sensor_type == "R_C" or (z[2] < self.acq["tier1_max_alt_m"] and np.isfinite(rcs)
                                     and rcs < self.acq["tier1_max_rcs_dbsm"]):
            return 1
        return 2

    def _new_track(self, z, R, t, sensor_type, rcs, vr=np.nan):
        if self.n == len(self.x):
            self._grow()
        i = self.n; self.n += 1
        tier = self._first_tier(z, sensor_type, rcs, vr)
        self.x[i] = np.r_[z, 0, 0, 0]
        self.P[i] = 0.0; self.P[i, :3, :3] = R
        self.P[i, 3:, 3:] = np.diag(np.square(self.tiers[tier]["vel_sigma_mps"]))
        self.t[i] = t; self.t0[i] = t; self.tier[i] = tier; self.hits[i] = 1; self.active[i] = True
        self.rcs[i] = [rcs]
        return i

    def _fix_tier(self, i):
        """Acquisition rule on the estimate: same thresholds as tracking.acquisition_tiers."""
        v = self.x[i, 3:]; sp = float(np.hypot(v[0], v[1]))
        alt = float(self.x[i, 2]); r = np.array(self.rcs.pop(i, []), float)
        rcs = float(np.nanmedian(r)) if np.isfinite(r).any() else 0.0
        if sp >= self.acq["tier3_min_speed_mps"]:
            t = 3
        elif sp < self.acq["tier1_max_speed_mps"] and alt < self.acq["tier1_max_alt_m"] and rcs < self.acq["tier1_max_rcs_dbsm"]:
            t = 1
        else:
            t = 2
        self.tier[i] = t; self.tier_fixed[i] = True

    # ------------------------------------------------------------------ one radar look
    def _assign(self, idx, xp, Pp, Z, R, used):
        """Gate and assign reports (not yet used) to the tracks idx; returns list of (track, report)."""
        free = np.flatnonzero(~used)
        if len(idx) == 0 or len(free) == 0:
            return []
        HP = Pp[:, :3, :3]
        # coarse gate on Euclidean distance before the exact Mahalanobis test
        d = np.linalg.norm(xp[:, None, :3] - Z[None, free], axis=2)
        bound = np.sqrt(self.g["gate_chi2"] * (np.trace(HP, axis1=1, axis2=2)[:, None]
                                               + np.trace(R[free], axis1=1, axis2=2)[None, :]))
        ti, rj = np.nonzero(d <= bound)
        if len(ti) == 0:
            return []
        S = HP[ti] + R[free[rj]]
        r = Z[free[rj]] - xp[ti, :3]
        d2 = np.einsum("ki,ki->k", r, np.linalg.solve(S, r[..., None])[..., 0])
        ok = d2 <= self.g["gate_chi2"]
        if not ok.any():
            return []
        ti, rj, d2, S = ti[ok], rj[ok], d2[ok], S[ok]
        cost = d2 + np.log(np.linalg.det(S))
        ut, inv_t = np.unique(ti, return_inverse=True); ur, inv_r = np.unique(rj, return_inverse=True)
        C = np.full((len(ut), len(ur)), 1e12); C[inv_t, inv_r] = cost
        rows, cols = linear_sum_assignment(C)
        keep = C[rows, cols] < 1e11
        return [(int(ut[a]), int(free[ur[b]])) for a, b in zip(rows[keep], cols[keep])]

    def _update(self, pairs, idx_pos, xp, Pp, Z, R, t):
        for p, j in pairs:
            i = self._cur_idx[p]
            x, P = xp[p], Pp[p]
            S = P[:3, :3] + R[j]
            K = P[:, :3] @ np.linalg.inv(S)
            self.x[i] = x + K @ (Z[j] - x[:3])
            Pn = (np.eye(6) - K @ H) @ P
            self.P[i] = (Pn + Pn.T) / 2
            self.t[i] = t; self.hits[i] += 1
            if i in self.rcs:
                self.rcs[i].append(self._cur_rcs[j])

    def process_look(self, t, Z, R, sensor_type, rcs, vr=None):
        """Z (m, 3), R (m, 3, 3). Returns the internal track index assigned to each report (-1 none)."""
        out = np.full(len(Z), -1)
        self._expire(t)
        idx = np.flatnonzero(self.active[:self.n])
        used = np.zeros(len(Z), bool)
        self._cur_rcs = rcs
        if len(idx):
            dt = np.maximum(t - self.t[idx], 0.0)
            q = np.array([self.tiers[k]["q"] for k in self.tier[idx]])
            xp, Pp = _predict(self.x[idx], self.P[idx], q, dt)
            self._cur_idx = idx
            for group in (self.confirmed[idx], ~self.confirmed[idx]):     # confirmed tracks choose first
                sub = np.flatnonzero(group)
                pairs = self._assign(sub, xp[sub], Pp[sub], Z, R, used)
                pairs = [(sub[a], j) for a, j in pairs]
                self._update(pairs, None, xp, Pp, Z, R, t)
                for p, j in pairs:
                    used[j] = True; out[j] = idx[p]
        if self.g["protect_resolved_s"] > 0:
            took = out[out >= 0]
            self._mark_resolved(t, took[self.confirmed[took]])
        for j in np.flatnonzero(~used):
            out[j] = self._new_track(Z[j], R[j], t, sensor_type, rcs[j], np.nan if vr is None else vr[j])
        # confirmation and tier fixing
        live = np.flatnonzero(self.active[:self.n])
        for i in live:
            if not self.tier_fixed[i] and t - self.t0[i] >= self.acq["window_s"] and self.hits[i] >= 2:
                self._fix_tier(i)
            if not self.confirmed[i] and self.hits[i] >= self.g["confirm_hits"] \
                    and t - self.t0[i] <= self.g["confirm_window_s"]:
                self.confirmed[i] = True; self.t_conf[i] = t
        return out

    def _expire(self, t):
        idx = np.flatnonzero(self.active[:self.n])
        if not len(idx):
            return
        coast = np.array([self.tiers[k]["coast_s"] for k in self.tier[idx]])
        conf = self.confirmed[idx]
        # confirmed tracks: coast limit; tentative tracks: only the confirmation window
        dead = (conf & (t - self.t[idx] > coast)) | (~conf & (t - self.t0[idx] > self.g["confirm_window_s"]))
        for i in idx[dead]:
            self.active[i] = False; self.rcs.pop(i, None)

    def _mark_resolved(self, t, tracks):
        """Tracks that took different reports of one look are different objects (protect from dropping)."""
        if self.g["protect_resolved_s"] <= 0 or len(tracks) < 2:
            return
        tracks = np.unique(np.asarray(tracks, int))
        X = self.x[tracks, :3]
        D = np.linalg.norm(X[:, None] - X[None], axis=2)
        a, b = np.nonzero(np.triu(D < 5000.0, 1))           # only neighbours can ever look like duplicates
        for i, j in zip(tracks[a], tracks[b]):
            self.resolved[(int(i), int(j))] = t

    def _protected(self, a, b, t):
        if self.g["protect_resolved_s"] <= 0:
            return False
        k = (a, b) if a < b else (b, a)
        return t - self.resolved.get(k, -1e18) <= self.g["protect_resolved_s"]

    def _merge_duplicates(self, idx, xs, Ps):
        """Count consecutive duplicate checks per pair; delete the younger track of a persistent pair."""
        keep = np.ones(len(idx), bool)
        if len(idx) < 2:
            self.close = {}
            return keep
        d = np.linalg.norm(xs[:, None, :3] - xs[None, :, :3], axis=2)
        sig = np.sqrt(np.trace(Ps[:, :3, :3], axis1=1, axis2=2))
        a, b = np.nonzero(np.triu(d <= 4.0 * (sig[:, None] + sig[None, :]), 1))
        new = {}
        for i, j in zip(a, b):
            dx = xs[i] - xs[j]
            d2 = float(dx @ np.linalg.solve(Ps[i] + Ps[j], dx))
            if d2 <= self.g["merge_chi2"]:
                if self._protected(int(idx[i]), int(idx[j]), getattr(self, "_t_now", 0.0)):
                    self.n_protected += 1
                    continue
                old, yng = (i, j) if self.t0[idx[i]] <= self.t0[idx[j]] else (j, i)
                key = (int(idx[old]), int(idx[yng]))
                new[key] = self.close.get(key, 0) + 1
        self.close = {}
        for (o, y), c in new.items():
            if c >= self.g["merge_checks"] and self.active[o] and self.active[y]:
                self.active[y] = False; self.n_merged += 1
                keep[np.flatnonzero(idx == y)] = False
            else:
                self.close[(o, y)] = c
        return keep

    def snapshot(self, tg):
        """Confirmed live tracks predicted to grid time tg (no update); duplicate tracks are dropped."""
        self._t_now = tg
        idx = np.flatnonzero(self.active[:self.n] & self.confirmed[:self.n])
        if not len(idx):
            return []
        coast = np.array([self.tiers[k]["coast_s"] for k in self.tier[idx]])
        age = tg - self.t[idx]
        idx, age = idx[age <= coast], age[age <= coast]
        if not len(idx):
            return []
        q = np.array([self.tiers[k]["q"] for k in self.tier[idx]])
        xs, Ps = _predict(self.x[idx], self.P[idx], q, np.maximum(age, 0.0))
        keep = self._merge_duplicates(idx, xs, Ps)
        idx, age, xs, Ps = idx[keep], age[keep], xs[keep], Ps[keep]
        ps = np.sqrt(np.trace(Ps[:, :3, :3], axis1=1, axis2=2)); vs = np.sqrt(np.trace(Ps[:, 3:, 3:], axis1=1, axis2=2))
        if getattr(self, "cov_log", None) is not None:          # bench only: full position covariance
            for k, i in enumerate(idx):
                self.cov_log[(int(i) + self.g["track_id_base"], float(tg))] = Ps[k, :3, :3]
        return [(int(i), tg, *xs[k], float(ps[k]), float(vs[k]), float(age[k])) for k, i in enumerate(idx)]

    # ------------------------------------------------------------------ whole run
    def run(self, renu_all: pd.DataFrame, grid: np.ndarray):
        """renu_all: all radar reports in ENU (clutter included), columns time, sensor_id, sensor_type,
        e, n, u, R (or R_trk: R plus registration-bias covariance), rcs_dbsm. Returns (est, rep_tid, tracks)."""
        r = renu_all.sort_values(["time", "sensor_id"], kind="mergesort")
        Zall = r[["e", "n", "u"]].values; Rall = np.stack((r.R_trk if "R_trk" in r else r.R).values); rcs = r.rcs_dbsm.values.astype(float)
        vr = r.vr_mps.values.astype(float)
        times = r.time.values; sens = r.sensor_id.values; stype = r.sensor_type.values
        assigned = np.full(len(r), -1)
        rho_all = getattr(self, "_rho_sorted", None)
        est = []; gi = 0
        bounds = np.flatnonzero(np.r_[True, (times[1:] != times[:-1]) | (sens[1:] != sens[:-1]), True])
        T = self.T; t_run0 = __import__("time").perf_counter()
        for a, b in zip(bounds[:-1], bounds[1:]):
            t = times[a]
            while gi < len(grid) and grid[gi] < t:          # grid sample after all looks up to it
                T.sim_t = grid[gi]
                with T.cycle("snapshot"):
                    est += self.snapshot(grid[gi])
                gi += 1
            T.sim_t = t
            self._cur_slice = (a, b)                        # lets subclasses read per-report extras
            with T.cycle("look"):
                if rho_all is None:
                    assigned[a:b] = self.process_look(t, Zall[a:b], Rall[a:b], stype[a], rcs[a:b], vr[a:b])
                else:                                       # soft trackers also need the clutter density
                    assigned[a:b] = self.process_look(t, Zall[a:b], Rall[a:b], stype[a], rcs[a:b], vr[a:b], rho_all[a:b])
        while gi < len(grid):
            T.sim_t = grid[gi]
            with T.cycle("snapshot"):
                est += self.snapshot(grid[gi])
            gi += 1
        self.run_seconds = __import__("time").perf_counter() - t_run0
        if T.enabled:
            self.timing_report = T.close(self.run_seconds, f"{type(self).__name__} timing")
            print(self.timing_report)
            if T.log_path is not None:
                print(f"timing log: {T.log_path}")
        base = self.g["track_id_base"]
        conf = self.confirmed[:self.n]
        tid = np.where((assigned >= 0) & conf[np.maximum(assigned, 0)], assigned + base, -1)
        rep_tid = pd.Series(tid, index=r.index).reindex(renu_all.index)
        cols = ["eid", "t", "e", "n", "u", "ve", "vn", "vu", "pos_sigma_m", "vel_sigma_mps", "age_s"]
        est = pd.DataFrame(est, columns=cols)
        est["eid"] += base
        n = self.n
        tracks = pd.DataFrame({"eid": np.arange(n) + base, "t_first": self.t0[:n], "t_confirm": self.t_conf[:n],
                               "n_updates": self.hits[:n], "filter_tier": self.tier[:n], "confirmed": conf})
        tracks = tracks[tracks.confirmed].drop(columns="confirmed")
        last = est.groupby("eid").t.max()
        tracks["t_last"] = tracks.eid.map(last)
        return est.sort_values(["eid", "t"]).reset_index(drop=True), rep_tid, tracks.reset_index(drop=True)
