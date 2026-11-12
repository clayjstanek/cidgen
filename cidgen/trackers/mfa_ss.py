"""Stone Soup multi-frame assignment (MFA) for confirmed tracks: method `mfa`.

Stone Soup (Dstl) implements MFA over a sliding window of scans (Xia et al. 2019, solved with OR-Tools):
each track is a Gaussian mixture whose components are hypotheses tagged with their measurement history
(one entry per scan, 0 = missed), and at each scan the associator picks the jointly best histories over the
last `slide_window` scans and prunes the rest.

Here, as for JPDA and VB, cidgen keeps its own track management (tentative tracks, M-of-N confirmation,
coast deletion, duplicate dropping with protection of resolved tracks, observable tier) and MFA handles the
CONFIRMED tracks only:

  scan          one radar look (all reports of one radar at one time)
  clusters      confirmed tracks and the look's reports are linked by the chi-square gate (on each track's
                best hypothesis); each connected cluster is one small Stone Soup MFA call (the joint
                problem and its integer programme stay small); a track linked to no report is not sent
  detections    the cluster's reports, each with its own measurement model (its R)
  hypothesiser  Stone Soup PDAHypothesiser (P_D, chi-square gate, clutter density = the look's specified
                density, as for JPDA/VB) wrapped in MFAHypothesiser
  predictor     Kalman, constant velocity, with each track's own q (its tier's), via a small subclass
  alignment     every confirmed track gets exactly ONE history entry per look, numbered by the report's
                position in the LOOK (not in its cluster's call), so histories from different calls agree: tracks with no report near
                the look are not sent to Stone Soup but are given the "missed" entry directly (that is all
                Stone Soup would give them); new tracks start with slide_window "missed" entries; histories
                are cut to the last 2 x slide_window entries (N-scan pruning has already settled older ones)
  estimate      the moment-matched mixture of the track's kept hypotheses (weights from Stone Soup) is the
                track's state for features, gating of tentative tracks, duplicate dropping and output
  reports       a report the confirmed tracks explain with total probability >= 0.5 is taken (labelled with
                the most likely track); the rest go on to tentative tracks and track initiation, as JPDA/VB

Needs `stonesoup` and `ortools` (pip install stonesoup ortools). Settings: tracking.mfa.
"""
from __future__ import annotations

import datetime

import numpy as np
from scipy.stats import chi2

from .gnn import _predict
from .soft import SoftTracker

MFA_DEFAULTS = {
    "slide_window": 3,          # scans (radar looks) jointly optimised
    "p_detect": 0.9,
    "max_components": 30,       # keep each track's highest-weight hypotheses, PLUS always its "missed since
                                # the window start" hypothesis, which never conflicts with another track, so
                                # every look stays solvable (0 = no cap; can explode in a swarm)
    "tag_keep": 2,              # keep tag_keep x slide_window history entries per hypothesis
    "min_weight": 1e-200,       # hypothesis weights below this are dropped (detections) or floored (missed)
}
_T0 = datetime.datetime(2000, 1, 1)
# cidgen state (e, n, u, ve, vn, vu) <-> Stone Soup constant-velocity order (e, ve, n, vn, u, vu)
_TO_SS = np.array([0, 3, 1, 4, 2, 5])
_FROM_SS = np.argsort(_TO_SS)


def psd_repair(P, rel=1e-12, abs_floor=1e-9):
    """Nearest valid covariance: symmetrise, then raise any eigenvalue below max(rel x largest, abs_floor)
    to that floor. Returns (P_fixed, min_eigenvalue_before / largest, repaired?). Non-finite input: (None, nan, True)."""
    P = np.asarray(P, float)
    if not np.isfinite(P).all():
        return None, np.nan, True
    S = (P + P.T) / 2
    w, V = np.linalg.eigh(S)
    top = max(float(w.max()), abs_floor)
    lo = max(rel * top, abs_floor)
    if w.min() >= lo:
        return S, float(w.min()) / top, False
    return (V * np.maximum(w, lo)) @ V.T, float(w.min()) / top, True


def mfa_settings(cfg):
    s = {**MFA_DEFAULTS, **((cfg.get("tracking") or {}).get("mfa") or {})}
    unknown = set(s) - set(MFA_DEFAULTS)
    if unknown:
        raise KeyError(f"tracking.mfa: unknown keys {sorted(unknown)}")
    return s


class MFATracker(SoftTracker):
    def __init__(self, cfg, debug=0, timing_log=None):
        super().__init__(cfg, "jpda", debug, timing_log)     # JPDA plumbing for everything but confirmed tracks
        self.mode = "mfa"
        try:
            from stonesoup.dataassociator.mfa import MFADataAssociator
            from stonesoup.hypothesiser.mfa import MFAHypothesiser
            from stonesoup.hypothesiser.probability import PDAHypothesiser
            from stonesoup.models.measurement.linear import LinearGaussian
            from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
            from stonesoup.predictor.kalman import KalmanPredictor
            from stonesoup.types.detection import Detection
            from stonesoup.types.mixture import GaussianMixture
            from stonesoup.types.numeric import Probability
            from stonesoup.types.state import TaggedWeightedGaussianState
            from stonesoup.types.track import Track
            from stonesoup.types.update import GaussianMixtureUpdate
            from stonesoup.updater.kalman import KalmanUpdater
        except ImportError as e:
            raise ImportError("tracking.method: mfa needs Stone Soup with OR-Tools: pip install stonesoup ortools") from e
        self.ss = dict(Detection=Detection, GaussianMixture=GaussianMixture, Probability=Probability,
                       TWGS=TaggedWeightedGaussianState, Track=Track, GMU=GaussianMixtureUpdate,
                       LinearGaussian=LinearGaussian)
        self.mf = mfa_settings(cfg)
        qs = sorted({float(v["q"]) for v in self.tiers.values()})
        models = {q: CombinedLinearGaussianTransitionModel([ConstantVelocity(q)] * 3) for q in qs}

        from stonesoup.types.array import CovarianceMatrix
        stats = self.stats
        stats.update(covar_fix_predict=0, covar_fix_update=0, covar_fix_mixture=0, covar_fix_input=0,
                     covar_nonfinite=0, ss_negative_dt=0)

        def _guard(state, key, fallback=None):
            """Make state.covar a valid covariance; count and remember the worst case per source."""
            P, worst, bad = psd_repair(state.covar)
            if P is None:
                stats["covar_nonfinite"] += 1
                if fallback is None:
                    return state
                P, _, _ = psd_repair(fallback)
                if P is None:
                    return state
            if bad:
                stats[key] += 1
                if np.isfinite(worst):
                    stats[key + "_worst"] = min(stats.get(key + "_worst", 0.0), round(worst, 6))
            state.covar = CovarianceMatrix(P)
            return state

        class TierPredictor(KalmanPredictor):
            """One predictor, each component predicted with its own track's q (attribute cid_q).
            Guarded: a prediction backwards in time is counted (ss_negative_dt), and the predicted
            covariance is always returned as a valid covariance (covar_fix_predict counts repairs)."""
            def predict(self, prior, timestamp=None, **kwargs):
                self.transition_model = models[getattr(prior, "cid_q", qs[-1])]
                if timestamp is not None and getattr(prior, "timestamp", None) is not None:
                    dt = (timestamp - prior.timestamp).total_seconds()
                    if dt < 0:
                        stats["ss_negative_dt"] += 1
                        stats["ss_negative_dt_worst_s"] = min(stats.get("ss_negative_dt_worst_s", 0.0), dt)
                return _guard(super().predict(prior, timestamp=timestamp, **kwargs), "covar_fix_predict")

        class SafeKalmanUpdater(KalmanUpdater):
            """Stone Soup's update uses P - K S K^T, which can lose positive-definiteness numerically; the
            posterior covariance is returned as a valid covariance (covar_fix_update counts repairs)."""
            def update(self, hypothesis, **kwargs):
                post = super().update(hypothesis, **kwargs)
                return _guard(post, "covar_fix_update", fallback=hypothesis.prediction.covar)
        self._guard = _guard
        self._models = models
        self.predictor = TierPredictor(models[qs[-1]])
        self.updater = SafeKalmanUpdater(LinearGaussian(6, (0, 2, 4), np.eye(3)))
        self.pda = PDAHypothesiser(self.predictor, self.updater, 1e-12,
                                   prob_gate=chi2.cdf(self.g["gate_chi2"], 3), prob_detect=self.mf["p_detect"])
        from stonesoup.types.multihypothesis import MultipleHypothesis
        from stonesoup.types.prediction import Prediction
        floor = self.mf["min_weight"]

        class SafeMFAHypothesiser(MFAHypothesiser):
            """Stone Soup scores each hypothesis with cost = -log(weight). A weight that underflows to 0 in
            double precision (very unlikely pairings inside a wide Tier 3 gate) gives an infinite cost,
            and Stone Soup's solver then fails ("objective value is infinite", "infeasible primal").
            Detection hypotheses below min_weight are dropped (they could never be chosen); a missed
            hypothesis that small is kept at min_weight (it may be needed for a valid joint solution)."""
            def hypothesise(self, track, detections, timestamp, detections_tuple, **kwargs):
                hs = super().hypothesise(track, detections, timestamp, detections_tuple, **kwargs)
                keep = []
                for h in hs:
                    w = float(h.prediction.weight)
                    if np.isfinite(w) and w >= floor:
                        keep.append(h)
                    elif not h:                                  # missed: keep, floored
                        h.prediction = Prediction.from_state(h.prediction, weight=Probability(floor))
                        keep.append(h); stats["floored"] = stats.get("floored", 0) + 1
                    else:
                        stats["dropped_tiny"] = stats.get("dropped_tiny", 0) + 1
                return MultipleHypothesis(keep)
        self.assoc = MFADataAssociator(SafeMFAHypothesiser(self.pda), slide_window=self.mf["slide_window"])
        self.sstracks = {}                                   # cidgen track index -> Stone Soup Track
        self.stats.update(mfa_calls=0, mfa_tracks=0, mfa_dets=0, padded=0, max_components=0)

    # ------------------------------------------------------------------ helpers
    def _ts(self, t):
        return _T0 + datetime.timedelta(seconds=float(t))

    def _component(self, x, P, t, weight, tag, q):
        Pf, worst, bad = psd_repair(P)
        if Pf is None:
            self.stats["covar_nonfinite"] += 1
            Pf = np.diag([1e6] * 3 + [1e6] * 3)                     # unusable state: restart wide
        elif bad:
            self.stats["covar_fix_input"] += 1
        P = Pf
        c = self.ss["TWGS"](x[_TO_SS].reshape(-1, 1), P[np.ix_(_TO_SS, _TO_SS)], timestamp=self._ts(t),
                            weight=self.ss["Probability"](weight), tag=list(tag))
        c.cid_q = q
        return c

    def _start(self, i, t):
        K = self.mf["slide_window"] * self.mf["tag_keep"]
        comp = self._component(self.x[i], self.P[i], t, 1.0, [0] * K, float(self.tiers[self.tier[i]]["q"]))
        self.sstracks[i] = self.ss["Track"]([self.ss["GaussianMixture"]([comp])])

    def _best(self, i):
        comps = list(self.sstracks[i].state.components)
        return max(comps, key=lambda c: float(c.weight))

    def _set_from_best(self, i):
        b = self._best(i)
        self.x[i] = np.ravel(b.state_vector)[_FROM_SS].astype(float)
        self.P[i] = np.asarray(b.covar, float)[np.ix_(_FROM_SS, _FROM_SS)]
        self.t[i] = (b.timestamp - _T0).total_seconds()

    def _pad(self, i):
        """No report near this look: the "missed" history entry, without calling Stone Soup."""
        K = self.mf["slide_window"] * self.mf["tag_keep"]
        for c in self.sstracks[i].state.components:
            c.tag = (list(c.tag) + [0])[-K:]
        self.stats["padded"] += 1

    # ------------------------------------------------------------------ life of tracks
    def _expire(self, t):
        super()._expire(t)
        for i in [k for k in self.sstracks if not self.active[k]]:
            self.sstracks.pop(i, None)

    def _confirm_loop(self, t):
        before = set(np.flatnonzero(self.confirmed[:self.n]))
        super()._confirm_loop(t)
        for i in np.flatnonzero(self.confirmed[:self.n] & self.active[:self.n]):
            if i not in before and i not in self.sstracks:
                self._start(int(i), t)

    # ------------------------------------------------------------------ one look, confirmed tracks
    def _soft_look(self, t, idx, ci, xp, Pp, Z, R, rho, rcs, out, used):
        tm = self.T
        live = [int(idx[p]) for p in ci]
        for i in live:                                     # confirmed before MFA existed for it
            if i not in self.sstracks:
                self._start(i, self.t[i])
        with tm("look/mfa/cluster"):
            # tracks and reports linked by the chi-square gate on each track's best hypothesis; clusters
            # are solved separately (small joint problems), isolated tracks included
            HP = Pp[ci, :3, :3]
            dist = np.linalg.norm(xp[ci, None, :3] - Z[None], axis=2)
            bound = np.sqrt(self.g["gate_chi2"] * (np.trace(HP, axis1=1, axis2=2)[:, None]
                                                   + np.trace(R, axis1=1, axis2=2)[None, :]))
            a, b = np.nonzero(dist <= bound)
            if len(a):
                S = HP[a] + R[b]; r = Z[b] - xp[ci[a], :3]
                d2 = np.einsum("ki,ki->k", r, np.linalg.solve(S, r[..., None])[..., 0])
                keep = d2 <= self.g["gate_chi2"]
                a, b = a[keep], b[keep]
            nT, nZ = len(ci), len(Z)
            linked = np.zeros(nT, bool); linked[a] = True
            clusters = []
            if len(a):
                from scipy.sparse import coo_matrix
                from scipy.sparse.csgraph import connected_components
                adj = coo_matrix((np.ones(len(a)), (a, nT + b)), shape=(nT + nZ, nT + nZ))
                _, lab = connected_components(adj, directed=False)
                for c in np.unique(lab[a]):
                    clusters.append((np.flatnonzero(lab[:nT] == c), np.flatnonzero(lab[nT:] == c)))
        for k, i in enumerate(live):
            if not linked[k]:
                self._pad(i)                               # nothing near: the "missed" entry
        if not clusters:
            return
        self.pda.clutter_spatial_density = float(np.median(rho))
        K = self.mf["slide_window"] * self.mf["tag_keep"]; winners = []
        report_mass, track_mass = {}, {}
        for T_, J_ in clusters:
            with tm("look/mfa/build"):
                dets = {self.ss["Detection"](Z[j].reshape(-1, 1), timestamp=self._ts(t),
                                             measurement_model=self.ss["LinearGaussian"](6, (0, 2, 4), R[j]),
                                             metadata={"j": int(j)}) for j in J_}
                tracks = {self.sstracks[live[k]]: live[k] for k in T_}
            with tm("look/mfa/associate"):
                hyps = self._associate(tracks, dets, t)
            self.stats["mfa_calls"] += 1; self.stats["mfa_tracks"] += len(tracks); self.stats["mfa_dets"] += len(dets)
            self.stats["max_cluster_tracks"] = max(self.stats["max_cluster_tracks"], len(tracks))
            with tm("look/mfa/update"):
                for track, hs in hyps.items():
                    i = tracks[track]; q = float(self.tiers[self.tier[i]]["q"])
                    comps, used_j = [], []
                    for h in hs:
                        c = h.prediction if not h else self.updater.update(h)
                        c.cid_q = q
                        j = None if not h else h.measurement.metadata.get("j")
                        # history entry for this look: the report's number IN THE LOOK (j + 1), not its
                        # position in this cluster's call, so histories from different clusters agree
                        c.tag = (list(c.tag)[:-1] + [0 if j is None else j + 1])[-K:]
                        comps.append(c); used_j.append(j)
                    order = np.argsort([-float(c.weight) for c in comps])
                    if self.mf["max_components"] > 0 and len(order) > self.mf["max_components"]:
                        W = self.mf["slide_window"]
                        # the safe branch: missed at every look after the window start (always present
                        # after Stone Soup's N-scan pruning, and conflict-free at the next look)
                        safe = [k for k in order if all(v == 0 for v in list(comps[k].tag)[-(W - 1):])]
                        top = list(order[: self.mf["max_components"]])
                        order = top + [k for k in safe[:1] if k not in top]
                        self.stats["capped"] = self.stats.get("capped", 0) + 1
                    comps = [comps[k] for k in order]; used_j = [used_j[k] for k in order]
                    w = np.array([float(c.weight) for c in comps])
                    w = np.where(np.isfinite(w), w, 0.0); w = w / (w.sum() or 1.0)
                    w = np.maximum(w, 1e-12); w = w / w.sum()
                    for c, wk in zip(comps, w):                     # stored weights stay well inside range
                        c.weight = self.ss["Probability"](float(wk))
                    det_mass = float(sum(wk for wk, j in zip(w, used_j) if j is not None))
                    track.append(self.ss["GMU"](components=comps, hypothesis=hs))
                    self.stats["max_components"] = max(self.stats["max_components"], len(comps))
                    self._set_from_mixture(i, comps, w)
                    if det_mass >= self.s["min_assoc_mass"]:
                        self.t_hit[i] = t; self.hits[i] += 1
                    # probability that this track produced each report (summed over its hypotheses)
                    for wk, j in zip(w, used_j):
                        if j is not None:
                            report_mass[j] = report_mass.get(j, 0.0) + wk
                            track_mass[(i, j)] = track_mass.get((i, j), 0.0) + wk
        # a report the confirmed tracks explain with probability >= 0.5 is taken (as for JPDA / VB): it
        # does not seed a new track; it is labelled with the track most likely to have produced it
        best_for = {}
        for (i, j), m in track_mass.items():
            if m > best_for.get(j, (None, 0.0))[1]:
                best_for[j] = (i, m)
        for j, m in report_mass.items():
            if m >= 0.5:
                used[j] = True
            i, mi = best_for[j]
            if mi >= 0.5:
                out[j] = i; winners.append(i)
                if i in self.rcs:
                    self.rcs[i].append(float(rcs[j]))
        self._mark_resolved(t, winners)

    def _set_from_mixture(self, i, comps, w):
        """Track state = the moment-matched mixture of its kept hypotheses (all at the look time)."""
        X = np.array([np.ravel(c.state_vector)[_FROM_SS].astype(float) for c in comps])
        Ps = np.array([np.asarray(c.covar, float)[np.ix_(_FROM_SS, _FROM_SS)] for c in comps])
        m = w @ X; d = X - m
        P = np.einsum("k,kab->ab", w, Ps) + np.einsum("k,ka,kb->ab", w, d, d)
        Pf, worst, bad = psd_repair(P)
        if Pf is None:                                              # keep the last valid state
            self.stats["covar_nonfinite"] += 1
            return
        if bad:
            self.stats["covar_fix_mixture"] += 1
        self.x[i] = m; self.P[i] = Pf
        self.t[i] = (comps[0].timestamp - _T0).total_seconds()

    def _associate(self, tracks, dets, t):
        """Stone Soup MFA on one cluster. If its solver fails (infeasible, or a numerical failure inside
        Stone Soup), collapse these tracks to their single best hypothesis (history reset to "missed") and
        solve again; if that fails too, the tracks are treated as missed at this look. Counted in stats."""
        import warnings
        for attempt in (0, 1):
            try:
                with warnings.catch_warnings(record=True) as wl:
                    warnings.simplefilter("always")
                    res = self.assoc.associate(set(tracks), dets, self._ts(t))
                n_fb = sum("Infeasible primal" in str(w.message) for w in wl)
                if n_fb:
                    self.stats["ss_primal_fallback"] = self.stats.get("ss_primal_fallback", 0) + n_fb
                return res
            except (ValueError, RuntimeError, IndexError) as e:
                key = "mfa_failed_reset" if attempt == 0 else "mfa_failed_missed"
                self.stats[key] = self.stats.get(key, 0) + 1
                self.stats.setdefault("mfa_last_error", repr(e)[:120])
                if attempt == 0:
                    K = self.mf["slide_window"] * self.mf["tag_keep"]
                    for track, i in tracks.items():
                        b = self._best(i)
                        track.append(self.ss["GaussianMixture"]([self._component(
                            np.ravel(b.state_vector)[_FROM_SS].astype(float),
                            np.asarray(b.covar, float)[np.ix_(_FROM_SS, _FROM_SS)],
                            (b.timestamp - _T0).total_seconds(), 1.0, [0] * K, float(self.tiers[self.tier[i]]["q"]))]))
        for i in tracks.values():                                 # give up on this look: all missed
            self._pad(i)
        return {}

    def snapshot(self, tg):
        self._t_now = tg
        with self.T("snapshot/output"):
            res = self._snapshot_out(tg)
        for i in [k for k in self.sstracks if not self.active[k]]:   # dropped as duplicates
            self.sstracks.pop(i, None)
        return res
