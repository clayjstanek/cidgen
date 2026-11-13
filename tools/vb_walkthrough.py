"""Traced run of the VB tracker on a tiny scenario, for the VB Tracker Code Walkthrough document.

    python tools/vb_walkthrough.py            # prints the trace (about 1 s)
    python tools/vb_walkthrough.py --out t.txt

Scenario: two aircraft flying north side by side, 200 m apart (2 radar sigmas), one radar (R_S) looking every
2 s with 100 m noise per axis, detection probability 0.9, and on average 2 clutter reports per look spread over
a 60 x 60 km box. The tracker code is not changed: the trace wraps a few functions from outside and records
what they were given and what they returned.
"""
from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cidgen.trackers.soft as soft  # noqa: E402
from cidgen.trackers import make_tracker  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=120)
SEP, SIGMA, PD, CLUTTER, DT, HORIZON = 200.0, 100.0, 0.9, 2, 2.0, 60.0
CFG = {"tracking": {"method": "vb", "soft": {"iterations": 5}, "gnn": {"protect_resolved_s": 30.0}},
       "sensors": {"clutter_rate": float(CLUTTER), "r_p_sector_halfwidth_deg": 60.0,
                   "radar_types": {"R_S": {"ref_range_km": 75.0, "sector": "full"}}}}


def scenario(seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for t in np.arange(0.0, HORIZON, DT):
        for k in range(2):                                   # aircraft 0 at e = 0, aircraft 1 at e = SEP
            if rng.random() < PD:
                z = np.array([k * SEP, -12000.0 + 200.0 * t, 5000.0]) + rng.normal(0, SIGMA, 3)
                rows.append((t, "R_S-1", "R_S", *z, 10.0, 0.0, k))
        for _ in range(rng.poisson(CLUTTER)):
            z = rng.uniform([-30000, -30000, 0], [30000, 30000, 1000])
            rows.append((t, "R_S-1", "R_S", *z, -20.0, 0.0, -1))
    r = pd.DataFrame(rows, columns=["time", "sensor_id", "sensor_type", "e", "n", "u", "rcs_dbsm", "vr_mps", "truth"])
    r["R"] = [np.eye(3) * SIGMA ** 2] * len(r)
    r["range_m"] = 50000.0; r["el_deg"] = 1.0             # every report 50 km out, 1 deg up (sets the clutter density)
    return r


# ---------------------------------------------------------------- tracing hooks (wrap, never edit, the tracker code)
TRACE = {"phase": None, "t": None, "marg": [], "pseudo": [], "solve": []}
_marginals, _pseudo, _soft_rts = soft.marginals, soft._pseudo, soft.soft_rts


def marginals_traced(W, exact_max=12):
    G = _marginals(W, exact_max)
    TRACE["marg"].append((TRACE["phase"], TRACE["t"], np.array(W), G))
    return G


def pseudo_traced(beta, Z, R, Ri=None):
    zt, Rt = _pseudo(beta, Z, R, Ri)
    if TRACE["phase"] == "look":
        TRACE["pseudo"].append((TRACE["t"], np.array(beta), np.array(Z), zt, Rt))
    return zt, Rt


soft.marginals, soft._pseudo = marginals_traced, pseudo_traced


class TracedVB(soft.SoftTracker):
    def _soft_look(self, t, idx, ci, xp, Pp, Z, R, rho, rcs, out, used):
        TRACE["phase"], TRACE["t"] = "look", t
        before = {int(idx[p]): (xp[p].copy(), Pp[p].copy()) for p in ci}
        n0 = len(TRACE["marg"])
        super()._soft_look(t, idx, ci, xp, Pp, Z, R, rho, rcs, out, used)
        TRACE.setdefault("looks", []).append(dict(t=t, Z=Z.copy(), rho=rho.copy(), before=before,
                                                  after={i: (self.x[i].copy(), self.P[i].copy()) for i in before},
                                                  marg=TRACE["marg"][n0:], out=out.copy(), used=used.copy()))
        TRACE["phase"] = None

    def _solve_cluster(self, members, t0):
        TRACE["phase"], TRACE["t"] = "window", t0
        n0 = len(TRACE["marg"])
        before = {i: (self.t[i], self.x[i].copy()) for i in members}
        hist = {i: [float(h[0]) for h in self.hist[i]] for i in members}
        mset = set(members)
        look_times = [float(L["t"]) for L in self.looks if mset.intersection(int(k) for k in L["tids"])]
        it0 = self.stats.get("window_iters", 0)
        super()._solve_cluster(members, t0)
        TRACE["solve"].append(dict(t0=t0, members=list(members), hist_times=hist, before=before,
                                   after={i: (self.t[i], self.x[i].copy()) for i in members},
                                   iters=self.stats.get("window_iters", 0) - it0, marg=TRACE["marg"][n0:],
                                   look_times=look_times))
        TRACE["phase"] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = scenario()
        print("=== 1. Input: radar reports (first 8 of", len(r), ")")
        print(r[["time", "sensor_id", "e", "n", "u", "rcs_dbsm", "truth"]].head(8).round(1).to_string(index=False))
        tr = TracedVB(CFG, "vb")
        print("\n=== 2. Construction")
        print("soft settings:", tr.s)
        print("gnn settings: ", tr.g)
        print("tier 2 filter settings:", tr.tiers[2])
        print(f"W offset ln(Pd/(1-Pd)) = {tr.wlog:.3f}")
        rho = soft.clutter_density(CFG, np.array(["R_S"]), np.array([50000.0]), np.array([1.0]))[0]
        print(f"clutter density at 50 km, 1 deg: rho = {rho:.3e} per m^3  (ln rho = {np.log(rho):.2f})")
        grid = np.arange(0.0, HORIZON, 2.0)
        est, rep, tracks = tr.run(r, grid)

        print("\n=== 3. Track life: one row per confirmed track (protection of resolved tracks on, 30 s)")
        print(tracks.to_string(index=False))
        cfg0 = {**CFG, "tracking": {"method": "vb", "soft": {"iterations": 5}}}
        _, _, tracks0 = soft.SoftTracker(cfg0, "vb").run(r, grid)
        print("same run with protection off (the plain VB default):")
        print(tracks0.to_string(index=False))
        # first soft look with a 2-track cluster sharing at least one report
        shared = [L for L in TRACE["looks"] for (ph, t, W, G) in L["marg"] if W.shape[0] >= 2 and (W > -1e3 + 1).sum(0).max() >= 2]
        L = shared[0]
        print(f"\n=== 4. A VB filter step: the look at t = {L['t']:.0f} s")
        print("reports in the look (e, n, u):"); print(L["Z"].round(1))
        print("ln rho per report:", np.log(L["rho"]).round(2))
        for i, (x, P) in L["before"].items():
            print(f"track {i + 1_000_000}: predicted position {x[:3].round(1)}, velocity {x[3:].round(1)}, "
                  f"position sigma {np.sqrt(np.trace(P[:3, :3])):.1f} m")
        for ph, t, W, G in L["marg"]:
            print("score matrix W (rows = tracks in the cluster, columns = gated reports; -1000 = outside the gate):")
            print(W.round(2))
            print("marginals G (last row = clutter; columns sum to 1):"); print(G.round(4))
            print("miss probability per track:", (1 - G[:-1].sum(1)).round(4))
        for t, beta, Z, zt, Rt in [p for p in TRACE["pseudo"] if p[0] == L["t"]]:
            print(f"pseudo-measurement: beta = {beta.round(4)}, mass = {beta.sum():.4f}")
            print(f"   z~ = {zt.round(1)},  sigma of R~ per axis = {np.sqrt(np.diag(Rt)).round(1)} m")
        for i, (x, P) in L["after"].items():
            print(f"track {i + 1_000_000}: updated position {x[:3].round(1)}, position sigma {np.sqrt(np.trace(P[:3, :3])):.1f} m")
        print("report -> track labels (-1 none):", L["out"], " used (not seeding new tracks):", L["used"])

        S = [s for s in TRACE["solve"] if s["t0"] + 20.0 > L["t"]][0]
        print(f"\n=== 5. A VB window solve: grid time {S['t0'] + 20:.0f} s, window from {S['t0']:.0f} s")
        print("cluster members:", [m + 1_000_000 for m in S["members"]], " iterations run:", S["iters"])
        for i in S["members"]:
            print(f"track {i + 1_000_000}: history times {S['hist_times'][i]}")
        nl = len(S["look_times"])
        for it in range(S["iters"]):
            for k, tl in enumerate(S["look_times"]):
                W, G = S["marg"][it * nl + k][2], S["marg"][it * nl + k][3]
                print(f"iteration {it + 1}, look at {tl:.0f} s: scores W =\n{W.round(2)}\n  marginals G =\n{G.round(4)}")
        for i in S["members"]:
            (tb, xb), (ta, xa) = S["before"][i], S["after"][i]
            print(f"track {i + 1_000_000}: filtered at {tb:.0f} s {xb[:3].round(1)} -> restarted from smoothed "
                  f"{xa[:3].round(1)} (moved {np.linalg.norm(xa[:3] - xb[:3]):.1f} m)")
        print("\nstats:", tr.stats)

        print("\n=== 6. Output: track estimates on the grid (first 6 rows)")
        print(est.head(6).round(1).to_string(index=False))
        print("report labels: first 12 values of rep_tid:", rep.values[:12])
        truth = r.truth.values
        for k in (0, 1):
            ids = rep.values[truth == k]; ids = ids[ids >= 0]
            print(f"aircraft {k}: reports labelled to tracks {sorted(set(ids.tolist()))}")
    text = buf.getvalue()
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
