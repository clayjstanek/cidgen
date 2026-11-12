"""How many aircraft is this track? A member-count estimate from radar RCS fluctuation (group tracks).

When radars cannot resolve a group (a drone swarm 50 m apart, a tight formation), the tracker holds ONE
track for several aircraft. Its reports are the sum of several independently fluctuating echoes, which
fluctuate less, relative to their mean, than one aircraft's echo: for n equal Swerling I members,
k = mean^2 / variance of the linear RCS is about n (lower in practice, because calibration error and aspect
changes add fluctuation of their own). k needs no aircraft type and no aspect, only several looks.

A second cue is stronger: when a group is partly resolved, a look holds several echoes close together.
near = 1 + the mean number of other echoes in the same radar look within cover_m of each of the track's
reports (near_free_only: count only echoes no confirmed track holds; on S12 this was the weaker choice).

Per track and snapshot time, over the last window_s seconds of the track's own radar reports:
  k_s       mean^2 / variance of linear RCS, per radar with at least min_looks_sensor reports
  k_hat     the look-weighted mean of k_s over radars; n_looks = reports used
  near      mean echoes within cover_m per look (1 = alone)
  n_hat     expected member count given (k_hat, near), from a calibration table fitted on training runs
            (binned conditional mean of the true group size); 1 when there are too few looks
True group size (calibration and scoring only): the number of true objects whose nearest track, within
cover_m, is this track.

    python -m cidgen.threat.group_count --fit "runs_gnn/S12_gnn_seed1" ... --test "runs_gnn/S12_gnn_seed18" ... \
        --out results_gnn/group_count.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_table
from ..models.common import expand

DEFAULTS = {"window_s": 20.0, "min_looks_sensor": 4, "min_looks": 6, "cover_m": 500.0, "k_bins": 4, "near_bins": 8,
            "near_free_only": False}


def _near_counts(r, sites, cover_m, free_only=False):
    """Echoes within cover_m in the same look, for every report."""
    from ..geometry import polar_to_enu
    sp = sites.loc[r.sensor_id, ["e", "n", "u"]].to_numpy()
    P = sp + polar_to_enu(r.range_m.to_numpy(), r.az_deg.to_numpy(), r.el_deg.to_numpy())
    out = np.ones(len(r))
    pos = np.arange(len(r))
    # echoes held by no confirmed track (tracker runs); a resolved neighbour has its own track and is
    # not counted, an unresolved member's intermittent extra echo usually is
    free = (r.track_id.to_numpy() < 0) if (free_only and "track_id" in r) else np.ones(len(r), bool)
    for _, idx in pd.Series(pos).groupby([r.time.to_numpy(), r.sensor_id.to_numpy()]).groups.items():
        idx = np.asarray(idx)
        if len(idx) > 1:
            Q = P[idx]
            close = np.linalg.norm(Q[:, None] - Q[None], axis=2) <= cover_m
            np.fill_diagonal(close, False)
            out[idx] = 1 + (close & free[idx][None, :]).sum(1)
    return out


def rcs_stats(run_dir, times, window_s=20.0, min_looks_sensor=4, cover_m=500.0, free_only=False):
    """(eid, t, k_hat, n_looks, near) for every track with reports in (t - window_s, t]."""
    r = read_table(run_dir, "reports_radar").reset_index(drop=True)
    sites = read_table(run_dir, "sites").set_index("site_id")
    r["near"] = _near_counts(r, sites, cover_m, free_only)
    tid = r.track_id if "track_id" in r else r.eid.where(~r.false_alarm.astype(bool), -1)
    r = r.assign(tid=tid.astype(np.int64), lin=10 ** (r.rcs_dbsm.astype(float) / 10))
    r = r[r.tid >= 0].sort_values(["tid", "sensor_id", "time"])
    times = np.sort(np.asarray(times, float))
    near_rows = []
    for tid, g in r.sort_values(["tid", "time"]).groupby("tid", sort=False):
        tt = g.time.to_numpy(); c = np.r_[0, np.cumsum(g.near.to_numpy())]
        lo = np.searchsorted(tt, times - window_s, side="right"); hi = np.searchsorted(tt, times, side="right")
        n = hi - lo; ok = n >= 1
        near_rows.append(pd.DataFrame({"eid": tid, "t": times[ok], "near": (c[hi[ok]] - c[lo[ok]]) / n[ok]}))
    near = pd.concat(near_rows, ignore_index=True) if near_rows else pd.DataFrame(columns=["eid", "t", "near"])
    times = np.sort(np.asarray(times, float))
    rows = []
    for (tid, _), g in r.groupby(["tid", "sensor_id"], sort=False):
        tt = g.time.to_numpy(); x = g.lin.to_numpy()
        c1 = np.r_[0, np.cumsum(x)]; c2 = np.r_[0, np.cumsum(x * x)]
        lo = np.searchsorted(tt, times - window_s, side="right"); hi = np.searchsorted(tt, times, side="right")
        n = hi - lo
        ok = n >= min_looks_sensor
        if not ok.any():
            continue
        s1 = c1[hi[ok]] - c1[lo[ok]]; s2 = c2[hi[ok]] - c2[lo[ok]]; nn = n[ok]
        m = s1 / nn; v = (s2 - nn * m * m) / (nn - 1)
        k = np.where(v > 0, m * m / np.maximum(v, 1e-300), np.nan)
        rows.append(pd.DataFrame({"eid": tid, "t": times[ok], "k": k, "n": nn}))
    if rows:
        d = pd.concat(rows, ignore_index=True).dropna()
        d["kn"] = d.k * d.n
        g = d.groupby(["eid", "t"], sort=False).agg(kn=("kn", "sum"), n_looks=("n", "sum")).reset_index()
        g["k_hat"] = g.kn / g.n_looks
        g = g.drop(columns="kn")
    else:
        g = pd.DataFrame(columns=["eid", "t", "n_looks", "k_hat"])
    return near.merge(g, on=["eid", "t"], how="left")


def true_group_size(run_dir, times, cover_m=500.0):
    """(eid, t, n_true): true objects (any kind) whose nearest track within cover_m is this track."""
    ts = read_table(run_dir, "truth_states")
    ts = ts[ts.t.isin(np.asarray(times, float))]
    f = read_table(run_dir, "features_track")[["eid", "t", "e", "n", "u"]]
    f = f[f.t.isin(np.asarray(times, float))]
    by_t = {t: g for t, g in f.groupby("t", sort=False)}
    out = []
    for t, s in ts.groupby("t", sort=False):
        g = by_t.get(t)
        if g is None:
            continue
        D = np.linalg.norm(s[["e", "n", "u"]].to_numpy()[:, None] - g[["e", "n", "u"]].to_numpy()[None], axis=2)
        j = D.argmin(1); ok = D[np.arange(len(j)), j] <= cover_m
        cnt = np.bincount(j[ok], minlength=len(g))
        out.append(pd.DataFrame({"eid": g.eid.to_numpy(), "t": t, "n_true": np.maximum(cnt, 0)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["eid", "t", "n_true"])


def _table(runs, s):
    rows = []
    for d in runs:
        times = np.unique(read_table(d, "features_track").t.to_numpy())
        times = times[np.isclose(times % 10.0, 0.0)]                      # threat-picture snapshot times
        st = rcs_stats(d, times, s["window_s"], s["min_looks_sensor"], s["cover_m"], s["near_free_only"])
        tr = true_group_size(d, times, s["cover_m"])
        rows.append(tr.merge(st, on=["eid", "t"], how="left").assign(run=Path(d).name))
    return pd.concat(rows, ignore_index=True)


def _bins(x, edges):
    return np.digitize(np.asarray(x, float), np.asarray(edges)[1:-1])


def _edges(x, nb):
    e = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    e[0], e[-1] = -np.inf, np.inf
    return e


def fit(runs, **kw):
    """Calibration table: mean true group size per (k_hat bin, near bin); k bin -1 = too few looks."""
    s = {**DEFAULTS, **kw}
    d = _table(runs, s)
    d = d[(d.n_true >= 1) & d.near.notna()].copy()
    okk = (d.n_looks.fillna(0) >= s["min_looks"]) & d.k_hat.notna()
    ke = _edges(d.k_hat[okk], s["k_bins"]); ne = _edges(d.near, s["near_bins"])
    d["kb"] = np.where(okk, _bins(d.k_hat.fillna(0), ke), -1); d["nb"] = _bins(d.near, ne)
    tab = d.groupby(["kb", "nb"]).n_true.agg(["mean", "size"]).reset_index()
    near_only = d.groupby("nb").n_true.mean()
    cal = dict(settings=s, k_edges=[float(e) for e in ke], near_edges=[float(e) for e in ne],
               table=[dict(kb=int(r.kb), nb=int(r.nb), mean=float(r["mean"]), n=int(r["size"])) for _, r in tab.iterrows()],
               near_only={int(k): float(v) for k, v in near_only.items()}, n_rows=int(len(d)))
    return cal


def apply(cal, k_hat, n_looks, near):
    s = cal["settings"]
    k = np.asarray(k_hat, float); n = np.asarray(n_looks, float); nr = np.asarray(near, float)
    okk = np.isfinite(k) & (np.nan_to_num(n) >= s["min_looks"])
    kb = np.where(okk, _bins(np.nan_to_num(k), cal["k_edges"]), -1)
    nb = _bins(np.nan_to_num(nr, nan=1.0), cal["near_edges"])
    lut = {(r["kb"], r["nb"]): r["mean"] for r in cal["table"] if r["n"] >= 20}
    out = np.array([lut.get((a, b), cal["near_only"].get(int(b), 1.0)) for a, b in zip(kb, nb)])
    out[~np.isfinite(nr)] = 1.0
    return np.maximum(out, 1.0)


def estimate(run_dir, times, cal):
    """(eid, t, k_hat, n_looks, n_hat) for the given snapshot times."""
    s = cal["settings"]
    st = rcs_stats(run_dir, times, s["window_s"], s["min_looks_sensor"], s["cover_m"], s.get("near_free_only", False))
    st["n_hat"] = apply(cal, st.k_hat, st.n_looks, st.near)
    return st


def accuracy(cal, runs):
    d = _table(runs, cal["settings"])
    d = d[d.n_true >= 1].copy()
    d["n_hat"] = apply(cal, d.k_hat, d.n_looks.fillna(0), d.near)
    d["grp"] = pd.cut(d.n_true, [0, 1, 2, 4, 1000], labels=["1", "2", "3-4", "5+"])
    try:
        from sklearn.metrics import roc_auc_score
        auc = {f"AUC n>={m}": round(float(roc_auc_score(d.n_true >= m, d.n_hat)), 3) for m in (2, 3, 5)}
    except Exception:
        auc = {}
    print(auc)
    t = d.groupby("grp", observed=True).agg(track_snapshots=("n_hat", "size"), n_hat_mean=("n_hat", "mean"),
                                             n_hat_median=("n_hat", "median"),
                                             called_group=("n_hat", lambda x: float((x >= 1.5).mean())))
    return t, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", nargs="+", required=True, help="calibration (training) runs")
    ap.add_argument("--test", nargs="+", default=None, help="runs to report accuracy on")
    ap.add_argument("--out", required=True, help="calibration json")
    a = ap.parse_args()
    cal = fit(expand(a.fit))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cal, open(a.out, "w"), indent=1)
    print(f"calibration on {cal['n_rows']:,} track-snapshots -> {a.out}")
    print("expected members by local-echo bin (upper edges", np.round(cal["near_edges"][1:-1], 2).tolist(), "):",
          {k: round(v, 2) for k, v in cal["near_only"].items()})
    if a.test:
        t, d = accuracy(cal, expand(a.test))
        pd.set_option("display.width", 160)
        print("\naccuracy on test runs (true group size = objects whose nearest track this is, within 500 m):")
        print(t.round(3).to_string())
        t.to_csv(Path(a.out).with_name("group_count_accuracy.csv"))


if __name__ == "__main__":
    main()
