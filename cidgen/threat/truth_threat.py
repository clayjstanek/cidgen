"""Truth threat for EVERY object at every snapshot time, tracked or not (evaluation only).

labels.threat_priority exists only for tracks, so a scoring loop that reads it alone cannot see a threat
the tracker has not confirmed yet, has dropped, or never held. This module recomputes the same quantity
(labels.threat_priority: capability * max_a value_a exp(-TTG/tau) 1{CPA < radius}, hostiles only) from the
stored truth_states, entities and assets tables, for all objects, so the threat list can be scored against
all real threats.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_table
from ..library import PLATFORMS


def first_detection(run_dir) -> pd.Series:
    """Time of each object's first radar report (true id; clutter excluded)."""
    r = read_table(run_dir, "reports_radar")[["time", "eid", "false_alarm"]]
    r = r[~r.false_alarm.astype(bool) & (r.eid >= 0)]
    return r.groupby("eid").time.min()


def _cover(run_dir, ts, cover_m):
    """Nearest track (features_track estimate) within cover_m of each true object, else -1."""
    f = read_table(run_dir, "features_track")[["eid", "t", "e", "n", "u"]]
    by_t = {t: g for t, g in f.groupby("t", sort=False)}
    out = np.full(len(ts), -1, np.int64)
    P = ts[["e", "n", "u"]].to_numpy()
    for t, idx in pd.Series(np.arange(len(ts))).groupby(ts.t.to_numpy()).groups.items():
        g = by_t.get(t)
        if g is None:
            continue
        D = np.linalg.norm(P[np.asarray(idx)][:, None] - g[["e", "n", "u"]].to_numpy()[None], axis=2)
        j = D.argmin(1); ok = D[np.arange(len(j)), j] <= cover_m
        out[np.asarray(idx)[ok]] = g.eid.to_numpy()[j[ok]]
    return out


def truth_threats(run_dir, times, cover_m=500.0) -> pd.DataFrame:
    """(truth_eid, t, threat_priority, detected) for hostile objects alive at the given times (same formula
    as labels). detected: the object has had at least one radar report by time t; scoring counts a real
    threat only from then on, so a threat no radar has seen yet is not charged to the tracker or to CID.
    cover_eid: the nearest track within cover_m of the object (-1 = none): with a real tracker, members of a
    group the radars cannot resolve (a drone swarm 50 m apart) share one group track rather than each having
    its own, and cover_eid credits the group track to all of them."""
    run_dir = Path(run_dir)
    tau = json.load(open(run_dir / "scenario_meta.json"))["config"]["labels"]["tau_s"]
    ent = read_table(run_dir, "entities")
    hos = ent[ent.allegiance == 2]
    cap = hos.set_index("eid").cls.map(lambda c: PLATFORMS[c].capability)
    ts = read_table(run_dir, "truth_states")
    ts = ts[ts.eid.isin(hos.eid) & ts.t.isin(np.asarray(times, float))]
    if ts.empty:
        return pd.DataFrame(columns=["truth_eid", "t", "threat_priority", "detected", "cover_eid"])
    assets = read_table(run_dir, "assets")
    p = ts[["e", "n"]].to_numpy(); v = ts[["ve", "vn"]].to_numpy()
    ok = np.isfinite(p).all(1) & np.isfinite(v).all(1)
    best = np.zeros(len(ts))
    vv = np.maximum((v * v).sum(1), 1e-6)
    for _, a in assets.iterrows():
        rel = p - np.array([a.e, a.n])
        r = np.linalg.norm(rel, axis=1)
        clos = -(rel * v).sum(1) / np.maximum(r, 1.0)
        consider = ~((clos <= 1.0) & (r > a.radius_m))
        ttg = np.where(clos > 1.0, r / np.where(clos > 1.0, clos, 1.0), 0.0)
        tc = np.maximum(-(rel * v).sum(1) / vv, 0.0)
        cpa = np.linalg.norm(rel + tc[:, None] * v, axis=1)
        val = np.where(consider & (cpa < a.radius_m), a.value * np.exp(-ttg / tau), 0.0)
        best = np.maximum(best, val)
    tp = np.where(ok, cap.loc[ts.eid].to_numpy() * best, 0.0)
    fd = first_detection(run_dir)
    det = ts.t.to_numpy() >= ts.eid.map(fd).fillna(np.inf).to_numpy()
    return pd.DataFrame({"truth_eid": ts.eid.to_numpy(), "t": ts.t.to_numpy(), "threat_priority": tp, "detected": det,
                         "cover_eid": _cover(run_dir, ts, cover_m)})
