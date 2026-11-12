"""The track picture: the interface between Combat ID and threat prioritisation.

One row per track per snapshot. Identity columns (p_friend, p_neutral, p_hostile) come from any CID
model; context columns come from the run's sensor-derived features and track-to-asset geometry.
Truth columns (allegiance, role, cls, threat_priority) are carried for evaluation only and are never
read by ThreatPrioritizer.

Identity sources
  truth            one-hot true allegiance: an upper bound (perfect Combat ID)
  <MODEL>          a column set '<MODEL>|Friend' ... in allpred_<name> (BN-learned, XGB-own, ...)
  <TAG>            a neural prediction file pred_<tag>_<name>.csv (GNN-SD30-NOTQ, GRU, ...)
  STACK:<a>+<b>+.. a stack of those members, fitted on validation rows (needs valpred_ files)
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_table
from ..models.bn import observable_tier
from ..models.common import eval_frame

CONTEXT = ["speed", "alt", "rad_rcs_med30", "iff_mode5_valid", "iff_ever_valid", "iff_mode3c", "corridor_dist_m",
           "airway_xtrack_m", "airway_alt_inband", "descending_closing", "threat_offset_deg", "asset_closure_mps",
           "esm_band", "asset_min_ttg_s", "asset_min_cpa_m"]
TRUTH = ["allegiance", "role", "cls", "threat_priority", "intent_phase", "truth_eid"]
CLS = ["Friend", "Neutral", "Hostile"]


def asset_geometry(run_dir, tau_s=300.0, cpa_margin=1.0, inside_ttg_s=0.0):
    """Per (eid, t): urgency against the most threatened asset, with that asset's TTG, CPA and value.
    Uses sensor-derived track-to-asset edges only."""
    ta = read_table(run_dir, "edges_track_asset")
    assets = read_table(run_dir, "assets").set_index("asset_id")
    ta = ta.join(assets[["radius_m"]], on="asset_id")
    inside = ta.range_m < ta.radius_m
    ttg = np.where(inside, inside_ttg_s, ta.ttg_s)
    hit = inside | ((ta.cpa_m < ta.radius_m * cpa_margin) & (ta.ttg_s < 1e5))
    ta = ta.assign(urgency_raw=np.where(hit, ta.asset_value * np.exp(-ttg / tau_s), 0.0), ttg_eff=ttg)
    ta = ta.sort_values("urgency_raw", ascending=False).drop_duplicates(["eid", "t"])
    return ta.rename(columns={"ttg_eff": "ttg_s_threat", "cpa_m": "cpa_m", "asset_value": "asset_value",
                              "radius_m": "asset_radius_m", "asset_id": "threatened_asset"})[
        ["eid", "t", "urgency_raw", "ttg_s_threat", "cpa_m", "asset_value", "asset_radius_m", "threatened_asset"]]


def build_frames(runs, doctrine):
    """Context + truth for every track-snapshot of the given runs (objects of interest, snapshot times)."""
    u = doctrine["urgency"]
    out = []
    for r in runs:
        f = eval_frame(r)
        g = asset_geometry(r, u["tau_s"], u.get("cpa_margin", 1.0), u.get("inside_radius_ttg_s", 0.0))
        f = f.merge(g, on=["eid", "t"], how="left")
        f["urgency_raw"] = f.urgency_raw.fillna(0.0)
        # time-to-go for rules: to the threatened asset if any, else the nearest-closing asset
        f["ttg_s"] = f.ttg_s_threat.fillna(pd.to_numeric(f.asset_min_ttg_s, errors="coerce"))
        f["obs_tier"] = observable_tier(f)
        if "truth_eid" not in f:                     # label-mode runs: the track IS the object
            f["truth_eid"] = f.eid
        keep = ["run", "eid", "t", "obs_tier", "urgency_raw", "ttg_s", "cpa_m", "asset_value", "asset_radius_m",
                "threatened_asset"] + [c for c in CONTEXT if c in f] + [c for c in TRUTH if c in f]
        out.append(f[keep])
    return pd.concat(out, ignore_index=True)


def _read_allpred(pred_dir, name):
    for ext in (".csv", ".parquet"):
        p = Path(pred_dir) / f"allpred_{name}{ext}"
        if p.exists():
            return pd.read_csv(p, low_memory=False) if ext == ".csv" else pd.read_parquet(p)
    return None


def identity(frame, source, pred_dir=None, name=None):
    """(p_friend, p_neutral, p_hostile) aligned to frame rows; NaN where the source has no prediction."""
    keys = frame[["run", "eid", "t"]]
    if source == "truth":
        P = np.eye(3)[frame.allegiance.astype(int).values]
        return P
    if source.startswith("STACK:"):
        return _stack(frame, source[6:].split("+"), pred_dir, name)
    ap = _read_allpred(pred_dir, name) if pred_dir else None
    if ap is not None and f"{source}|Friend" in ap:
        m = keys.merge(ap[ap.split == "test"][["run", "eid", "t"] + [f"{source}|{c}" for c in CLS]],
                       on=["run", "eid", "t"], how="left")
        return m[[f"{source}|{c}" for c in CLS]].values.astype(float)
    hits = glob.glob(str(Path(pred_dir) / f"pred_{source.lower()}_{name}.csv")) if pred_dir else []
    if hits:
        pr = pd.read_csv(hits[0])
        m = keys.merge(pr, on=["run", "eid", "t"], how="left")
        return m[["p_Friend", "p_Neutral", "p_Hostile"]].values.astype(float)
    raise SystemExit(f"identity source {source!r}: not 'truth', not a model in allpred_{name}, "
                     f"and no pred_{source.lower()}_{name}.csv in {pred_dir}")


def _stack(frame, members, pred_dir, name):
    from ..models.ensemble import Stacker, common_rows, load, norm
    fr, probs = load(pred_dir, name, log=lambda m: None)
    missing = [m for m in members if m not in probs]
    if missing:
        raise SystemExit(f"stack members {missing} lack validation + test predictions; have {sorted(probs)}")
    fr, P = common_rows(fr, probs, members, log=lambda m: None)
    st = Stacker().fit([P[m]["val"] for m in members], fr["val"].allegiance.values)
    q = st.predict([P[m]["test"] for m in members])
    t = fr["test"][["run", "eid", "t"]].assign(p0=q[:, 0], p1=q[:, 1], p2=q[:, 2])
    m = frame[["run", "eid", "t"]].merge(t, on=["run", "eid", "t"], how="left")
    return m[["p0", "p1", "p2"]].values.astype(float)


def pictures(frame, P):
    """Attach identity and yield (run, t, picture) for every snapshot, rows with identity only."""
    f = frame.assign(p_friend=P[:, 0], p_neutral=P[:, 1], p_hostile=P[:, 2])
    f = f[np.isfinite(P).all(1)]
    for (run, t), snap in f.groupby(["run", "t"], sort=True):
        yield run, t, snap.reset_index(drop=True)
