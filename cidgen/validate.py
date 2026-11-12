"""Generator validation (spec Section 10).

    python -m cidgen.validate runs/S0_baseline_seed1 runs/S1_zero_relational_seed1
    python -m cidgen.validate --determinism S0

Checks
  1. Sensor statistics: radar measurement residuals vs. stated sigma; Pd vs. range.
  2. Label-leakage audit: entity ids and report ordering carry no allegiance information.
  3. Relational check: accuracy gain from neighbour aggregates. S0 should show a gain;
     S1 (zero-relational) should show approximately none.
  4. Determinism: same scenario + seed gives identical tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_meta, read_table

OWN = ["speed", "alt", "climb", "turn_rate", "accel", "rad_rcs_med30", "rad_rcs_std30", "rad_md_rate",
       "rad_md_spread", "rad_md_period", "rad_jam_flag", "iff_mode3c", "iff_mode5_valid", "iff_level2",
       "iff_l2_resid_m", "iff_ever_valid", "iff_in_coverage", "jem_top_conf", "jem_bpf_feature",
       "lidar_len_fuselage_m", "lidar_len_span_m", "eoir_top_conf", "esm_conf", "esm_band", "esm_n_tracks_in_gate",
       "airway_xtrack_m", "airway_alt_inband", "corridor_dist_m", "asset_range_m", "asset_closure_mps",
       "asset_min_ttg_s", "asset_min_cpa_m", "descending_closing", "threat_offset_deg", "range_origin_m"]
NB = ["speed", "alt", "climb", "rad_rcs_med30", "iff_ever_valid", "iff_mode5_valid", "asset_closure_mps",
      "descending_closing", "esm_conf", "airway_xtrack_m"]


# --------------------------------------------------------------------------- 1. sensor statistics
def _truth_at(tr, eid, time):
    """Linearly interpolate truth position for (eid, time) pairs from the 1 s truth table."""
    base = tr.set_index(["eid", "t"])[["e", "n", "u"]]
    t0 = np.floor(time).astype(float); f = (time - t0)[:, None]
    a = base.reindex(pd.MultiIndex.from_arrays([eid, t0])).values
    b = base.reindex(pd.MultiIndex.from_arrays([eid, t0 + 1])).values
    b = np.where(np.isnan(b), a, b)
    return a * (1 - f) + b * f


def sensor_stats(run):
    r = read_table(run, "reports_radar"); tr = read_table(run, "truth_states"); sites = read_table(run, "sites")
    r = r[~r.false_alarm.astype(bool) & (r.n_merged == 1) & ~r.jam_flag.astype(bool)].reset_index(drop=True)
    P = _truth_at(tr, r.eid.values, r.time.values)
    ok = np.isfinite(P).all(1)
    r, P = r[ok], P[ok]
    sp = sites.set_index("site_id")
    d = P - sp.loc[r.sensor_id, ["e", "n", "u"]].values
    r = r.assign(dr=r.range_m.values - np.linalg.norm(d, axis=1),
                 daz=(r.az_deg.values - np.rad2deg(np.arctan2(d[:, 0], d[:, 1])) % 360 + 180) % 360 - 180)
    out = []
    for sid, g in r.groupby("sensor_id"):
        out.append(dict(sensor=sid, n=len(g), range_resid_std_m=round(float(g.dr.std()), 1),
                        stated_sigma_r_m=round(float(g.sigma_r_m.median()), 1),
                        range_bias_m=round(float(g.dr.median()), 1),
                        az_resid_std_deg=round(float(g.daz.std()), 3),
                        stated_sigma_az_deg=round(float(g.sigma_az_deg.median()), 3),
                        az_bias_deg=round(float(g.daz.median()), 3)))
    return pd.DataFrame(out)


def pd_vs_range(run, sensor_type="R_S", cls="A"):
    """Empirical Pd per scan for one class at one radar. A target counts as detected if it is in
    any report at that scan, including as a member of a merged (unresolved) report."""
    cfg = read_meta(run)["config"]
    rt = cfg["sensors"]["radar_types"][sensor_type]
    r = read_table(run, "reports_radar"); tr = read_table(run, "truth_states"); ent = read_table(run, "entities")
    sites = read_table(run, "sites")
    ids = set(ent[ent.cls == cls].eid)
    sid = sites[sites.kind == sensor_type].site_id.iloc[0]
    s0 = sites.set_index("site_id").loc[sid]
    tr = tr[tr.eid.isin(ids) & np.isclose(tr.t % rt["update_s"], 0)].copy()
    d = tr[["e", "n", "u"]].values - s0[["e", "n", "u"]].values.astype(float)
    tr["range_km"] = np.linalg.norm(d, axis=1) / 1000
    rr = r[(r.sensor_id == sid) & ~r.false_alarm.astype(bool)][["time", "merged_eids", "jam_flag"]].copy()
    rr["eids"] = rr.merged_eids.astype(str).str.split(";")
    rr = rr.explode("eids"); rr = rr[rr.eids != ""]
    rr["eid"] = rr.eids.astype(int); rr["t"] = rr.time.astype(float); rr["hit"] = 1
    m = tr.merge(rr[["eid", "t", "hit"]].drop_duplicates(), on=["eid", "t"], how="left").fillna({"hit": 0})
    bins = np.arange(0, rt["ref_range_km"] * rt["max_range_factor"] + 1, rt["ref_range_km"] / 5)
    m["bin"] = pd.cut(m.range_km, bins)
    return m.groupby("bin", observed=True).hit.agg(["mean", "size"]).rename(columns={"mean": "Pd_empirical", "size": "n"})


# --------------------------------------------------------------------------- 2. leakage audit
def leakage(run):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    lab = read_table(run, "labels"); lab = lab[lab.is_object_of_interest.astype(bool)]
    first = lab.groupby("eid").t.min().rename("first_seen")
    x = lab[["eid", "group_id", "allegiance"]].drop_duplicates("eid").join(first, on="eid")
    clf = HistGradientBoostingClassifier(max_iter=100)
    pred = cross_val_predict(clf, x[["eid"]], x.allegiance, groups=x.group_id, cv=GroupKFold(4))
    chance = 1 / x.allegiance.nunique()
    return dict(eid_only_bal_acc=round(float(balanced_accuracy_score(x.allegiance, pred)), 3), chance=round(chance, 3),
                n_entities=len(x))


# --------------------------------------------------------------------------- 3. relational check
def neighbour_frame(run):
    f = read_table(run, "features_track"); lab = read_table(run, "labels"); tt = read_table(run, "edges_track_track")
    snap = sorted(tt.t.unique())
    f = f[f.t.isin(snap)]
    lab = lab[lab.t.isin(snap)]
    df = f.merge(lab[["eid", "t", "allegiance", "group_id", "is_object_of_interest"]], on=["eid", "t"])
    nb = tt.merge(f[["eid", "t"] + NB], left_on=["dst_eid", "t"], right_on=["eid", "t"]).drop(columns="eid")
    agg = nb.groupby(["src_eid", "t"]).agg(**{f"nb_{c}": (c, "mean") for c in NB},
                                           nb_count=("dst_eid", "size"), nb_comove=("comove_corr", "mean"),
                                           nb_dist=("dist_m", "mean"), nb_speed_ratio=("speed_ratio", "mean"),
                                           nb_heading_diff=("heading_diff_deg", "mean")).reset_index()
    df = df.merge(agg, left_on=["eid", "t"], right_on=["src_eid", "t"], how="left").drop(columns="src_eid")
    df["nb_count"] = df.nb_count.fillna(0)
    return df[df.is_object_of_interest.astype(bool)]


def relational_gain(runs, seeds=3):
    """Pooled over runs of one scenario; folds split by (run, group) so formation mates and
    whole runs' worth of a behavior never straddle train and test."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score, log_loss
    from sklearn.model_selection import GroupKFold
    runs = [runs] if isinstance(runs, (str, Path)) else list(runs)
    frames = []
    for k, r in enumerate(runs):
        d = neighbour_frame(r); d["group_id"] = d.group_id.astype(int) + 100_000 * k
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    nbc = [c for c in df.columns if c.startswith("nb_")]
    y = df.allegiance.values
    res = {}
    for name, cols in (("own", OWN), ("own+neighbours", OWN + nbc)):
        accs, lls = [], []
        Xd = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        Xd = Xd.loc[:, Xd.nunique() > 1].fillna(-9999.0)    # trees split the sentinel as "missing"
        X = Xd.values
        for s in range(seeds):
            pred = np.zeros(len(df), int); prob = np.zeros((len(df), 3))
            # split by entity group so formation mates never straddle train and test
            gkf = GroupKFold(n_splits=4, shuffle=True, random_state=s)
            for tr_i, te_i in gkf.split(X, y, df.group_id):
                keep = np.array([len(np.unique(X[tr_i, c])) > 1 for c in range(X.shape[1])])
                clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=s)
                clf.fit(X[tr_i][:, keep], y[tr_i])
                pred[te_i] = clf.predict(X[te_i][:, keep])
                p = clf.predict_proba(X[te_i][:, keep]); full = np.zeros((len(te_i), 3)); full[:, clf.classes_] = p
                prob[te_i] = full
            accs.append(balanced_accuracy_score(y, pred)); lls.append(log_loss(y, np.clip(prob, 1e-6, 1), labels=[0, 1, 2]))
        res[name] = dict(bal_acc=round(float(np.mean(accs)), 3), log_loss=round(float(np.mean(lls)), 3))
    res["gain_bal_acc"] = round(res["own+neighbours"]["bal_acc"] - res["own"]["bal_acc"], 3)
    res["gain_log_loss"] = round(res["own"]["log_loss"] - res["own+neighbours"]["log_loss"], 3)
    res["n_rows"] = len(df)
    return res


# --------------------------------------------------------------------------- 4. determinism
def determinism(scenario="S0", seed=7, horizon=300):
    from .generate import run_scenario
    hashes = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as td:
            out = run_scenario(scenario, seed, td, {"run": {"horizon_s": horizon}}, fmt="csv", verbose=False)
            h = hashlib.sha256()
            for p in sorted(Path(out).glob("*.csv")):
                h.update(p.read_bytes())
            hashes.append(h.hexdigest()[:16])
    return dict(identical=hashes[0] == hashes[1], hashes=hashes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--determinism", default=None, help="scenario to test for determinism")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    report = {}
    for run in a.runs:
        name = Path(run).name
        print(f"\n=== {name} ===")
        ss = sensor_stats(run); print("Radar residuals vs stated sigma:\n", ss.to_string(index=False))
        pdr = pd_vs_range(run); print("R_S-1 Pd vs range, class A (design: 0.9 x 0.95 = 0.86 at ~99 km mean RCS):\n", pdr.round(3).to_string())
        lk = leakage(run); print("Leakage (eid only):", lk)
        report[name] = dict(sensor_stats=ss.to_dict("records"), leakage=lk)
    by_scn = {}
    for run in a.runs:
        by_scn.setdefault(read_meta(run)["scenario"], []).append(run)
    for scn, rs in by_scn.items():
        rg = relational_gain(rs); print(f"\nRelational check, {scn} pooled over {len(rs)} runs:", rg)
        report[f"relational_{scn}"] = rg
    if a.determinism:
        d = determinism(a.determinism); print("\nDeterminism:", d); report["determinism"] = d
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
