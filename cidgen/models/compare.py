"""Score every model family on the same held-out runs and write one comparison table per scenario.

    python -m cidgen.models.compare --runs "runs/S0_*" "runs/S1_*" --preds results --out results
    python -m cidgen.models.compare --runs "runs/S0_*" --test-runs "runs/S9_*" --preds results --out results
    ... --save-preds      also write allpred_<name> (validation + test predictions) for models/ensemble.py

Models
  BN-learned, BN-oracle, BN-SME, BN-context   (models/bn.py)
  Trees-own, Trees-own+nb                      XGBoost if installed, else sklearn HistGradientBoosting
  GRU, GNN                                     read from results/pred_*.csv written by train_gnn.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_meta
from .bn import TieredBN, observable_tier
from .common import bootstrap, eval_frame, expand, metrics_table, split_runs

TREE_OWN = ["speed", "alt", "climb", "turn_rate", "accel", "rad_rcs_med30", "rad_rcs_std30", "rad_md_rate",
            "rad_md_spread", "rad_md_period", "rad_jam_flag", "iff_mode3c", "iff_mode5", "iff_mode5_valid",
            "iff_level2", "iff_l2_resid_m", "iff_ever_valid", "iff_in_coverage", "jem_top_conf", "jem_bpf_feature",
            "lidar_len_fuselage_m", "lidar_len_span_m", "lidar_len_tail_m", "eoir_top_conf", "eoir_ir_intensity",
            "esm_conf", "esm_band", "esm_n_tracks_in_gate", "airway_xtrack_m", "airway_alt_inband",
            "corridor_dist_m", "asset_range_m", "asset_closure_mps", "asset_min_ttg_s", "asset_min_cpa_m",
            "descending_closing", "threat_offset_deg", "range_origin_m"]
TREE_CAT = ["jem_top_engine", "esm_top_emitter", "eoir_top_class"]
# Tracker metadata the neural models receive but the trees and BNs normally do not. "XGB-own+tq" adds
# them to the trees, so the families can be compared on identical inputs.
TRACK_QUALITY = ["pos_sigma_m", "vel_sigma_mps", "age_s"]


def tree_matrix(df, with_nb, with_tq=False):
    X = df[TREE_OWN + (TRACK_QUALITY if with_tq else [])].apply(pd.to_numeric, errors="coerce")
    for c in TREE_CAT:
        X[c] = df[c].astype("category").cat.codes.replace(-1, np.nan) if c in df else np.nan
    if with_nb:
        nb = [c for c in df.columns if c.startswith("nb_")]
        X = pd.concat([X, df[nb].apply(pd.to_numeric, errors="coerce")], axis=1)
    return X.replace([np.inf, -np.inf], np.nan).astype(float)


def fit_trees(tr, te, with_nb, seed=0, with_tq=False, extra=()):
    """Fit on tr, predict te (and any extra frames, e.g. validation). Returns (name, p_te[, [p_extra]])."""
    Xtr, Xte = tree_matrix(tr, with_nb, with_tq), tree_matrix(te, with_nb, with_tq)
    Xex = [tree_matrix(e, with_nb, with_tq) for e in extra]
    # align categorical codes between train and test (extra frames use the same codes; unseen -> -1)
    for c in TREE_CAT:
        cats = pd.Categorical(pd.concat([tr[c], te[c]]).astype(str)).categories
        Xtr[c] = pd.Categorical(tr[c].astype(str), categories=cats).codes
        Xte[c] = pd.Categorical(te[c].astype(str), categories=cats).codes
        for X, e in zip(Xex, extra):
            X[c] = pd.Categorical(e[c].astype(str), categories=cats).codes
    y = tr.allegiance.values
    w = (len(y) / (3 * np.maximum(np.bincount(y, minlength=3), 1)))[y]
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.06, subsample=0.8,
                                colsample_bytree=0.8, tree_method="hist", random_state=seed)
        name = "XGB"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=seed)
        name = "HGB"
    clf.fit(Xtr.values, y, sample_weight=w)

    def proba(X):
        p = np.zeros((len(X), 3)); p[:, clf.classes_] = clf.predict_proba(X.values)
        return p
    if extra:
        return name, proba(Xte), [proba(X) for X in Xex]
    return name, proba(Xte)


ROLE_NAMES = {"own_uas": "own drones", "rtb": "returning on corridor", "tactical_rtb": "tactical return",
              "cap": "combat air patrol", "aew": "AEW", "helo": "helicopters"}


def breakdowns(res):
    """Per-model diagnostics: by tier, friendly-as-hostile by friendly role, and own drones split by
    whether the EO/IR camera had reported on them."""
    tier_rows, role_rows, eo_rows = [], [], []
    for name, (p, d) in res.items():
        pred = np.asarray(p).argmax(1); y = d.allegiance.values
        for t in sorted(d.tier.unique()):
            m = d.tier.values == t
            rec = [((pred[m] == k) & (y[m] == k)).sum() / max((y[m] == k).sum(), 1) if (y[m] == k).any() else np.nan
                   for k in range(3)]
            fr = m & (y == 0)
            tier_rows.append(dict(model=name, tier=int(t), snapshots=int(m.sum()), bal_acc=float(np.nanmean(rec)),
                                  friend_called_hostile=float((pred[fr] == 2).mean()) if fr.any() else np.nan,
                                  missed_hostile=float(1 - rec[2]) if not np.isnan(rec[2]) else np.nan))
        fr = y == 0
        r = {"model": name}
        for role in sorted(d.role[fr].unique()):
            m = fr & (d.role.values == role)
            r[ROLE_NAMES.get(role, role)] = float((pred[m] == 2).mean())
        role_rows.append(r)
        ou = fr & (d.role.values == "own_uas")
        if ou.any():
            seen = d.eoir_top_class.notna().values
            eo_rows.append(dict(model=name, own_drone_snapshots=int(ou.sum()),
                                share_seen_by_eoir=float(seen[ou].mean()),
                                called_hostile_all=float((pred[ou] == 2).mean()),
                                called_hostile_eoir_seen=float((pred[ou & seen] == 2).mean()) if (ou & seen).any() else np.nan,
                                called_hostile_eoir_unseen=float((pred[ou & ~seen] == 2).mean()) if (ou & ~seen).any() else np.nan))
    as_table = lambda rows: (pd.DataFrame(rows).set_index("model") if rows
                             else pd.DataFrame(index=pd.Index([], name="model")))      # e.g. no friendly drones in the test runs
    return pd.DataFrame(tier_rows), as_table(role_rows), as_table(eo_rows)


CLASS_COLS = ["p_Friend", "p_Neutral", "p_Hostile"]
CONTEXT = ["allegiance", "tier", "role", "cls", "eoir_top_class"] + TRACK_QUALITY


def load_neural(preds_dir, name, frame, prefix="pred", log=print, what="test"):
    """{TAG: (p, rows)} from <prefix>_<tag>_<name>.csv, aligned to the rows of `frame`."""
    out = {}
    for f in sorted(glob.glob(str(Path(preds_dir) / f"{prefix}_*_{name}.csv"))):
        tag = Path(f).stem[len(prefix) + 1:].split("_")[0].upper()
        pr = pd.read_csv(f)
        m = frame[["run", "eid", "t"]].merge(pr, on=["run", "eid", "t"], how="left")
        ok = m.p_Friend.notna().values
        if ok.mean() < 0.99:
            log(f"warning: {tag} predictions cover {ok.mean():.1%} of {name} {what} rows; scoring covered rows")
        out[tag] = (m.loc[ok, CLASS_COLS].values, frame[ok].reset_index(drop=True))
    return out


def range_flags(tr, frames, q=0.005):
    """Out-of-training-range flags. oor_tq: a track-quality input (position / velocity uncertainty, track
    age) above the training runs' 99.5th percentile (the S9 drone failure mode). oor_count: number of
    numeric inputs outside the training [0.5%, 99.5%] range."""
    cols = [c for c in TREE_OWN + TRACK_QUALITY if c in tr]
    X = tr[cols].apply(pd.to_numeric, errors="coerce")
    lo, hi = X.quantile(q), X.quantile(1 - q)
    ranges = pd.DataFrame(dict(lo=lo, hi=hi))
    out = []
    for f in frames:
        Z = f[cols].apply(pd.to_numeric, errors="coerce")
        outside = (Z.lt(lo) | Z.gt(hi)) & Z.notna()
        tq = [c for c in TRACK_QUALITY if c in Z]
        out.append(pd.DataFrame(dict(oor_tq=(Z[tq].gt(hi[tq]) & Z[tq].notna()).any(axis=1).values,
                                     oor_count=outside.sum(axis=1).values)))
    return ranges, out


def save_all_preds(name, fr, val_res, res, out_dir, log=print):
    """One table of validation and test rows, labels and context, and every deployable model's
    probabilities (columns '<model>|Friend', ...). Neural models are read by ensemble.py from their own
    pred_ / valpred_ files."""
    ranges, (fv, ft) = range_flags(fr["train"], [fr["val"], fr["test"]])
    parts = []
    for split, frame, flags, rr in (("val", fr["val"], fv, val_res), ("test", fr["test"], ft, res)):
        d = frame[["run", "eid", "t"] + [c for c in CONTEXT if c in frame]].copy().reset_index(drop=True)
        d.insert(0, "split", split)
        d["obs_tier"] = observable_tier(frame)          # tier from speed, altitude and RCS: usable for gating
        d = pd.concat([d, flags], axis=1)
        for m, p in rr.items():
            for k, c in enumerate(("Friend", "Neutral", "Hostile")):
                d[f"{m}|{c}"] = np.asarray(p)[:, k].astype(np.float32)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    path = Path(out_dir) / f"allpred_{name}.parquet"
    try:
        d.to_parquet(path, index=False)
    except Exception:
        path = path.with_suffix(".csv"); d.to_csv(path, index=False)
    ranges.to_csv(Path(out_dir) / f"ranges_{name}.csv")
    log(f"\nsaved {path.name}: {int((d.split == 'val').sum()):,} validation and {int((d.split == 'test').sum()):,} "
        f"test track-snapshots, models {', '.join(val_res)}; out of range (track quality): "
        f"val {d.oor_tq[d.split == 'val'].mean():.1%}, test {d.oor_tq[d.split == 'test'].mean():.1%}")
    return path


def fit_baselines(fr, log=print, want_val=False):
    """BN and tree baselines fitted on fr['train']. Returns (test predictions, validation predictions of the
    deployable models or {})."""
    tr, te, va = fr["train"], fr["test"], fr["val"]
    want_val = want_val and va is not None and len(va)
    allr = pd.concat([x for x in (tr, va, te) if x is not None], ignore_index=True)
    res, val = {}, {}
    for nm, ctx in (("BN-learned", False), ("BN-context", True)):
        bn = TieredBN(context=ctx).fit(tr)
        res[nm] = bn.predict_proba(te)
        if want_val:
            val[nm] = bn.predict_proba(va)
        if nm == "BN-learned":
            res["BN-oracle"] = TieredBN().fit(allr).predict_proba(te)
            res["BN-SME"] = TieredBN().fit(allr).sme(seed=1).predict_proba(te)
    for nb, tq in ((False, False), (True, False), (False, True)):
        out = fit_trees(tr, te, nb, with_tq=tq, extra=(va,) if want_val else ())
        tname, p = out[0], out[1]
        key = f"{tname}-own{'+nb' if nb else ''}{'+tq' if tq else ''}"
        res[key] = p
        if want_val:
            val[key] = out[2][0]
    order = ["BN-learned", "BN-oracle", "BN-SME", "BN-context"]
    res = {k: res[k] for k in order + [k for k in res if k not in order]}
    return res, val


def score_split(name, split, preds_dir, out_dir, log=print, B=2000, save_preds=False):
    """Fit the BN and tree baselines on split['train'], load GNN/GRU predictions, score split['test']."""
    fr = {k: (pd.concat([eval_frame(r) for r in v], ignore_index=True) if v else None) for k, v in split.items()}
    te = fr["test"]       # same training runs as the GNN (it uses val only to stop and calibrate)
    base, val = fit_baselines(fr, log, want_val=save_preds)
    res = {k: (p, te) for k, p in base.items()}
    res.update(load_neural(preds_dir, name, te, log=log))
    if save_preds:
        if val:
            save_all_preds(name, fr, val, {k: base[k] for k in val}, out_dir, log)
        else:
            log("--save-preds: no validation runs in this split; nothing saved")
    t = metrics_table(res)
    ci, diff = bootstrap(res, B=B, refs=("BN-learned", "BN-oracle", "GRU"))
    log(f"\n=== {name} (test runs: {', '.join(Path(r).name for r in split['test'])}) ===")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        log(t[["n", "bal_acc", "log_loss", "brier", "ece", "recall_friend", "recall_neutral", "recall_hostile",
               "airliner_called_hostile", "friend_called_hostile"]].to_string())
        show = pd.DataFrame({m: [f"{ci.loc[k, m]:.3f} [{ci.loc[k, m + '_lo']:.3f}, {ci.loc[k, m + '_hi']:.3f}]"
                                 for k in ci.index] for m in ("bal_acc", "log_loss", "friend_called_hostile")},
                            index=ci.index)
        show.insert(0, "aircraft", ci.n_entities)
        log("\n95% intervals, resampling whole aircraft:")
        log(show.to_string())
        key = diff[~diff.model.isin(["BN-learned", "BN-SME", "XGB-own", "HGB-own", "XGB-own+tq", "HGB-own+tq"]) & (diff.vs == "BN-oracle")]
        if len(key):
            log("\nDifference vs BN-oracle (negative log loss / friend_called_hostile = better):")
            log(key.assign(diff=key["diff"].round(3), lo=key.lo.round(3), hi=key.hi.round(3),
                           p_better=key.p_better.round(2)).to_string(index=False))
        gg = diff[(diff.vs == "GRU") & ~diff.model.str.startswith(("BN-", "XGB-", "HGB-"))]
        if len(gg):
            log("\nNeural variants vs GRU (GNN vs GRU = the value of the graph layers) (negative log loss / friend_called_hostile = better):")
            log(gg.assign(diff=gg["diff"].round(3), lo=gg.lo.round(3), hi=gg.hi.round(3),
                          p_better=gg.p_better.round(2)).to_string(index=False))
        bt, br, be = breakdowns(res)
        log("\nBy tier (tier 1 = drones):")
        log(bt.pivot(index="model", columns="tier", values=["bal_acc", "friend_called_hostile", "missed_hostile"])
            .round(3).to_string())
        log("\nFriendly track-snapshots called Hostile, by friendly role:")
        log(br.round(3).to_string())
        if len(be):
            log("\nOwn drones called Hostile, by whether the EO/IR camera had reported on them:")
            log(be.round(3).to_string())
    bt.round(4).to_csv(Path(out_dir) / f"compare_{name}_by_tier.csv", index=False)
    br.round(4).to_csv(Path(out_dir) / f"compare_{name}_by_role.csv")
    if len(be):
        be.round(4).to_csv(Path(out_dir) / f"compare_{name}_own_drones_eoir.csv")
    t.to_csv(Path(out_dir) / f"compare_{name}.csv")
    ci.round(4).to_csv(Path(out_dir) / f"compare_{name}_ci.csv")
    diff.round(4).to_csv(Path(out_dir) / f"compare_{name}_diffs.csv", index=False)
    return t, ci, diff


def compare(runs, preds_dir, out_dir, log=print, test_runs=None, B=2000, save_preds=False):
    """Per scenario (own test seeds), or train-on-A / test-on-B when test_runs is given."""
    if test_runs:
        tr_scn = sorted({read_meta(r)["scenario"] for r in runs})
        te_scn = sorted({read_meta(r)["scenario"] for r in test_runs})
        split = split_runs(runs, test_frac=0.0)
        split["test"] = list(test_runs)
        name = f"{'+'.join(tr_scn)}__on__{'+'.join(te_scn)}"
        return {name: score_split(name, split, preds_dir, out_dir, log, B, save_preds)}
    by = {}
    for r in runs:
        by.setdefault(read_meta(r)["scenario"], []).append(r)
    return {scn: score_split(scn, split_runs(rs), preds_dir, out_dir, log, B, save_preds) for scn, rs in sorted(by.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="runs (per-scenario mode) or training runs (shift mode)")
    ap.add_argument("--test-runs", nargs="+", default=None, help="shift mode: score models trained on --runs here")
    ap.add_argument("--preds", default="results")
    ap.add_argument("--out", default="results")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--save-preds", action="store_true",
                    help="also save validation and test predictions of the deployable BN and tree models "
                         "(allpred_<name>) for models/ensemble.py")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    compare(expand(a.runs), a.preds, a.out, test_runs=expand(a.test_runs) if a.test_runs else None, B=a.bootstrap,
            save_preds=a.save_preds)


if __name__ == "__main__":
    main()
