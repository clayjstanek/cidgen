"""Shared evaluation layer: one evaluation table, one run-level split and one metric set, so that
every model family (BN variants, XGBoost, GNN) is scored on exactly the same rows.

Evaluation rows = objects of interest (birds excluded) at snapshot times (default every 10 s),
because that is where the graph edges exist.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..io import read_meta, read_table

CLASSES = ["Friend", "Neutral", "Hostile"]
NB_COLS = ["speed", "alt", "climb", "rad_rcs_med30", "iff_ever_valid", "iff_mode5_valid", "asset_closure_mps",
           "descending_closing", "esm_conf", "airway_xtrack_m"]


def eval_frame(run_dir) -> pd.DataFrame:
    """Per-track features + labels + mean-neighbour aggregates at snapshot times, objects of interest only."""
    f = read_table(run_dir, "features_track"); lab = read_table(run_dir, "labels")
    tt = read_table(run_dir, "edges_track_track")
    snap = np.sort(tt.t.unique())
    f = f[f.t.isin(snap)]
    lab = lab[lab.t.isin(snap)]
    df = f.merge(lab, on=["eid", "t"])
    nb = tt.merge(f[["eid", "t"] + NB_COLS], left_on=["dst_eid", "t"], right_on=["eid", "t"]).drop(columns="eid")
    agg = nb.groupby(["src_eid", "t"]).agg(**{f"nb_{c}": (c, "mean") for c in NB_COLS},
                                           nb_count=("dst_eid", "size"), nb_comove=("comove_corr", "mean"),
                                           nb_dist=("dist_m", "mean"), nb_speed_ratio=("speed_ratio", "mean"),
                                           nb_heading_diff=("heading_diff_deg", "mean"),
                                           nb_any_valid_iff=("iff_ever_valid", "max")).reset_index()
    df = df.merge(agg, left_on=["eid", "t"], right_on=["src_eid", "t"], how="left").drop(columns="src_eid")
    df["nb_count"] = df.nb_count.fillna(0)
    meta = read_meta(run_dir)
    df = df.copy().assign(run=Path(run_dir).name, scenario=meta["scenario"], seed=meta["seed"],
                   threat_axis_deg=meta.get("threat_axis_deg", 0.0)).copy()
    df = df[df.is_object_of_interest.astype(bool)].reset_index(drop=True)
    df["allegiance"] = df.allegiance.astype(int)
    return df


def expand(patterns):
    """Run folders from paths or glob patterns (quote patterns on the command line)."""
    import glob
    runs = sorted({p for pat in patterns for p in glob.glob(str(pat)) if (Path(p) / "scenario_meta.json").exists()})
    if not runs:
        raise SystemExit(f"no runs match {patterns}")
    return runs


def split_runs(run_dirs, val_frac=0.15, test_frac=0.15):
    """Run-level split, deterministic by seed order within each scenario."""
    by = {}
    for r in run_dirs:
        m = read_meta(r)
        by.setdefault(m["scenario"], []).append((m["seed"], str(r)))
    split = {"train": [], "val": [], "test": []}
    for scn, rs in by.items():
        rs = [r for _, r in sorted(rs)]
        n = len(rs)
        n_te = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
        n_va = max(1, int(round(n * val_frac))) if n >= 3 else 0
        split["test"] += rs[n - n_te:]
        split["val"] += rs[n - n_te - n_va:n - n_te]
        split["train"] += rs[:n - n_te - n_va]
    return split


def ece(prob, y, n_bins=10):
    conf = prob.max(1); pred = prob.argmax(1)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(prob: np.ndarray, df: pd.DataFrame) -> dict:
    """Headline metrics plus the two guardrails (spec S3 and S8)."""
    y = df.allegiance.values
    prob = np.clip(prob, 1e-7, 1.0); prob = prob / prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    rec = [float((pred[y == k] == k).mean()) if (y == k).any() else np.nan for k in range(3)]
    out = dict(n=len(y), bal_acc=float(np.nanmean(rec)), acc=float((pred == y).mean()),
               log_loss=float(-np.log(prob[np.arange(len(y)), y]).mean()),
               brier=float(((prob - np.eye(3)[y]) ** 2).sum(1).mean()), ece=ece(prob, y),
               recall_friend=rec[0], recall_neutral=rec[1], recall_hostile=rec[2])
    air = (df.cls.values == "E") & (y == 1)
    out["airliner_called_hostile"] = float((pred[air] == 2).mean()) if air.any() else np.nan     # S3 guardrail
    fr = y == 0
    out["friend_called_hostile"] = float((pred[fr] == 2).mean()) if fr.any() else np.nan       # S8 guardrail
    return out


def metrics_table(results: dict) -> pd.DataFrame:
    """results: {model_name: (prob, df)} -> one row per model."""
    rows = [dict(model=k, **metrics(p, d)) for k, (p, d) in results.items()]
    t = pd.DataFrame(rows).set_index("model")
    return t.round(3)


# ----------------------------------------------------------------------------- bootstrap CIs
CI_METRICS = ["bal_acc", "log_loss", "recall_friend", "recall_hostile", "friend_called_hostile",
              "airliner_called_hostile"]


def _entity_sums(prob, df, ent_index):
    """Per-entity sufficient statistics for the CI metrics: (n_entities, 12)."""
    y = df.allegiance.values
    prob = np.clip(prob, 1e-7, 1.0); prob = prob / prob.sum(1, keepdims=True)
    pred = prob.argmax(1)
    air = (df.cls.values == "E") & (y == 1)
    cols = [(y == k).astype(float) for k in range(3)] + [((y == k) & (pred == k)).astype(float) for k in range(3)]
    cols += [-np.log(prob[np.arange(len(y)), y]), np.ones(len(y)),
             (y == 0).astype(float), ((y == 0) & (pred == 2)).astype(float),
             air.astype(float), (air & (pred == 2)).astype(float)]
    X = np.stack(cols, 1)
    e = ent_index.loc[list(zip(df.run.values, df.eid.values))].values
    S = np.zeros((len(ent_index), X.shape[1]))
    np.add.at(S, e, X)
    return S


def _from_sums(S):
    """Metrics from (possibly weighted) sums; S is (..., 12)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = S[..., 3:6] / S[..., 0:3]
        return dict(bal_acc=np.nanmean(rec, -1), log_loss=S[..., 6] / S[..., 7],
                    recall_friend=rec[..., 0], recall_hostile=rec[..., 2],
                    friend_called_hostile=S[..., 9] / S[..., 8], airliner_called_hostile=S[..., 11] / S[..., 10])


def bootstrap(results: dict, B=2000, seed=0, refs=("BN-learned", "BN-oracle")):
    """Entity-cluster Poisson bootstrap. Snapshots of one aircraft are correlated, so whole aircraft
    are resampled. The same weights are used for every model, so differences are paired.

    Returns (ci_table, diff_table)."""
    keys = pd.concat([d[["run", "eid"]] for _, d in results.values()]).drop_duplicates()
    ent_index = pd.Series(np.arange(len(keys)), index=pd.MultiIndex.from_frame(keys))
    W = np.random.default_rng(seed).poisson(1.0, size=(B, len(keys))).astype(float)
    point, boot = {}, {}
    for name, (p, d) in results.items():
        S = _entity_sums(p, d, ent_index)
        point[name] = _from_sums(S.sum(0))
        boot[name] = _from_sums(W @ S)
    rows = []
    for name in results:
        r = {"model": name, "n_entities": int((_entity_sums(*results[name], ent_index)[:, 7] > 0).sum())}
        for m in CI_METRICS:
            lo, hi = np.nanpercentile(boot[name][m], [2.5, 97.5])
            r[m] = point[name][m]; r[f"{m}_lo"] = lo; r[f"{m}_hi"] = hi
        rows.append(r)
    ci = pd.DataFrame(rows).set_index("model")
    drows = []
    for ref in refs:
        if ref not in results:
            continue
        for name in results:
            if name == ref:
                continue
            for m in ("bal_acc", "log_loss", "friend_called_hostile"):
                d = boot[name][m] - boot[ref][m]
                d = d[np.isfinite(d)]
                if not len(d):
                    continue
                lo, hi = np.percentile(d, [2.5, 97.5])
                better = (d < 0).mean() if m != "bal_acc" else (d > 0).mean()
                drows.append(dict(model=name, vs=ref, metric=m, diff=point[name][m] - point[ref][m],
                                  lo=lo, hi=hi, p_better=better,
                                  verdict=("better" if (hi < 0 if m != "bal_acc" else lo > 0) else
                                           "worse" if (lo > 0 if m != "bal_acc" else hi < 0) else "no clear difference")))
    return ci, pd.DataFrame(drows)
