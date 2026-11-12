"""Drill into model prediction files (pred_<model>_<scenario>.csv) by joining them to the run truth.

    python analyze_preds.py --preds results/pred_gru-sd30-notq_S12_hypersonic.csv --runs runs --out analysis
    python analyze_preds.py --preds "results/pred_*_S12_hypersonic.csv" --runs runs --out analysis   # several models
  python analyze_preds.py --preds results/valpred_gru-sd30-notq_S12_hypersonic.csv ...   # validation split

valpred_ files cover the validation runs (used for early stopping and calibration, so slightly
optimistic); they are useful mainly to double the number of aircraft when the test split is small.

A prediction file holds one row per (run, eid, t) track snapshot with the model's calibrated
probabilities p_Friend, p_Neutral, p_Hostile. It has no truth, so this script joins each row to the
run's labels and entities tables. The decision is the most likely class, and balanced accuracy,
friendly-called-hostile and missed-hostile are defined exactly as in models/compare.py.

Writes to --out (and prints a short report):
  by_tier.csv              metrics per tier, with 95% intervals from resampling whole aircraft
  by_class_role.csv        error rates per true allegiance, class and role
  worst_entities.csv       the aircraft with the most wrong snapshots
  by_track_age.csv         error rate vs time since the track's first prediction
  confusion.csv            true x called counts per tier
  by_neighbours.csv        error rate vs the true allegiance of the track's graph neighbours
                           (edges_track_track: the tracks the GNN passes messages between)
  report.txt               the printed report
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cidgen.io import read_table

ALG = {0: "Friend", 1: "Neutral", 2: "Hostile"}
P_COLS = ["p_Friend", "p_Neutral", "p_Hostile"]
AGE_BINS = [0, 10, 30, 60, 120, 300, np.inf]
AGE_LABELS = ["0-10 s", "10-30 s", "30-60 s", "60-120 s", "120-300 s", "300+ s"]


def model_name(path):
    """pred_<model>_<scenario>.csv (test split) or valpred_<model>_<scenario>.csv (validation split)."""
    m = re.match(r"(val)?pred_(.+?)_(S\d+.*)\.csv$", Path(path).name)
    if not m:
        return Path(path).stem, "?"
    return m.group(2).upper() + (" (validation)" if m.group(1) else ""), m.group(3)


def load(pred_path, runs_root):
    p = pd.read_csv(pred_path)
    lab, ent, nbr = [], [], []
    for r in p.run.unique():
        d = Path(runs_root) / r
        if not d.exists():
            raise SystemExit(f"run folder not found: {d}  (pass the folder that holds it with --runs)")
        lab.append(read_table(d, "labels").assign(run=r))
        e = read_table(d, "entities").assign(run=r)
        if "flag_profile" not in e:
            e["flag_profile"] = np.nan
        ent.append(e[["run", "eid", "flag_profile"]])
        # graph neighbours at each snapshot, by their TRUE allegiance (diagnostic only)
        ed = read_table(d, "edges_track_track")[["t", "src_eid", "dst_eid"]]
        # neighbour's true allegiance from the labels at that time (tracker runs: track -> matched
        # object; false tracks are -1 and count as none of the three)
        la = lab[-1][["eid", "t", "allegiance"]].rename(columns={"eid": "dst_eid", "allegiance": "nb_alg"})
        ed = ed.merge(la, on=["dst_eid", "t"], how="left")
        nb = ed.groupby(["t", "src_eid"]).nb_alg.agg(nb_n="size", nb_friend=lambda s: int((s == 0).sum()),
                                                      nb_neutral=lambda s: int((s == 1).sum()),
                                                      nb_hostile=lambda s: int((s == 2).sum())).reset_index()
        nbr.append(nb.rename(columns={"src_eid": "eid"}).assign(run=r))
    lab = pd.concat(lab, ignore_index=True); ent = pd.concat(ent, ignore_index=True)
    nbr = pd.concat(nbr, ignore_index=True)
    m = p.merge(lab, on=["run", "eid", "t"], how="left", indicator=True)
    missing = int((m._merge != "both").sum())
    if missing:
        print(f"WARNING: {missing} of {len(p)} predictions had no truth row and were dropped")
    m = m[m._merge == "both"].drop(columns="_merge")
    if "truth_eid" in m:                     # tracker runs: entity columns via the matched true object
        m = m[m.is_object_of_interest.astype(bool)].merge(ent.rename(columns={"eid": "truth_eid"}),
                                                          on=["run", "truth_eid"], how="left")
    else:
        m = m[m.is_object_of_interest.astype(bool)].merge(ent, on=["run", "eid"], how="left")
    m = m.merge(nbr, on=["run", "eid", "t"], how="left")
    for c in ("nb_n", "nb_friend", "nb_neutral", "nb_hostile"):
        m[c] = m[c].fillna(0).astype(int)
    m["y"] = m.allegiance.astype(int)
    m["pred"] = m[P_COLS].values.argmax(1)
    m["conf"] = m[P_COLS].values.max(1)
    m["wrong"] = m.pred != m.y
    m["track_age_s"] = m.t - m.groupby(["run", "eid"]).t.transform("min")
    # one key per aircraft (tracker runs: the matched true object, so a broken track is not counted twice)
    m["key"] = m.run + ":" + (m.truth_eid if "truth_eid" in m else m.eid).astype(str)
    return m


def tier_metrics(d):
    """compare.py definitions: mean per-class recall over classes present; friend->hostile; 1 - hostile recall."""
    y, pr = d.y.values, d.pred.values
    rec = [((pr == k) & (y == k)).sum() / (y == k).sum() if (y == k).any() else np.nan for k in range(3)]
    fr = y == 0
    return dict(bal_acc=float(np.nanmean(rec)),
                friend_called_hostile=float((pr[fr] == 2).mean()) if fr.any() else np.nan,
                missed_hostile=float(1 - rec[2]) if not np.isnan(rec[2]) else np.nan)


def entity_bootstrap(d, B, rng):
    """95% intervals by resampling whole aircraft (an aircraft's snapshots are not independent)."""
    groups = [g for _, g in d.groupby("key")]
    stats = []
    for _ in range(B):
        pick = rng.integers(len(groups), size=len(groups))
        stats.append(tier_metrics(pd.concat([groups[i] for i in pick], ignore_index=True)))
    s = pd.DataFrame(stats)
    q = lambda c, x: float(np.nanpercentile(s[c], x)) if s[c].notna().any() else np.nan   # e.g. no friends in tier 3
    return {f"{c}_lo": q(c, 2.5) for c in s} | {f"{c}_hi": q(c, 97.5) for c in s}


def analyze(m, model, B, rng, top):
    rows_t, rows_c, rows_age, rows_conf, rows_nb = [], [], [], [], []
    for t, d in m.groupby("tier"):
        met = tier_metrics(d)
        rows_t.append(dict(model=model, tier=int(t), aircraft=d.key.nunique(), snapshots=len(d),
                           friends=d[d.y == 0].key.nunique(), neutrals=d[d.y == 1].key.nunique(),
                           hostiles=d[d.y == 2].key.nunique(), **met, **entity_bootstrap(d, B, rng)))
        for (a, c, role), g in d.groupby(["y", "cls", "role"]):
            per = g.groupby("key").wrong.mean()
            rows_c.append(dict(model=model, tier=int(t), true=ALG[a], cls=c, role=role, aircraft=len(per),
                               snapshots=len(g), error_rate=float(g.wrong.mean()),
                               aircraft_with_any_error=int((per > 0).sum()),
                               aircraft_mostly_wrong=int((per > 0.5).sum()),
                               top_wrong_call=ALG[int(g[g.wrong].pred.mode()[0])] if g.wrong.any() else ""))
        d = d.assign(age=pd.cut(d.track_age_s, AGE_BINS, labels=AGE_LABELS, right=False))
        for age, g in d.groupby("age", observed=True):
            rows_age.append(dict(model=model, tier=int(t), track_age=str(age), snapshots=len(g),
                                 error_rate=float(g.wrong.mean())))
        for a, other, col in ((2, 0, "nb_friend"), (0, 2, "nb_hostile"), (1, 0, "nb_friend")):
            g = d[d.y == a]
            if not len(g):
                continue
            b = pd.cut(g[col], [-1, 0, 2, np.inf], labels=["none", "1-2", "3+"])
            for lvl, gg in g.groupby(b, observed=True):
                rows_nb.append(dict(model=model, tier=int(t), true=ALG[a], neighbours=f"{ALG[other].lower()} neighbours: {lvl}",
                                    aircraft=gg.key.nunique(), snapshots=len(gg), error_rate=float(gg.wrong.mean()),
                                    called_as_that_side=float((gg.pred == other).mean())))
        cm = pd.crosstab(d.y.map(ALG), d.pred.map(ALG)).reindex(index=list(ALG.values()), columns=list(ALG.values()),
                                                                 fill_value=0)
        rows_conf.append(cm.assign(model=model, tier=int(t)).reset_index().rename(columns={"y": "true"}))
    per = m.groupby("key").agg(run=("run", "first"), eid=("eid", "first"), tier=("tier", "first"), cls=("cls", "first"),
                               role=("role", "first"), true=("y", lambda s: ALG[int(s.iloc[0])]),
                               profile=("flag_profile", "first"), snapshots=("wrong", "size"),
                               wrong=("wrong", "sum"), first_t=("t", "min"), last_t=("t", "max"))
    per["frac_wrong"] = per.wrong / per.snapshots
    w = m[m.wrong]
    per["called_as"] = w.groupby("key").pred.agg(lambda s: ALG[int(s.mode()[0])]).reindex(per.index).fillna("")
    per["mean_conf_when_wrong"] = w.groupby("key").conf.mean().reindex(per.index)
    worst = per[per.wrong > 0].sort_values(["wrong", "frac_wrong"], ascending=False).head(top)
    return (pd.DataFrame(rows_t), pd.DataFrame(rows_c), worst.assign(model=model).reset_index(drop=True),
            pd.DataFrame(rows_age), pd.concat(rows_conf, ignore_index=True), pd.DataFrame(rows_nb), per)


def report(model, scen, m, bt, bc, per, age, lines):
    say = lines.append
    say(f"\n=== {model} on {scen}: {m.run.nunique()} runs, {m.key.nunique()} aircraft, {len(m):,} snapshots ===")
    for r in bt.itertuples():
        fch = "" if np.isnan(r.friend_called_hostile) else \
            f"friend->hostile {100*r.friend_called_hostile:.1f}% [{100*r.friend_called_hostile_lo:.1f}-{100*r.friend_called_hostile_hi:.1f}]  "
        mh = "" if np.isnan(r.missed_hostile) else \
            f"missed hostile {100*r.missed_hostile:.1f}% [{100*r.missed_hostile_lo:.1f}-{100*r.missed_hostile_hi:.1f}]"
        say(f"Tier {r.tier}: {r.aircraft} aircraft ({r.friends} friend, {r.neutrals} neutral, {r.hostiles} hostile), "
            f"bal acc {r.bal_acc:.3f}  {fch}{mh}")
    say("  [brackets: 95% interval from resampling whole aircraft; wide = too few aircraft to be sure]")
    for t, d in m.groupby("tier"):
        p = per[per.tier == t]
        nw = int(p.wrong.sum())
        if not nw:
            say(f"Tier {t}: no wrong snapshots"); continue
        top5 = p.sort_values("wrong", ascending=False).head(5)
        say(f"Tier {t}: {nw} wrong snapshots from {int((p.wrong > 0).sum())} of {len(p)} aircraft; the worst 5 aircraft "
            f"account for {100*top5.wrong.sum()/nw:.0f}% ({', '.join(f'{r.true} {r.cls}/{r.role} {r.wrong}' for r in top5.itertuples())})")
        early = d[d.track_age_s < 30]; late = d[d.track_age_s >= 30]
        for a, other, col in ((2, 0, "nb_friend"), (0, 2, "nb_hostile")):
            g = d[d.y == a]
            if len(g) and (g[col] > 0).any() and (g[col] == 0).any():
                w, wo = g[g[col] > 0], g[g[col] == 0]
                say(f"         {ALG[a]} snapshots called {ALG[other]}: {100*(w.pred == other).mean():.1f}% with a "
                    f"{ALG[other].lower()} neighbour ({len(w)} snapshots), {100*(wo.pred == other).mean():.1f}% without ({len(wo)})")
        say(f"         error rate in a track's first 30 s {100*early.wrong.mean():.1f}%, after 30 s {100*late.wrong.mean():.1f}%; "
            f"wrong calls made with >90% confidence: {100*(d[d.wrong].conf > 0.9).mean():.0f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="+", required=True, help="pred_*.csv files or quoted glob patterns")
    ap.add_argument("--runs", default="runs", help="folder holding the run folders named in the predictions")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--bootstrap", type=int, default=500, help="resamples for the intervals (0 to skip)")
    ap.add_argument("--top", type=int, default=25, help="rows in worst_entities.csv per model")
    a = ap.parse_args()
    files = sorted({f for p in a.preds for f in (glob.glob(p) or [p])})
    files = [f for f in files if re.match(r"(val)?pred_", Path(f).name)]
    if not files:
        raise SystemExit("no pred_*.csv or valpred_*.csv files matched --preds")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    acc = {k: [] for k in ("by_tier", "by_class_role", "worst_entities", "by_track_age", "confusion", "by_neighbours")}
    lines = []
    for f in files:
        model, scen = model_name(f)
        m = load(f, a.runs)
        bt, bc, worst, age, cm, nbt, per = analyze(m, model, max(a.bootstrap, 1), rng, a.top)
        for k, v in zip(acc, (bt, bc, worst, age, cm, nbt)):
            acc[k].append(v.assign(scenario=scen))
        report(model, scen, m, bt, bc, per, age, lines)
    for k, v in acc.items():
        pd.concat(v, ignore_index=True).round(4).to_csv(out / f"{k}.csv", index=False)
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)
    print(f"\nTables written to {out.resolve()}")


if __name__ == "__main__":
    main()
