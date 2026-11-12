"""Replay runs through Combat ID identities and threat prioritisation, and score the ranked lists.

    python -m cidgen.threat.run_tp --runs "runs/S9_sensor_shift_*" --preds results_transfer \
        --name S0_baseline__on__S9_sensor_shift \
        --identity truth BN-learned XGB-own GNN-SD30-NOTQ STACK:BN-learned+XGB-own+GNN-SD30-NOTQ \
        --doctrine configs/threat/doctrine_default.yaml --out results_tp

For each identity source, three rankers are run on every snapshot (10 s apart in the stored runs):
  ph        P(Hostile), ties broken by P(Friend): the starting point
  threat    expected threat (identity x lethality x urgency), no doctrine rules
  doctrine  expected threat plus the doctrine's rules and engagement tiers

Outputs in --out
  tp_<name>__<doctrine>.csv                 one row per identity x ranker, metrics in models/threat/evaluate.py
  tp_<name>__<doctrine>__lists_<run>.csv    full ranked lists for --dump-run (every snapshot, with reasons)

Several --doctrine files can be given; each is scored separately so YAML changes can be compared.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from ..models.common import expand
from .evaluate import score_lists
from .truth_threat import truth_threats
from .picture import build_frames, identity, pictures
from .prioritize import PHRanker, ThreatPrioritizer, no_rules

KEEP = ["run", "t", "eid", "rank", "tier", "score_smoothed", "p_friend", "p_neutral", "p_hostile", "ttg_s",
        "threatened_asset", "lethality_class", "flags", "reasons", "allegiance", "role", "cls", "threat_priority", "truth_eid"]


def replay(frame, P, ranker):
    """Run one ranker over every snapshot in time order; returns the concatenated ranked lists."""
    out, cur_run = [], None
    for run, t, pic in pictures(frame, P):
        if run != cur_run:
            ranker.reset(); cur_run = run
        ranked = ranker.prioritize(pic, t)
        truth = pic[["eid", "allegiance", "role", "cls", "threat_priority", "ttg_s", "threatened_asset", "truth_eid"]]
        ranked = ranked.drop(columns=[c for c in ("ttg_s", "threatened_asset") if c in ranked]).merge(truth, on="eid")
        ranked["run"] = run
        out.append(ranked[[c for c in KEEP if c in ranked]])
    return pd.concat(out, ignore_index=True)


def _group_counts(frame, runs, mode, calib):
    """group_n per (run, eid, t): 1 (none), the RCS/echo estimate (est), or the true group size (true)."""
    if mode == "none":
        return frame
    import json
    from .group_count import estimate, true_group_size
    cal = json.load(open(calib)) if mode == "est" else None
    parts = []
    for r in runs:
        rn = Path(r).name; times = frame.t[frame.run == rn].unique()
        g = estimate(r, times, cal)[["eid", "t", "n_hat"]].rename(columns={"n_hat": "group_n"}) if mode == "est" \
            else true_group_size(r, times).rename(columns={"n_true": "group_n"})
        parts.append(g.assign(run=rn))
    g = pd.concat(parts, ignore_index=True)
    f = frame.drop(columns=[c for c in ("group_n",) if c in frame]).merge(g, on=["run", "eid", "t"], how="left")
    f["group_n"] = f.group_n.fillna(1.0).clip(lower=1.0)
    return f


def run(runs, pred_dir, name, sources, doctrine_paths, out_dir, dump_run=None, threat_min=0.05, log=print,
        group_count="none", group_calib=None, tag=None):
    sfx = f"__{tag}" if tag else ""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for dp in doctrine_paths:
        doc = yaml.safe_load(open(dp)); dname = Path(dp).stem
        log(f"\n=== doctrine {dname} (weapons control: {doc.get('weapons_control')}) ===")
        frame = build_frames(runs, doc)
        log(f"{len(frame):,} track-snapshots in {frame.run.nunique()} runs")
        frame = _group_counts(frame, runs, group_count, group_calib)
        if group_count != "none":
            log(f"group tracks ({group_count}): mean members {frame.group_n.mean():.2f}, "
                f"share >= 1.5: {(frame.group_n >= 1.5).mean():.3f}")
        # every real threat at every snapshot time, tracked or not (for the *_all metrics)
        truth = pd.concat([truth_threats(r, frame.t[frame.run == Path(r).name].unique()).assign(run=Path(r).name)
                           for r in runs], ignore_index=True)
        rows, dumps = [], []
        for src in sources:
            P = identity(frame, src, pred_dir, name)
            cov = float(pd.DataFrame(P).notna().all(axis=1).mean())
            for rname, ranker in (("ph", PHRanker()), ("threat", ThreatPrioritizer(no_rules(doc))),
                                  ("doctrine", ThreatPrioritizer(doc))):
                lists = replay(frame, P, ranker)
                m = score_lists(lists, threat_min=threat_min, truth=truth)
                rows.append(dict(identity=src, ranker=rname, coverage=round(cov, 4), **m))
                log(f"  {src:40s} {rname:9s} captured3 {m['threat_captured3']:.3f}  top-threat rank {m['top_threat_rank']:.1f}  "
                    f"tau {m['order_tau']:.2f}  ndcg10 {m['ndcg10']:.3f}  "
                    f"delay {m['first_top3_delay_s']:.0f}s  friend-above-threat {m['friend_above_threat']:.3f}  "
                    f"friend-engage {m['friend_engage']:.4f}  |  all threats: tracked {m['threat_tracked']:.3f}  "
                    f"covered {m['threat_covered']:.3f}  captured3_cov {m['threat_captured3_cov']:.3f}  "
                    f"captured3 {m['threat_captured3_all']:.3f}  delay {m['first_top3_delay_all_s']:.0f}s")
                if dump_run is not None:
                    dumps.append(lists[lists.run.str.endswith(str(dump_run))].assign(identity=src, ranker=rname))
        tab = pd.DataFrame(rows)
        tab.round(4).to_csv(out_dir / f"tp_{name}__{dname}{sfx}.csv", index=False)
        if dumps:
            d = pd.concat(dumps, ignore_index=True)
            d.to_csv(out_dir / f"tp_{name}__{dname}{sfx}__lists_{dump_run}.csv", index=False)
        results[dname] = tab
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="test run folders or patterns (quote them)")
    ap.add_argument("--preds", default=None, help="folder with allpred_/pred_/valpred_ files")
    ap.add_argument("--name", default=None, help="prediction name, e.g. S0_baseline__on__S9_sensor_shift")
    ap.add_argument("--identity", nargs="+", default=["truth"], help="identity sources (see picture.py)")
    ap.add_argument("--doctrine", nargs="+", default=["configs/threat/doctrine_default.yaml"])
    ap.add_argument("--out", default="results_tp")
    ap.add_argument("--dump-run", default=None, help="write full ranked lists for the run ending with this, e.g. seed3")
    ap.add_argument("--threat-min", type=float, default=0.05, help="truth threat priority that counts as a real threat")
    ap.add_argument("--group-count", choices=["none", "est", "true"], default="none",
                    help="scale each track's threat by its members: none, est (RCS/echo estimate, needs "
                         "--group-calib from cidgen.threat.group_count), true (true group size: upper bound)")
    ap.add_argument("--group-calib", default=None, help="calibration json from cidgen.threat.group_count")
    ap.add_argument("--tag", default=None, help="suffix for the output file names, e.g. grp_est")
    a = ap.parse_args()
    if a.group_count == "est" and not a.group_calib:
        ap.error("--group-count est needs --group-calib")
    run(expand(a.runs), a.preds, a.name, a.identity, a.doctrine, a.out, a.dump_run, a.threat_min,
        group_count=a.group_count, group_calib=a.group_calib, tag=a.tag)


if __name__ == "__main__":
    main()
