"""Tracker quality across runs made with a real tracker (tracking.method: gnn).

    python track_report.py --runs "runs_gnn/S12_gnn_seed*"  [--out results_gnn]

Reads each run's tracker_summary and tracker_objects tables and prints
  * per run: objects detected, confirmed tracks, coverage, tracks per object, false-track rows,
    duplicate tracks dropped, track swaps, median confirmation delay, median position error;
  * per tier (objects of interest): coverage, tracks per object, confirmation delay, position error.
Definitions
  coverage            share of grid times, between an object's first and last radar detection, at which
                      some confirmed track is matched to it (label-mode runs score about 0.97-1.0 here)
  tracks per object   distinct tracks matched to the object over its life (1 = never broken)
  false-track rows    share of track rows matched to no object (clutter tracks, duplicates, coasting
                      tracks whose object has left the match gate); these rows are not objects of
                      interest, so CID metrics and the threat list skip them
  track swaps         times a track's matched object changes from one object to another
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cidgen.io import read_table
from cidgen.models.common import expand


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="folder for tracker_runs.csv and tracker_tiers.csv")
    a = ap.parse_args()
    summ, objs = [], []
    for d in expand(a.runs):
        try:
            s = read_table(d, "tracker_summary"); o = read_table(d, "tracker_objects")
        except Exception:
            print(f"skipping {d}: no tracker tables (label-mode run?)")
            continue
        summ.append(s.assign(run=Path(d).name)); objs.append(o.assign(run=Path(d).name))
    if not summ:
        raise SystemExit("no tracker runs found")
    s = pd.concat(summ, ignore_index=True); o = pd.concat(objs, ignore_index=True)
    o = o[o.is_ooi.astype(bool)]
    cols = ["run", "n_objects_detected", "n_tracks_confirmed", "coverage_mean", "tracks_per_object_mean",
            "false_track_row_frac", "duplicate_tracks_dropped", "track_swaps", "confirm_delay_median_s",
            "pos_err_median_m"]
    cols = [c for c in cols if c in s]
    pd.set_option("display.width", 200)
    print("\n=== per run ===")
    print(s[cols].round(3).to_string(index=False))
    print("\n=== mean over runs ===")
    print(s[cols[1:]].mean().round(3).to_string())
    t = o.groupby("tier").agg(objects=("truth_eid", "size"), coverage_mean=("coverage", "mean"),
                              coverage_median=("coverage", "median"),
                              tracks_per_object_median=("n_tracks", "median"),
                              tracks_per_object_p90=("n_tracks", lambda x: x.quantile(0.9)),
                              never_tracked=("first_track_s", lambda x: int(x.isna().sum())),
                              confirm_delay_median_s=("confirm_delay_s", "median"),
                              pos_err_median_m=("mean_err_m", "median")).round(3)
    print("\n=== per tier, objects of interest, all runs ===")
    print(t.to_string())
    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        s.to_csv(out / "tracker_runs.csv", index=False); t.to_csv(out / "tracker_tiers.csv")
        print(f"\nwrote {out / 'tracker_runs.csv'} and {out / 'tracker_tiers.csv'}")


if __name__ == "__main__":
    main()
