"""Tier-routed predictions: combine neural models by the OBSERVED tier of each track snapshot.

    python -m cidgen.models.route_preds --dir results --name S12_hypersonic --runs runs \\
        --route 1=gnn-sd10-notq-eg 2=gru-sd30-notq 3=gru-sd30-notq --tag route-eg-gru

Reads pred_<model>_<name>.csv (test rows) and valpred_<model>_<name>.csv (validation rows) for each
model named in --route, and writes pred_<tag>_<name>.csv and valpred_<tag>_<name>.csv in which every
row's probabilities come from the model assigned to that row's observed tier. compare.py, ensemble.py
and threat/run_tp.py then treat the routed model like any other (identity source ROUTE-EG-GRU, ...).

The tier is taken from observables only, with the same rule the tiered BN uses
(models.bn.observable_tier: speed >= 1000 m/s -> 3; speed < 60 m/s, altitude < 1500 m and RCS < -5 dBsm
-> 1; else 2), computed from each run's features_track at the snapshot time. No truth is used.
All routed models must have been trained on the same split, so their files hold the same rows.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from ..io import read_table
from .bn import observable_tier

P_COLS = ["p_Friend", "p_Neutral", "p_Hostile"]
KEY = ["run", "eid", "t"]


def parse_route(items):
    route = {}
    for it in items:
        m = re.fullmatch(r"([123])=([A-Za-z0-9-]+)", it.strip())
        if not m:
            raise SystemExit(f"--route {it!r}: use TIER=MODEL, e.g. 1=gnn-sd10-notq-eg (model as in pred_<model>_<name>.csv)")
        route[int(m.group(1))] = m.group(2).lower()
    missing = {1, 2, 3} - set(route)
    if missing:
        raise SystemExit(f"--route: no model for tier(s) {sorted(missing)}; give all three")
    return route


def observed_tiers(keys, runs_root):
    """Observed tier for each (run, eid, t) row."""
    out = []
    for run, g in keys.groupby("run", sort=False):
        d = Path(runs_root) / run
        if not d.exists():
            raise SystemExit(f"run folder not found: {d} (pass the folder that holds it with --runs)")
        f = read_table(d, "features_track")[["eid", "t", "speed", "alt", "rad_rcs_med30"]]
        m = g.merge(f, on=["eid", "t"], how="left")
        if m.speed.isna().all():
            raise SystemExit(f"{run}: no features found for the predicted rows")
        out.append(m.assign(obs_tier=observable_tier(m))[KEY + ["obs_tier"]])
    return pd.concat(out, ignore_index=True)


def route_split(prefix, d, name, route, runs_root, log):
    """Combine one split (prefix 'pred' or 'valpred'); returns the routed frame and a usage table."""
    frames = {}
    for model in sorted(set(route.values())):
        f = d / f"{prefix}_{model}_{name}.csv"
        if not f.exists():
            raise SystemExit(f"missing {f.name} (train that model first, or check the name)")
        frames[model] = pd.read_csv(f).set_index(KEY).sort_index()
    idx = None
    for model, fr in frames.items():
        if fr.index.duplicated().any():
            raise SystemExit(f"{prefix}_{model}_{name}.csv has duplicate (run, eid, t) rows")
        if idx is None:
            idx = fr.index
        elif not fr.index.equals(idx):
            raise SystemExit(f"{prefix}_{model}_{name}.csv covers different rows from the other models: "
                             "were they trained on the same runs and split?")
    tiers = observed_tiers(idx.to_frame(index=False), runs_root).set_index(KEY).reindex(idx)
    out = pd.DataFrame(index=idx, columns=P_COLS, dtype=float)
    usage = []
    for tier, model in sorted(route.items()):
        m = (tiers.obs_tier == tier).values
        out.loc[m, P_COLS] = frames[model].loc[m, P_COLS].values
        usage.append(dict(split=prefix, observed_tier=tier, model=model, rows=int(m.sum())))
    if out[P_COLS].isna().any().any():
        raise SystemExit("internal error: some rows were not routed")
    return out.reset_index(), pd.DataFrame(usage)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results", help="folder holding the pred_ / valpred_ files (output goes here too)")
    ap.add_argument("--name", required=True, help="scenario or shift name in the file names, e.g. S12_hypersonic")
    ap.add_argument("--runs", default="runs", help="folder holding the run folders named in the predictions")
    ap.add_argument("--route", nargs="+", required=True, help="TIER=MODEL for tiers 1, 2 and 3")
    ap.add_argument("--tag", required=True, help="output model name: letters, digits and - only (e.g. route-eg-gru)")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9-]+", a.tag):
        raise SystemExit(f"--tag {a.tag!r}: letters, digits and '-' only (no '_')")
    route = parse_route(a.route)
    d = Path(a.dir)
    usage = []
    for prefix in ("valpred", "pred"):
        if prefix == "valpred" and not all((d / f"valpred_{m}_{a.name}.csv").exists() for m in set(route.values())):
            print("note: not every routed model has a valpred_ file, so no valpred_ is written; "
                  "the routed model can be scored and prioritized but not used in a STACK")
            continue
        df, u = route_split(prefix, d, a.name, route, a.runs, print)
        path = d / f"{prefix}_{a.tag.lower()}_{a.name}.csv"
        df.to_csv(path, index=False)
        usage.append(u)
        print(f"wrote {path.name}: {len(df):,} rows")
    u = pd.concat(usage, ignore_index=True)
    print("\nrows routed by observed tier:")
    print(u.to_string(index=False))


if __name__ == "__main__":
    main()
