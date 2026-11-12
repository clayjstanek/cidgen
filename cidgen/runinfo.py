"""List run folders with the conditions that matter for transfer experiments.

    python -m cidgen.runinfo runs_transfer
    python -m cidgen.runinfo runs runs_transfer --csv run_inventory.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .io import read_meta

FIELDS = [("world", "visibility", "EO/IR visibility"), ("sensors", "radar_noise_scale", "radar noise"),
          ("sensors", "n_R_S", "Sentinel radars"), ("ambiguity", "p_ESM_correct", "ESM correct"),
          ("ambiguity", "p_nctr_persistent_error", "NCTR misread"), ("iff", "outage_fraction", "IFF outage"),
          ("composition", "n_hostile_groups", "strike packages"), ("composition", "n_drone_swarms", "drone swarms")]


def inventory(folders):
    rows = []
    for f in folders:
        for run in sorted(Path(f).glob("*/scenario_meta.json")):
            d = run.parent
            meta = read_meta(d)
            cfg = yaml.safe_load(open(d / "config_resolved.yaml")) if (d / "config_resolved.yaml").exists() else meta["config"]
            r = dict(folder=str(Path(f)), run=d.name, scenario=meta["scenario"], seed=meta["seed"],
                     version=meta.get("generator_version"), format=meta.get("format"))
            for sec, key, label in FIELDS:
                r[label] = cfg.get(sec, {}).get(key)
            r["randomised"] = bool(cfg.get("randomized_values"))
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--csv", default=None, help="also write the full per-run list here")
    a = ap.parse_args()
    df = inventory(a.folders)
    if df.empty:
        raise SystemExit("no runs found")
    if a.csv:
        df.to_csv(a.csv, index=False)
    num = [lab for _, _, lab in FIELDS]
    summary = df.groupby(["folder", "scenario"]).agg(
        runs=("run", "size"), seeds=("seed", lambda s: f"{min(s)}-{max(s)}"),
        versions=("version", lambda s: ", ".join(sorted(set(map(str, s))))),
        **{lab: (lab, lambda s: f"{s.min():g}" if s.min() == s.max() else f"{s.min():.3g}-{s.max():.3g}") for lab in num})
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string())


if __name__ == "__main__":
    main()
