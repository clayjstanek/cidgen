"""Recreate validation predictions from saved GNN / GRU models, without retraining.

    python -m cidgen.models.predict results_v3/gnn_S0_baseline__on__S9_sensor_shift.pt --cache <cache> --out results_transfer --check-test
    python -m cidgen.models.predict "results_transfer/*__on__S9_sensor_shift.pt" --cache <cache> --out results_transfer --check-test

Ensembles must be fitted on validation predictions (training scenario), never on test predictions.
Models trained before v0.5 saved only test predictions, so this reloads each .pt file, rebuilds its
exact validation split from the settings stored inside it, and writes valpred_<model>_<name>.csv.

--check-test also recomputes the test predictions and compares them with the existing
pred_<model>_<name>.csv, which confirms the model was reloaded exactly (differences should be
below 0.001; GPU arithmetic is not bit-for-bit repeatable).

Run from the same folder you trained from: the run patterns stored in the model are relative to it.
Use --runs / --test-runs to override them if you have moved the run folders.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from ..io import read_meta
from .common import expand, split_runs
from .gnn import CIDGNN, SnapshotDataset, predict
from .gnn_data import build_run_cache, load_cache
from .train_gnn import drop_list, write_preds

DEFAULTS = dict(drop_features=None, sensor_dropout=0.0, tag=None, test_runs=None, val_frac=0.15, test_frac=0.15,
                window=15, hidden=128, layers=2, batch=16, workers=0, seed=0)


def rebuild_split(ckpt, name, runs=None, test_runs=None):
    """The split the model was trained with: stored in v0.5+ models, else rebuilt from its settings."""
    a = ckpt["args"]
    if ckpt.get("split") and not runs and not test_runs:
        sp = ckpt["split"]
        missing = [r for v in sp.values() for r in v if not (Path(r) / "scenario_meta.json").exists()]
        if not missing:
            return sp
        print(f"  {len(missing)} run folders saved in the model were not found (e.g. {missing[0]}); "
              f"rebuilding the split from the run patterns")
    runs = expand(runs or a["runs"])
    test_pat = test_runs or a.get("test_runs")
    if "__on__" in name:
        if not test_pat:
            raise SystemExit(f"{name}: shift-mode model but no test runs stored; pass --test-runs")
        sp = split_runs(runs, a.get("val_frac", 0.15), 0.0)
        sp["test"] = expand(test_pat)
        return sp
    runs = [r for r in runs if read_meta(r)["scenario"] == name]      # per-scenario mode: one model per scenario
    if not runs:
        raise SystemExit(f"{name}: none of the stored run patterns hold this scenario; pass --runs")
    return split_runs(runs, a.get("val_frac", 0.15), a.get("test_frac", 0.15))


def recover(pt, cache, out_dir, device="cuda", runs=None, test_runs=None, check_test=False, log=print):
    pt = Path(pt)
    tag, name = pt.stem.split("_", 1)
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    a = SimpleNamespace(**{**DEFAULTS, **ckpt["args"]})
    split = rebuild_split(ckpt, name, runs, test_runs)
    log(f"\n{pt.name}: model {tag}, {name}; val runs {', '.join(Path(r).name for r in split['val'])}")
    if not split["val"]:
        raise SystemExit("  no validation runs in this split")
    want = ["val"] + (["test"] if check_test else [])
    blobs = {k: [load_cache(build_run_cache(r, cache, comove=getattr(a, "comove_features", False),
                                            tier1_gate_km=getattr(a, "tier1_gate_km", None)))
                 for r in split[k]] for k in want}
    drop = drop_list(a)
    ds = {k: SnapshotDataset(blobs[k], ckpt["norm"], a.window, sensor_dropout=0.0, seed=a.seed, drop_features=drop)
          for k in want}
    n_feat = ckpt.get("n_feat", blobs["val"][0]["cube"].shape[2])
    n_fix = ckpt.get("n_fix", ds["val"].n_fix)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = CIDGNN(n_feat, n_fix=n_fix, hidden=a.hidden, layers=a.layers, graph=not a.no_graph,
                   tt_edge_dim=ckpt.get("tt_edge_dim", 10), tt_conv="gated" if getattr(a, "edge_gate", False) else "gatv2")
    model.load_state_dict(ckpt["state"])
    model.to(dev)
    T = float(ckpt["temperature"])
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    lv, yv, rv, ev, tv = predict(model, DataLoader(ds["val"], batch_size=a.batch, shuffle=False), dev)
    dfv, pv = write_preds(lv, rv, ev, tv, T, [b["run"] for b in blobs["val"]], out_dir / f"valpred_{tag}_{name}.csv")
    log(f"  wrote valpred_{tag}_{name}.csv: {len(dfv):,} track-snapshots, accuracy {(pv.argmax(1) == yv.numpy()).mean():.3f}"
        f" (temperature {T:.3f}{', inputs removed: ' + ', '.join(drop) if drop else ''})")
    res = dict(model=tag, name=name, n_val=len(dfv), val_acc=float((pv.argmax(1) == yv.numpy()).mean()))
    if check_test:
        old = None
        for d in (out_dir, pt.parent):
            f = d / f"pred_{tag}_{name}.csv"
            if f.exists():
                old = pd.read_csv(f); break
        lt, yt, rt, et, tt = predict(model, DataLoader(ds["test"], batch_size=a.batch, shuffle=False), dev)
        p = torch.softmax(lt / T, -1).numpy()
        new = pd.DataFrame(dict(run=[blobs["test"][i]["run"] for i in rt.numpy()], eid=et.numpy(),
                                t=tt.numpy().astype(float), q_Friend=p[:, 0], q_Neutral=p[:, 1], q_Hostile=p[:, 2]))
        if old is None:
            log(f"  check: pred_{tag}_{name}.csv not found next to the model or in --out; nothing to compare")
        else:
            m = old.merge(new, on=["run", "eid", "t"], how="inner")
            diff = np.abs(m[["p_Friend", "p_Neutral", "p_Hostile"]].values - m[["q_Friend", "q_Neutral", "q_Hostile"]].values)
            agree = float((m[["p_Friend", "p_Neutral", "p_Hostile"]].values.argmax(1)
                           == m[["q_Friend", "q_Neutral", "q_Hostile"]].values.argmax(1)).mean())
            ok = len(m) == len(old) == len(new) and diff.max() < 1e-3
            log(f"  check against saved test predictions: {len(m):,} of {len(old):,} rows matched, largest difference "
                f"{diff.max():.2e}, same decision on {agree:.2%} -> {'OK' if ok else 'MISMATCH: do not use this valpred file'}")
            res.update(check_rows=len(m), check_max_diff=float(diff.max()), check_ok=bool(ok))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help=".pt files or glob patterns (quote them)")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default=None, help="where to write valpred files (default: next to each model)")
    ap.add_argument("--runs", nargs="+", default=None, help="override the training run patterns stored in the model")
    ap.add_argument("--test-runs", nargs="+", default=None, help="override the test run patterns stored in the model")
    ap.add_argument("--check-test", action="store_true", help="also recompute test predictions and compare")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    pts = set()
    for pat in a.models:
        hits = [p for p in glob.glob(pat) if p.endswith(".pt")]
        if not hits:
            print(f"note: no .pt files match {pat}; skipped")
        pts.update(hits)
    pts = sorted(pts)
    if not pts:
        raise SystemExit(f"no .pt files match {a.models}")
    rows = [recover(p, a.cache, a.out or Path(p).parent, a.device, a.runs, a.test_runs, a.check_test) for p in pts]
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    if any(r.get("check_ok") is False for r in rows):
        raise SystemExit("one or more models did not reproduce their saved test predictions (see MISMATCH above)")


if __name__ == "__main__":
    main()
