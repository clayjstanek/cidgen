"""Train the GNN (or its temporal-only ablation) per scenario and write calibrated test predictions.

    python -m cidgen.models.train_gnn --runs "runs/S0_*" "runs/S1_*" --out results
    python -m cidgen.models.train_gnn --runs "runs/S0_*" --no-graph --out results     # GRU only
    python -m cidgen.models.train_gnn --runs "runs/S0_*" --test-runs "runs/S9_*" --out results  # shift
    python -m cidgen.models.train_gnn --runs "runs/S12_hypersonic_*" --out results --edge-gate --comove-features \
        --tier1-gate-km 0.5 --sensor-dropout 0.1 --drop-features track-quality    # gated graph variant

Graph options (all off by default, so existing commands and models are unchanged):
  --comove-features   add six history-based co-movement features to every track-track edge
  --tier1-gate-km X   drop track-track edges longer than X km that touch an observed Tier 1 track
  --edge-gate         gated track-track messages (GatedEdgeConv) instead of GATv2 attention
They add -cm, -t1gNNNm and -eg to the model name.

Writes results/pred_<model>_<scenario>.csv (run, eid, t, p_Friend, p_Neutral, p_Hostile) for the
test runs, results/valpred_<model>_<scenario>.csv for the validation runs (used to fit ensembles; see
models/ensemble.py), plus the model weights and training log. Splits come from models.common.split_runs, so
compare.py scores the BN and tree baselines on exactly the same test runs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from ..io import read_meta
from .common import expand, split_runs
from .gnn import CIDGNN, SnapshotDataset, fit_temperature, predict, train
from .gnn_data import build_run_cache, class_weights, fit_normalizer, load_cache


TRACK_QUALITY = ["pos_sigma_m", "vel_sigma_mps", "age_s"]


def drop_list(a):
    spec = getattr(a, "drop_features", None) or ""
    if spec == "track-quality":
        return list(TRACK_QUALITY)
    return [c.strip() for c in spec.split(",") if c.strip()]


def model_tag(a):
    """Output name: --tag if given, else gnn / gru, plus -sdNN for sensor dropout and -notq or -drop
    when inputs are removed."""
    import re
    spec = getattr(a, "drop_features", None) or ""
    tag = a.tag or (("gru" if a.no_graph else "gnn") + (f"-sd{round(100 * a.sensor_dropout)}" if a.sensor_dropout else "")
                    + ("-notq" if spec == "track-quality" else "-drop" if spec else "")
                    + ("-cm" if getattr(a, "comove_features", False) else "")
                    + (f"-t1g{int(round(1000 * a.tier1_gate_km))}m" if getattr(a, "tier1_gate_km", None) else "")
                    + ("-eg" if getattr(a, "edge_gate", False) else ""))
    if not re.fullmatch(r"[A-Za-z0-9-]+", tag):
        raise SystemExit(f"--tag '{tag}': use letters, digits and '-' only (no '_')")
    return tag


def write_preds(logits, run_idx, eid, t, T, names, path):
    """Calibrated probabilities with their (run, eid, t) keys, as read by compare.py and ensemble.py."""
    p = torch.softmax(logits / T, -1).numpy()
    df = pd.DataFrame(dict(run=[names[i] for i in run_idx.numpy()], eid=eid.numpy(), t=t.numpy().astype(float),
                           p_Friend=p[:, 0], p_Neutral=p[:, 1], p_Hostile=p[:, 2]))
    df.to_csv(path, index=False)
    return df, p


def run_one(scenario, runs, a, out_dir, log, split=None):
    split = split or split_runs(runs, a.val_frac, a.test_frac)
    for k, v in (("comove_features", False), ("tier1_gate_km", None), ("edge_gate", False)):
        if not hasattr(a, k):                  # callers that build their own args (tests, older scripts)
            setattr(a, k, v)
    log(f"\n=== {scenario}: train {len(split['train'])}, val {len(split['val'])}, test {len(split['test'])} runs")
    cache = Path(a.cache)
    blobs = {k: [load_cache(build_run_cache(r, cache, comove=a.comove_features, tier1_gate_km=a.tier1_gate_km))
                 for r in v] for k, v in split.items()}
    if a.comove_features or a.tier1_gate_km:
        eo = [b["edge_opts"] for v in blobs.values() for b in v]
        log(f"graph options: co-movement features {'on' if a.comove_features else 'off'}, Tier 1 gate "
            f"{str(a.tier1_gate_km) + ' km' if a.tier1_gate_km else 'off'}; track-track edges "
            f"{sum(e['edges_before'] for e in eo):,} -> {sum(e['edges_after'] for e in eo):,}")
    norm = fit_normalizer(blobs["train"])
    w = class_weights(blobs["train"])
    drop = drop_list(a)
    ds = {k: SnapshotDataset(v, norm, a.window, sensor_dropout=(a.sensor_dropout if k == "train" else 0.0), seed=a.seed,
                             drop_features=drop) for k, v in blobs.items()}
    if drop:
        log(f"inputs removed from every split: {', '.join(drop)}")
    if a.sensor_dropout > 0:
        log(f"sensor dropout {a.sensor_dropout:.2f} per family per track during training: "
            f"{', '.join(ds['train'].groups)}")
    log(f"snapshots: " + ", ".join(f"{k} {len(v)}" for k, v in ds.items()) + f" | class weights {np.round(w, 2)}")
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ld = {k: DataLoader(v, batch_size=a.batch, shuffle=(k == "train"), num_workers=a.workers)
          for k, v in ds.items()}
    n_feat = blobs["train"][0]["cube"].shape[2]
    tt_dim = next((np.asarray(s["edges"][("track", "near", "track")][1]).shape[1] for b in blobs["train"]
                  for s in b["snaps"] if len(s["edges"][("track", "near", "track")][1])), 10)
    model = CIDGNN(n_feat, n_fix=ds["train"].n_fix, hidden=a.hidden, layers=a.layers, graph=not a.no_graph,
                   tt_edge_dim=tt_dim, tt_conv="gated" if a.edge_gate else "gatv2").to(dev)
    log(f"model {'GRU only' if a.no_graph else 'GRU + hetero ' + ('gated-edge' if a.edge_gate else 'GATv2') + '/SAGE'}: "
        f"{sum(p.numel() for p in model.parameters()):,} parameters on {dev}")
    t0 = time.time()
    model = train(model, ld["train"], ld["val"], dev, w, epochs=a.epochs, lr=a.lr, patience=a.patience, log=log)
    log(f"trained in {time.time() - t0:.0f}s")
    lv, yv, rv, ev, tv = predict(model, ld["val"], dev)
    T = fit_temperature(lv, yv, log=log)
    log(f"temperature {T:.3f}")
    tag = model_tag(a)
    write_preds(lv, rv, ev, tv, T, [b["run"] for b in blobs["val"]], out_dir / f"valpred_{tag}_{scenario}.csv")
    lt, yt, rt, et, tt = predict(model, ld["test"], dev)
    df, p = write_preds(lt, rt, et, tt, T, [b["run"] for b in blobs["test"]], out_dir / f"pred_{tag}_{scenario}.csv")
    torch.save(dict(state=model.state_dict(), norm=norm, temperature=T, args=vars(a), n_feat=n_feat, tt_edge_dim=tt_dim,
                    n_fix=ds["train"].n_fix, split={k: [str(r) for r in v] for k, v in split.items()}),
               out_dir / f"{tag}_{scenario}.pt")
    acc = float((p.argmax(1) == yt.numpy()).mean())
    log(f"test accuracy {acc:.3f} on {len(df):,} track-snapshots (full metrics: python -m cidgen.models.compare)")
    return dict(scenario=scenario, model=tag, temperature=T, test_acc=acc, n_test=len(df))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders or glob patterns (quote them)")
    ap.add_argument("--test-runs", nargs="+", default=None,
                    help="shift mode: train/validate on --runs, test on these (e.g. S0 -> S9)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--no-graph", action="store_true", help="temporal-only ablation (no message passing)")
    ap.add_argument("--sensor-dropout", type=float, default=0.0,
                    help="training only: probability of hiding each sensor family per track (e.g. 0.3)")
    ap.add_argument("--drop-features", default=None,
                    help="remove inputs from every split: 'track-quality' (position and velocity uncertainty, track "
                         "age) or a comma-separated list of feature names")
    ap.add_argument("--tag", default=None, help="name for outputs (default gnn / gru); letters, digits and - only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=15, help="history steps (x feature_dt, 2 s)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comove-features", action="store_true",
                    help="add six history-based co-movement features to every track-track edge (own cache files)")
    ap.add_argument("--tier1-gate-km", type=float, default=None,
                    help="drop track-track edges longer than this that touch an observed Tier 1 track (e.g. 0.5)")
    ap.add_argument("--edge-gate", action="store_true",
                    help="gated track-track messages instead of GATv2 attention (a track can ignore its neighbours)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = model_tag(a)
    logf = open(out / f"train_{tag}.log", "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    runs = expand(a.runs)
    if a.test_runs:
        test = expand(a.test_runs)
        name = (f"{'+'.join(sorted({read_meta(r)['scenario'] for r in runs}))}__on__"
                f"{'+'.join(sorted({read_meta(r)['scenario'] for r in test}))}")
        split = split_runs(runs, a.val_frac, 0.0); split["test"] = test
        summary = [run_one(name, runs, a, out, log, split=split)]
        (out / f"summary_{tag}_{name}.json").write_text(json.dumps(summary, indent=2))
        return
    by = {}
    for r in runs:
        by.setdefault(read_meta(r)["scenario"], []).append(r)
    summary = [run_one(s, rs, a, out, log) for s, rs in sorted(by.items())]
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
