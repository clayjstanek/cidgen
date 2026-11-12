"""GNN data pipeline, numpy only (no torch needed to build or inspect the cache).

For each run it stores
  cube[track, grid_step, F]   per-track feature history (float16), the same encoding as graph_export
  present[track, grid_step]   whether the track was held at that step
  snapshots                   one entry per snapshot time: node rows into the cube, edges, labels

Sequences for the temporal encoder are cut from the cube at load time, so the cache stays small.

Optional track-track edge options (off by default; a cache built with options gets its own file):
  comove=True          six history-based co-movement features appended to each (track, near, track)
                       edge, from the tracker's position / velocity estimates over the last 60 s (no truth):
                       range rate, relative speed / own speed, closest-approach distance and time,
                       spread of the relative offset, and time spent close together. Formation mates are
                       stable and long-lived; tracks crossing or intercepting close fast and briefly.
  tier1_gate_km=x      drop edges longer than x km that touch a track whose OBSERVED tier is 1
                       (speed < 60 m/s, altitude < 1500 m, RCS < -5 dBsm, as in models.bn.observable_tier).
                       It can only remove edges: the generator's 2 km Tier 1 gate is the upper limit.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from ..graph_export import CATEGORICAL, MASK_GROUPS, TRACK_NUMERIC, RunGraphs, _signed_log, track_matrix
from ..io import read_meta

CACHE_FORMAT = 2


COMOVE_WINDOW_S = 60.0
COMOVE_COLS = ["range_rate_mps", "rel_speed_frac", "cpa_m", "tcpa_s", "offset_spread_m", "together_frac"]


def edge_opts_tag(comove=False, tier1_gate_km=None):
    """'' for the default edges, else a short tag used in cache file names and fingerprints."""
    parts = (["cm"] if comove else []) + ([f"t1g{int(round(1000 * tier1_gate_km))}m"] if tier1_gate_km else [])
    return "-".join(parts)


def fingerprint(run_dir, comove=False, tier1_gate_km=None):
    """Identifies the exact run contents and feature layout a cache entry was built from.
    A cache entry whose fingerprint differs is rebuilt, so stale entries can never be reused."""
    meta = read_meta(run_dir)
    layout = repr((TRACK_NUMERIC, sorted(MASK_GROUPS.items()), sorted((k, list(v)) for k, v in CATEGORICAL.items())))
    key = f"{CACHE_FORMAT}|{meta.get('generator_version')}|{meta.get('config_hash')}|{meta.get('seed')}|{layout}"
    opts = edge_opts_tag(comove, tier1_gate_km)
    if opts:                                   # default caches keep their original fingerprint
        key += f"|{opts}|{COMOVE_WINDOW_S}|{COMOVE_COLS}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _observed_tier1(f_t):
    """Observed Tier 1 per track at one time (same rule and missing-value handling as models.bn.observable_tier)."""
    num = lambda c: np.nan_to_num(f_t[c].to_numpy(dtype=float), nan=0.0)
    sp, alt, rcs = num("speed"), num("alt"), num("rad_rcs_med30")
    return (sp < 60) & (alt < 1500) & (rcs < -5) & (sp < 1000.0)


def _comove_features(P, V, held, src, dst, step, win):
    """(E, 6) co-movement features for edges src -> dst (cube rows) at grid step `step`."""
    dP = P[dst, step] - P[src, step]; dV = V[dst, step] - V[src, step]
    r = np.linalg.norm(dP, axis=1)
    rr = np.einsum("ij,ij->i", dP, dV) / np.maximum(r, 1.0)
    vmax = np.maximum(np.maximum(np.linalg.norm(V[src, step], axis=1), np.linalg.norm(V[dst, step], axis=1)), 1.0)
    rel = np.clip(np.linalg.norm(dV, axis=1) / vmax, 0, 5)
    vv = np.einsum("ij,ij->i", dV, dV)
    tc = np.clip(-np.einsum("ij,ij->i", dP, dV) / np.maximum(vv, 1e-6), 0.0, 600.0)
    cpa = np.linalg.norm(dP + dV * tc[:, None], axis=1)
    idx = np.arange(max(0, step - win + 1), step + 1)
    both = held[src][:, idx] & held[dst][:, idx]                                   # (E, w)
    off = P[dst][:, idx] - P[src][:, idx]                                          # (E, w, 3)
    n = both.sum(1)
    mean = np.where(n[:, None] > 0, (np.nan_to_num(off) * both[..., None]).sum(1) / np.maximum(n, 1)[:, None], 0.0)
    dev = np.linalg.norm(np.nan_to_num(off) - mean[:, None, :], axis=2)
    spread = np.where(n >= 4, np.sqrt((dev ** 2 * both).sum(1) / np.maximum(n, 1)), np.nan)
    close = (np.linalg.norm(np.nan_to_num(off), axis=2) < np.maximum(1.5 * r, 500.0)[:, None]) & both
    together = close.sum(1) / len(idx)
    out = np.stack([_signed_log(rr), rel, _signed_log(cpa), _signed_log(tc), _signed_log(spread), together], 1)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)

EDGE_TYPES = [("track", "near", "track"), ("track", "threatens", "asset"), ("asset", "threatened_by", "track"),
              ("esm_fix", "candidate", "track"), ("track", "hears", "esm_fix")]


def build_run_cache(run_dir, cache_dir, overwrite=False, comove=False, tier1_gate_km=None):
    run_dir = Path(run_dir); cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    opts = edge_opts_tag(comove, tier1_gate_km)
    out = cache_dir / (f"{run_dir.name}__{opts}.pkl" if opts else f"{run_dir.name}.pkl")
    fp = fingerprint(run_dir, comove, tier1_gate_km)
    if out.exists() and not overwrite:
        try:
            with open(out, "rb") as fh:
                if pickle.load(fh).get("fingerprint") == fp:
                    return out
        except Exception:
            pass
        print(f"cache: {out.name} is stale or unreadable; rebuilding from {run_dir}", flush=True)
    rg = RunGraphs(run_dir)
    meta = read_meta(run_dir)
    f = rg.f
    X, cols = track_matrix(f)
    eids = np.sort(f.eid.unique()); e_idx = {e: i for i, e in enumerate(eids)}
    dt = meta["config"]["run"]["feature_dt_s"]
    n_grid = int(round(meta["config"]["run"]["horizon_s"] / dt))
    ti = np.round(f.t.values / dt).astype(int)
    ri = f.eid.map(e_idx).values
    cube = np.zeros((len(eids), n_grid, X.shape[1]), np.float16)
    present = np.zeros((len(eids), n_grid), bool)
    cube[ri, ti] = X.astype(np.float16)
    present[ri, ti] = True
    if opts:                                   # tracker position / velocity history for edge options
        P = np.full((len(eids), n_grid, 3), np.nan, np.float32); V = np.full_like(P, np.nan)
        P[ri, ti] = f[["e", "n", "u"]].to_numpy(dtype=np.float32)
        V[ri, ti] = f[["ve", "vn", "vu"]].to_numpy(dtype=np.float32)
        win = max(1, int(round(COMOVE_WINDOW_S / dt)))
    tt_key = ("track", "near", "track")
    n_edges = [0, 0]
    lab = rg.lab.set_index(["eid", "t"])
    snaps = []
    for t in rg.times:
        s = rg.snapshot(float(t))
        if len(s.eid) == 0:
            continue
        cls = np.asarray(lab.loc[list(zip(s.eid, [t] * len(s.eid))), "cls"].astype(str).to_numpy(), dtype=object)
        if opts:
            ei, ea = s.edges[tt_key]
            rows = np.array([e_idx[e] for e in s.eid]); step = int(round(t / dt))
            n_edges[0] += ei.shape[1]
            if tier1_gate_km and ei.shape[1]:
                f_t = f[np.isclose(f.t, t)].set_index("eid").loc[s.eid]
                t1 = _observed_tier1(f_t)
                dist = np.linalg.norm(P[rows[ei[1]], step] - P[rows[ei[0]], step], axis=1)
                keep = ~((t1[ei[0]] | t1[ei[1]]) & (dist > 1000.0 * tier1_gate_km))
                ei, ea = ei[:, keep], ea[keep]
            if comove:
                cm = _comove_features(P, V, present, rows[ei[0]], rows[ei[1]], step, win) if ei.shape[1] \
                    else np.zeros((0, len(COMOVE_COLS)), np.float32)
                ea = np.concatenate([ea, cm], 1).astype(np.float32)
            n_edges[1] += ei.shape[1]
            s.edges[tt_key] = (np.ascontiguousarray(ei), ea)
        snaps.append(dict(t=float(t), step=int(round(t / dt)), rows=np.array([e_idx[e] for e in s.eid]),
                          eid=s.eid, y=s.y, y_multi=s.y_multi, y_threat=s.y_threat, cls=cls,
                          x_asset=s.x_asset, x_fix=s.x_fix, edges=s.edges))
    blob = dict(run=run_dir.name, scenario=meta["scenario"], seed=meta["seed"], cols=cols, cube=cube,
                present=present, snaps=snaps, fingerprint=fp,
                edge_opts=dict(comove=comove, tier1_gate_km=tier1_gate_km, edges_before=n_edges[0], edges_after=n_edges[1]))
    with open(out, "wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_cache(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def sequence(blob, snap, window):
    """(N, window, F + 1) history ending at the snapshot; last channel = 'track held at this step'."""
    rows, step = snap["rows"], snap["step"]
    lo = step - window + 1
    idx = np.arange(lo, step + 1)
    valid = idx >= 0
    seq = np.zeros((len(rows), window, blob["cube"].shape[2] + 1), np.float32)
    if valid.any():
        ii = idx[valid]
        seq[:, valid, :-1] = blob["cube"][rows][:, ii].astype(np.float32)
        seq[:, valid, -1] = blob["present"][rows][:, ii]
    return seq


def fit_normalizer(blobs):
    """Per-feature mean / std of node features and edge attributes over training snapshots."""
    widths = {b["cube"].shape[2] for b in blobs}
    if len(widths) > 1:
        raise ValueError(f"cache entries have different feature counts {sorted(widths)}: delete the cache folder")
    xs = [b["cube"][s["rows"], s["step"]].astype(np.float64) for b in blobs for s in b["snaps"]]
    X = np.concatenate(xs)
    norm = dict(x_mean=X.mean(0), x_std=np.maximum(X.std(0), 1e-3))
    for et in EDGE_TYPES:
        ea = [s["edges"][et][1] for b in blobs for s in b["snaps"] if len(s["edges"][et][1])]
        if ea:
            A = np.concatenate(ea).astype(np.float64)
            norm[et] = (A.mean(0), np.maximum(A.std(0), 1e-3))
    fx = [s["x_fix"] for b in blobs for s in b["snaps"] if len(s["x_fix"])]
    if fx:
        F = np.concatenate(fx).astype(np.float64)
        norm["fix"] = (F.mean(0), np.maximum(F.std(0), 1e-3))
    norm = {k: (tuple(np.asarray(x, np.float32) for x in v) if isinstance(v, tuple) else np.asarray(v, np.float32))
            for k, v in norm.items()}
    return norm


def class_weights(blobs):
    y = np.concatenate([s["y"] for b in blobs for s in b["snaps"]])
    y = y[y >= 0]
    cnt = np.bincount(y, minlength=3).astype(float)
    w = cnt.sum() / np.maximum(cnt, 1) / 3
    return w / w.mean()
