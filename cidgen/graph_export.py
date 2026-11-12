"""Graph snapshots for GNN training (spec Section 6.3).

Two layers:
  build_snapshot_arrays()  pure numpy: node features, edge indices, labels for one time t
  to_heterodata()          wraps those arrays as a torch_geometric HeteroData

Node types   track, asset, esm_fix
Edge types   (track, near, track)            gated k-nearest neighbours, edge_attr = relative kinematics
             (track, threatens, asset)       range / closure / TTG / CPA
             (asset, threatened_by, track)   reverse of the above
             (esm_fix, candidate, track)     fix-to-track gate membership
             (track, hears, esm_fix)         reverse of the above

DGL users: HeteroData ~ dgl.heterograph. data['track'].x ~ g.nodes['track'].data['feat'];
data['track','near','track'].edge_index is a 2 x E tensor (row 0 = src, row 1 = dst), i.e. torch.stack(g.edges()).

Truth columns (allegiance, class, merged_eids, eid_true, is_ghost, ...) are only ever used as labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io import read_table
from .library import EMITTER_CODES, ENGINE_CODES, EOIR_CLASSES

# numeric per-track features; distances are log-scaled, missing values get an explicit mask
TRACK_NUMERIC = ["speed", "alt", "climb", "turn_rate", "accel", "pos_sigma_m", "vel_sigma_mps", "age_s",
                 "rad_rcs_med30", "rad_rcs_std30", "rad_md_rate", "rad_md_spread", "rad_md_period",
                 "rad_jam_flag", "rad_has_R_C", "rad_has_R_S", "rad_has_R_P",
                 "iff_mode3c", "iff_mode5", "iff_mode5_valid", "iff_level2", "iff_age_s", "iff_l2_resid_m",
                 "iff_ever_valid", "iff_in_coverage",
                 "jem_top_conf", "jem_bpf_feature", "jem_age_s",
                 "lidar_len_fuselage_m", "lidar_len_span_m", "lidar_len_tail_m", "lidar_age_s",
                 "eoir_top_conf", "eoir_friend_variant_recognized", "eoir_ir_intensity", "eoir_age_s",
                 "esm_conf", "esm_band", "esm_dist_norm", "esm_n_tracks_in_gate", "esm_age_s",
                 "airway_xtrack_m", "airway_alt_inband", "corridor_dist_m", "threat_offset_deg", "range_origin_m",
                 "asset_range_m", "asset_closure_mps", "asset_min_ttg_s", "asset_min_cpa_m", "descending_closing"]
LOG_COLS = {"alt", "range_origin_m", "pos_sigma_m", "iff_l2_resid_m", "airway_xtrack_m", "corridor_dist_m", "asset_range_m",
            "asset_min_ttg_s", "asset_min_cpa_m", "esm_n_tracks_in_gate", "iff_age_s", "jem_age_s", "lidar_age_s",
            "eoir_age_s", "esm_age_s", "age_s"}
MASK_GROUPS = {"iff": "iff_mode3c", "jem": "jem_top_conf", "lidar": "lidar_age_s", "eoir": "eoir_top_conf",
               "esm": "esm_conf", "md": "rad_md_rate"}
CATEGORICAL = {"jem_top_engine": ENGINE_CODES, "eoir_top_class": EOIR_CLASSES, "esm_top_emitter": EMITTER_CODES}
TT_EDGE = ["de", "dn", "du", "dve", "dvn", "dvu", "dist_m", "heading_diff_deg", "speed_ratio", "comove_corr"]
TA_EDGE = ["range_m", "closure_mps", "ttg_s", "cpa_m", "tcpa_s", "heading_offset_deg", "asset_value"]


def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def track_matrix(feats: pd.DataFrame):
    """Encode per-track features: numeric (log-scaled where marked), NaN masks, one-hot categoricals."""
    cols, blocks = [], []
    for c in TRACK_NUMERIC:
        v = pd.to_numeric(feats[c], errors="coerce").astype(float).values if c in feats else np.full(len(feats), np.nan)
        v = np.where(np.isinf(v), np.nan, v)
        if c in LOG_COLS:
            v = _signed_log(v)
        blocks.append(np.nan_to_num(v, nan=0.0)); cols.append(c)
    for g, c in MASK_GROUPS.items():
        blocks.append(feats[c].notna().values.astype(float) if c in feats else np.zeros(len(feats))); cols.append(f"has_{g}")
    for c, codes in CATEGORICAL.items():
        vals = feats[c].astype(str).values if c in feats else np.array([""] * len(feats))
        for code in codes:
            blocks.append((vals == code).astype(float)); cols.append(f"{c}={code}")
    return np.stack(blocks, axis=1).astype(np.float32), cols


@dataclass
class SnapshotArrays:
    t: float
    eid: np.ndarray
    x_track: np.ndarray
    track_cols: list
    y: np.ndarray                     # allegiance 0/1/2, -1 for non-objects-of-interest (birds)
    y_threat: np.ndarray
    y_multi: np.ndarray
    tier: np.ndarray
    x_asset: np.ndarray
    x_fix: np.ndarray
    edges: dict = field(default_factory=dict)        # (src, rel, dst) -> (edge_index (2,E), edge_attr (E,k))
    fix_is_ghost: np.ndarray | None = None           # label only


class RunGraphs:
    """Loads one run's tables once and serves snapshots at any snapshot time."""

    def __init__(self, run_dir):
        self.run = run_dir
        self.f = read_table(run_dir, "features_track")
        self.lab = read_table(run_dir, "labels")
        self.tt = read_table(run_dir, "edges_track_track")
        self.ta = read_table(run_dir, "edges_track_asset")
        self.et = read_table(run_dir, "edges_esm_track")
        self.fx = read_table(run_dir, "esm_fixes")
        self.assets = read_table(run_dir, "assets")
        self.times = np.array(sorted(self.tt.t.unique())) if len(self.tt) else np.array([])

    def snapshot(self, t: float) -> SnapshotArrays:
        f = self.f[np.isclose(self.f.t, t)].reset_index(drop=True)
        lab = self.lab[np.isclose(self.lab.t, t)].set_index("eid").loc[f.eid]
        x, cols = track_matrix(f)
        node = {e: i for i, e in enumerate(f.eid.values)}
        edges = {}
        tt = self.tt[np.isclose(self.tt.t, t)]
        tt = tt[tt.src_eid.isin(node) & tt.dst_eid.isin(node)]
        ei = np.stack([tt.src_eid.map(node).values, tt.dst_eid.map(node).values]).astype(np.int64)
        ea = tt[TT_EDGE].to_numpy(dtype=float, copy=True)
        ea[:, :7] = _signed_log(ea[:, :7]); ea = np.nan_to_num(ea, nan=0.0)
        edges[("track", "near", "track")] = (ei, ea.astype(np.float32))
        ta = self.ta[np.isclose(self.ta.t, t) & self.ta.eid.isin(node)]
        ei = np.stack([ta.eid.map(node).values, ta.asset_id.values]).astype(np.int64)
        ea = ta[TA_EDGE].to_numpy(dtype=float, copy=True)
        ea[:, [0, 2, 3, 4]] = _signed_log(ea[:, [0, 2, 3, 4]]); ea = np.nan_to_num(ea, nan=0.0).astype(np.float32)
        edges[("track", "threatens", "asset")] = (ei, ea)
        edges[("asset", "threatened_by", "track")] = (ei[::-1].copy(), ea)
        # ESM fixes heard in the last 4 s
        fx = self.fx[(self.fx.time > t - 4.0) & (self.fx.time <= t)].reset_index(drop=True)
        fnode = {fid: i for i, fid in enumerate(fx.fix_id.values)}
        em = np.stack([(fx.top_emitter.astype(str).values == c).astype(float) for c in EMITTER_CODES], 1) if len(fx) \
            else np.zeros((0, len(EMITTER_CODES)))
        x_fix = np.concatenate([np.stack([_signed_log(fx.sigma_m.values.astype(float)), fx.conf.values.astype(float),
                                          fx.band.values.astype(float), fx.n_sites.values.astype(float),
                                          _signed_log(fx.u.values.astype(float))], 1) if len(fx) else np.zeros((0, 5)),
                                em], 1).astype(np.float32)
        et = self.et[np.isclose(self.et.t, t) & self.et.eid.isin(node) & self.et.fix_id.isin(fnode)]
        ei = np.stack([et.fix_id.map(fnode).values, et.eid.map(node).values]).astype(np.int64)
        ea = et[["dist_norm"]].to_numpy(dtype=np.float32, copy=True)
        edges[("esm_fix", "candidate", "track")] = (ei, ea)
        edges[("track", "hears", "esm_fix")] = (ei[::-1].copy(), ea)
        x_asset = self.assets[["value", "radius_m", "e", "n"]].to_numpy(dtype=float, copy=True)
        x_asset[:, 1:] = _signed_log(x_asset[:, 1:])
        return SnapshotArrays(t=t, eid=f.eid.to_numpy(dtype=np.int64, copy=True), x_track=x, track_cols=cols, y=lab.allegiance.values.astype(np.int64),
                              y_threat=lab.threat_priority.values.astype(np.float32),
                              y_multi=lab.multi_aircraft.astype(bool).values.astype(np.int64),
                              tier=lab.tier.values.astype(np.int64), x_asset=x_asset.astype(np.float32),
                              x_fix=x_fix, edges=edges, fix_is_ghost=fx.is_ghost.to_numpy(dtype=bool, copy=True))


def to_heterodata(s: SnapshotArrays, device=None):
    """Wrap a snapshot as torch_geometric.data.HeteroData (requires torch + torch_geometric)."""
    import torch
    from torch_geometric.data import HeteroData
    d = HeteroData()
    c = lambda a: np.array(a, copy=True)        # pandas copy-on-write can hand back read-only arrays
    d["track"].x = torch.from_numpy(c(s.x_track))
    d["track"].y = torch.from_numpy(c(s.y))
    d["track"].y_threat = torch.from_numpy(c(s.y_threat))
    d["track"].y_multi = torch.from_numpy(c(s.y_multi))
    d["track"].tier = torch.from_numpy(c(s.tier))
    d["track"].eid = torch.from_numpy(s.eid.astype(np.int64))
    d["asset"].x = torch.from_numpy(c(s.x_asset))
    d["esm_fix"].x = torch.from_numpy(c(s.x_fix))
    for (src, rel, dst), (ei, ea) in s.edges.items():
        d[src, rel, dst].edge_index = torch.from_numpy(np.array(ei, dtype=np.int64, order="C", copy=True))
        d[src, rel, dst].edge_attr = torch.from_numpy(np.array(ea, dtype=np.float32, order="C", copy=True))
    d.t = float(s.t)
    return d.to(device) if device is not None else d


def iter_snapshots(run_dirs, every_n=1):
    """Yield SnapshotArrays across runs (every_n-th snapshot time)."""
    for rd in run_dirs:
        rg = RunGraphs(rd)
        for t in rg.times[::every_n]:
            yield rg.snapshot(float(t))
