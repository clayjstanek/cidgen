"""Derived features (spec Section 5) computed from sensor reports and filtered track estimates only.

Outputs
  features_track     one row per (eid, t) on the feature grid while the track is held
  edges_track_track  gated track pairs at snapshot times
  edges_track_asset  track-to-asset relations at snapshot times
  edges_esm_track    ESM fix -> candidate track edges at snapshot times
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import angle_diff_deg, point_to_segment_dist


def _asof(left: pd.DataFrame, right: pd.DataFrame, cols, tol, prefix, time_col="time"):
    """Attach the most recent report (per eid, within tol seconds) to each track row."""
    if right is None or len(right) == 0:
        for c in cols:
            left[f"{prefix}{c}"] = np.nan
        left[f"{prefix}age_s"] = np.nan
        return left
    r = right[["eid", time_col] + cols].rename(columns={time_col: "_rt"}).sort_values("_rt")
    r = r.rename(columns={c: f"{prefix}{c}" for c in cols})
    l = left.sort_values("t")
    m = pd.merge_asof(l, r, left_on="t", right_on="_rt", by="eid", direction="backward", tolerance=tol)
    m[f"{prefix}age_s"] = m.t - m._rt
    return m.drop(columns="_rt")


def track_features(cfg, world, est: pd.DataFrame, radar_enu: pd.DataFrame, iff: pd.DataFrame,
                   jem: pd.DataFrame, lidar: pd.DataFrame, eoir: pd.DataFrame, esm_fixes: pd.DataFrame) -> pd.DataFrame:
    f = est.copy()
    f["speed"] = np.hypot(f.ve, f.vn)
    f["alt"] = f.u
    f["climb"] = f.vu
    f["heading"] = np.rad2deg(np.arctan2(f.ve, f.vn)) % 360.0
    f = f.sort_values(["eid", "t"])
    g = f.groupby("eid", sort=False)
    dt = g.t.diff()
    f["turn_rate"] = angle_diff_deg(f.heading, g.heading.shift()) / dt
    f["accel"] = g.speed.diff() / dt
    f.loc[dt > 3 * cfg["run"]["feature_dt_s"], ["turn_rate", "accel"]] = np.nan

    # --- radar-derived: RCS statistics, micro-Doppler, jam flag, radar coverage mix (30 s window)
    r = radar_enu.copy()
    r["tbin"] = np.floor(r.time / cfg["run"]["feature_dt_s"]) * cfg["run"]["feature_dt_s"]
    agg = r.groupby(["eid", "tbin"]).agg(rcs_med=("rcs_dbsm", "median"), rcs_n=("rcs_dbsm", "size"),
                                          md_rate=("md_rate", "mean"), md_spread=("md_spread", "mean"),
                                          md_period=("md_period", "mean"), jam_flag=("jam_flag", "max"),
                                          vr_abs=("vr_mps", lambda x: np.abs(x).mean()),
                                          has_R_C=("sensor_type", lambda x: (x == "R_C").any()),
                                          has_R_S=("sensor_type", lambda x: (x == "R_S").any()),
                                          has_R_P=("sensor_type", lambda x: (x == "R_P").any())).reset_index()
    agg = agg.rename(columns={"tbin": "time"})
    for c in ["jam_flag", "has_R_C", "has_R_S", "has_R_P"]:
        agg[c] = agg[c].astype(float)
    # 30 s rolling RCS median and spread per track
    agg = agg.sort_values(["eid", "time"])
    roll = agg.groupby("eid").rcs_med.rolling(15, min_periods=1)
    agg["rcs_med30"] = roll.median().reset_index(level=0, drop=True)
    agg["rcs_std30"] = agg.groupby("eid").rcs_med.rolling(15, min_periods=2).std().reset_index(level=0, drop=True)
    f = _asof(f, agg, ["rcs_med30", "rcs_std30", "md_rate", "md_spread", "md_period", "jam_flag",
                       "has_R_C", "has_R_S", "has_R_P"], 10.0, "rad_")

    # --- IFF: last reply (within 30 s), L2 position residual against the track estimate
    if len(iff):
        ii = iff.copy()
        for c in ["mode3c", "mode5", "mode5_valid", "level2"]:
            ii[c] = ii[c].astype(float)
        f = _asof(f, ii, ["mode3c", "mode5", "mode5_valid", "level2", "l2_e", "l2_n", "l2_u"], 30.0, "iff_")
        f["iff_l2_resid_m"] = np.sqrt((f.iff_l2_e - f.e) ** 2 + (f.iff_l2_n - f.n) ** 2 + (f.iff_l2_u - f.u) ** 2)
        f = f.drop(columns=["iff_l2_e", "iff_l2_n", "iff_l2_u"])
        f = f.sort_values(["eid", "t"])
        f["iff_ever_valid"] = f.groupby("eid").iff_mode5_valid.transform(lambda x: x.fillna(0).cummax())
    else:
        for c in ["iff_mode3c", "iff_mode5", "iff_mode5_valid", "iff_level2", "iff_age_s", "iff_l2_resid_m", "iff_ever_valid"]:
            f[c] = np.nan
    # in IFF coverage (any R-S within 1.5x its reference range) but silent -> informative absence
    rs = world.sites_of("R_S")
    if rs:
        rref = cfg["sensors"]["radar_types"]["R_S"]["ref_range_km"] * 1500.0
        d = np.min([np.hypot(f.e - s.pos[0], f.n - s.pos[1]) for s in rs], axis=0)
        f["iff_in_coverage"] = (d < rref).astype(float)
    else:
        f["iff_in_coverage"] = 0.0

    # --- NCTR: JEM, LIDAR, EO/IR (most recent within window)
    pj = [c for c in jem.columns if c.startswith("p_")] if len(jem) else []
    f = _asof(f, jem, ["top_engine", "top_conf", "bpf_feature"] + pj, 60.0, "jem_")
    f = _asof(f, lidar, ["len_fuselage_m", "len_span_m", "len_tail_m"], 60.0, "lidar_")
    pe = [c for c in eoir.columns if c.startswith("p_")] if len(eoir) else []
    f = _asof(f, eoir, ["top_class", "top_conf", "friend_variant_recognized", "ir_intensity"] + pe, 20.0, "eoir_")

    # --- ESM: best fix inside the track's gate in the last 4 s, and how many tracks share that gate
    f = _esm_gate_features(f, esm_fixes, tol=4.0)

    # --- context: airways, corridor, assets
    xy = f[["e", "n"]].values
    dmin = np.full(len(f), np.inf); inband = np.zeros(len(f))
    for aw in world.airways:
        d, _ = point_to_segment_dist(xy, aw.a, aw.b)
        better = d < dmin
        dmin = np.where(better, d, dmin)
        inband = np.where(better, ((f.u >= aw.alt_band_m[0] - 300) & (f.u <= aw.alt_band_m[1] + 300)).astype(float), inband)
    f["airway_xtrack_m"] = dmin
    f["airway_alt_inband"] = inband
    dc = np.full(len(f), np.inf)
    for a, b in zip(world.corridor[:-1], world.corridor[1:]):
        dc = np.minimum(dc, point_to_segment_dist(xy, a, b)[0])
    f["corridor_dist_m"] = dc
    brg = np.rad2deg(np.arctan2(f.e.values, f.n.values)) % 360.0
    f["threat_offset_deg"] = np.abs(angle_diff_deg(brg, world.threat_axis_deg))
    f["range_origin_m"] = np.hypot(f.e.values, f.n.values)
    A = np.stack([a.pos[:2] for a in world.assets])
    rel = xy[:, None, :] - A[None]                                   # track minus asset
    rng_a = np.linalg.norm(rel, axis=2)
    vxy = f[["ve", "vn"]].values
    clos = -np.einsum("ijk,ik->ij", rel, vxy) / np.maximum(rng_a, 1.0)
    j = np.argmin(rng_a, axis=1)
    ii = np.arange(len(f))
    f["asset_range_m"] = rng_a[ii, j]
    f["asset_closure_mps"] = clos[ii, j]
    ttg = np.where(clos > 1.0, rng_a / np.maximum(clos, 1e-3), np.inf)
    f["asset_min_ttg_s"] = np.minimum(np.min(ttg, axis=1), 1e5)
    vv = np.maximum((vxy ** 2).sum(1), 1e-6)
    tcpa = np.clip(-np.einsum("ijk,ik->ij", rel, vxy) / vv[:, None], 0, None)
    cpa = np.linalg.norm(rel + tcpa[..., None] * vxy[:, None, :], axis=2)
    f["asset_min_cpa_m"] = np.min(cpa, axis=1)
    f["descending_closing"] = ((f.climb < -5.0) & (f.asset_closure_mps > 20.0)).astype(float)
    f = f.sort_values(["eid", "t"])
    f["descending_closing"] = f.groupby("eid").descending_closing.transform(lambda x: x.rolling(10, min_periods=1).mean())
    return f.reset_index(drop=True)


def _esm_gate_features(f, fixes, tol):
    out_cols = ["esm_top_emitter", "esm_conf", "esm_band", "esm_dist_norm", "esm_n_tracks_in_gate", "esm_age_s"]
    for c in out_cols:
        f[c] = np.nan
    if fixes is None or len(fixes) == 0:
        return f
    f = f.sort_values("t").reset_index(drop=True)
    fx = fixes.sort_values("time")
    tg = f.t.values
    ft = fx.time.values
    # for every grid time, fixes within the last tol seconds
    lo = np.searchsorted(ft, tg - tol, "left"); hi = np.searchsorted(ft, tg, "right")
    em = np.empty(len(f), object); em[:] = None
    conf = np.full(len(f), np.nan); band = np.full(len(f), np.nan); dn = np.full(len(f), np.nan)
    ntr = np.full(len(f), np.nan); age = np.full(len(f), np.nan)
    FX = fx[["e", "n", "u", "sigma_m", "conf", "band", "time"]].values
    tops = fx.top_emitter.values
    for t_val, idx in f.groupby("t").indices.items():
        k = idx[0]
        a, b = lo[k], hi[k]
        if b <= a:
            continue
        P = f.loc[idx, ["e", "n", "pos_sigma_m"]].values
        Fk = FX[a:b]
        d = np.hypot(P[:, None, 0] - Fk[None, :, 0], P[:, None, 1] - Fk[None, :, 1])
        sig = np.sqrt(Fk[None, :, 3] ** 2 + P[:, None, 2] ** 2)
        z = d / sig
        ingate = z < 3.0
        counts = ingate.sum(0)                         # tracks per fix
        zz = np.where(ingate, z, np.inf)
        best = np.argmin(zz, axis=1)
        has = np.isfinite(zz[np.arange(len(idx)), best])
        rows = idx[has]; bb = best[has]
        em[rows] = tops[a:b][bb]; conf[rows] = Fk[bb, 4]; band[rows] = Fk[bb, 5]
        dn[rows] = zz[has, bb]; ntr[rows] = counts[bb]; age[rows] = t_val - Fk[bb, 6]
    f["esm_top_emitter"] = em; f["esm_conf"] = conf; f["esm_band"] = band
    f["esm_dist_norm"] = dn; f["esm_n_tracks_in_gate"] = ntr; f["esm_age_s"] = age
    return f


def snapshot_edges(cfg, world, feats: pd.DataFrame, fixes: pd.DataFrame, tier_of: dict):
    """Track-track, track-asset and ESM-track edges at snapshot times."""
    snap = cfg["run"]["snapshot_dt_s"]
    gate2 = cfg["graph"]["edge_gate_km"] * 1000.0
    gate1 = cfg["graph"]["tier1_edge_gate_km"] * 1000.0
    win = int(cfg["graph"]["comove_window_s"] / cfg["run"]["feature_dt_s"])
    A = np.stack([a.pos[:2] for a in world.assets]); vals = np.array([a.value for a in world.assets])
    asset_rmax = cfg["graph"]["asset_edge_range_km"] * 1000.0
    f = feats.sort_values(["eid", "t"])
    # heading-rate history for co-movement correlation
    f["_tr"] = f.turn_rate.fillna(0.0)
    f["_cl"] = f.climb.fillna(0.0)
    hist = {eid: (g.t.values, g._tr.values, g._cl.values) for eid, g in f.groupby("eid")}
    tt, ta, te = [], [], []
    times = np.arange(0.0, cfg["run"]["horizon_s"], snap)
    fx_t = fixes.time.values if len(fixes) else np.array([])
    for t in times:
        s = f[np.isclose(f.t, t)]
        if len(s) < 1:
            continue
        P = s[["e", "n", "u"]].values; V = s[["ve", "vn", "vu"]].values; E = s.eid.values
        tiers = np.array([tier_of[e] for e in E])
        # track-asset
        rel = P[:, None, :2] - A[None]
        r = np.linalg.norm(rel, axis=2)
        clos = -np.einsum("ijk,ik->ij", rel, V[:, :2]) / np.maximum(r, 1.0)
        vv = np.maximum((V[:, :2] ** 2).sum(1), 1e-6)
        tc = np.clip(-np.einsum("ijk,ik->ij", rel, V[:, :2]) / vv[:, None], 0, None)
        cpa = np.linalg.norm(rel + tc[..., None] * V[:, None, :2], axis=2)
        brg = np.rad2deg(np.arctan2(-rel[..., 0], -rel[..., 1])) % 360.0         # track -> asset bearing
        hdg = np.rad2deg(np.arctan2(V[:, 0], V[:, 1])) % 360.0
        for i, j in zip(*np.nonzero(r < asset_rmax)):
            ttg = r[i, j] / clos[i, j] if clos[i, j] > 1.0 else 1e5
            ta.append((t, E[i], j, r[i, j], clos[i, j], min(ttg, 1e5), cpa[i, j], tc[i, j],
                       angle_diff_deg(hdg[i], brg[i, j]), vals[j]))
        # track-track
        d = np.linalg.norm(P[:, None] - P[None], axis=2)
        g = np.where((tiers[:, None] == 1) | (tiers[None] == 1), gate1, gate2)
        dd = np.where((d < g) & ~np.eye(len(E), dtype=bool), d, np.inf)
        kmax = int(cfg["graph"].get("max_neighbors", 8))
        nn = np.argsort(dd, axis=1)[:, :kmax]                       # k nearest within the gate
        ii = np.repeat(np.arange(len(E)), nn.shape[1]); jj = nn.ravel()
        keep = np.isfinite(dd[ii, jj]); ii, jj = ii[keep], jj[keep]
        for i, j in zip(ii, jj):
            hi_, ti_, ci_ = hist[E[i]]; hj, tj, cj = hist[E[j]]
            ki = np.searchsorted(hi_, t); kj = np.searchsorted(hj, t)
            a_, b_ = ti_[max(0, ki - win):ki + 1], tj[max(0, kj - win):kj + 1]
            n = min(len(a_), len(b_))
            corr = np.nan
            if n >= 8 and a_[-n:].std() > 1e-6 and b_[-n:].std() > 1e-6:
                corr = float(np.corrcoef(a_[-n:], b_[-n:])[0, 1])
            si = np.hypot(*V[i, :2]); sj = np.hypot(*V[j, :2])
            tt.append((t, E[i], E[j], *(P[j] - P[i]), *(V[j] - V[i]), d[i, j],
                       abs(angle_diff_deg(hdg[i], hdg[j])), (sj + 0.1) / (si + 0.1), corr))
        # ESM fix -> track candidates (fixes in the last 4 s)
        if len(fx_t):
            a, b = np.searchsorted(fx_t, t - 4.0), np.searchsorted(fx_t, t, "right")
            for fr in fixes.iloc[a:b].itertuples():
                dd = np.hypot(P[:, 0] - fr.e, P[:, 1] - fr.n)
                sig = np.sqrt(fr.sigma_m ** 2 + s.pos_sigma_m.values ** 2)
                for k in np.flatnonzero(dd / sig < 3.0):
                    te.append((t, fr.fix_id, E[k], dd[k] / sig[k]))
    edges_tt = pd.DataFrame(tt, columns=["t", "src_eid", "dst_eid", "de", "dn", "du", "dve", "dvn", "dvu", "dist_m",
                                         "heading_diff_deg", "speed_ratio", "comove_corr"])
    edges_ta = pd.DataFrame(ta, columns=["t", "eid", "asset_id", "range_m", "closure_mps", "ttg_s", "cpa_m",
                                         "tcpa_s", "heading_offset_deg", "asset_value"])
    edges_et = pd.DataFrame(te, columns=["t", "fix_id", "eid", "dist_norm"])
    return edges_tt, edges_ta, edges_et
