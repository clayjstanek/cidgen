"""Glue between a multi-target tracker and the rest of the generator.

With a real tracker the feature, label and graph tables are keyed by TRACK number instead of true
object id. This module
  * converts every radar report (clutter included) to ENU for the tracker;
  * matches tracks to true objects at every grid time (labels and scoring only, never a model input);
  * re-attributes the other sensors' reports to tracks:
      IFF        each reply is gated to the nearest track by its measured range and azimuth
      JEM, LIDAR, EO/IR (cued)  each report goes to the track that is following the object the sensor
                 was cued to (the generator's own cue mis-association is kept)
      ESM        unchanged: fixes are already gated to tracks geometrically in features.py
  * builds labels per (track, t) from the matched true object; rows with no match are false-track rows
    (allegiance -1, not an object of interest, so models and TP skip them like birds);
  * summarises tracker quality (coverage, false tracks, breaks, swaps, confirmation delay).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..geometry import enu_to_polar, polar_cov_to_enu, polar_to_enu
from ..labels import build_labels

MATCH_DEFAULTS = {"match_gate_m": 2000.0, "match_gate_sigmas": 4.0, "iff_gate_chi2": 9.21}


def match_settings(cfg):
    m = {**MATCH_DEFAULTS, **((cfg.get("tracking") or {}).get("match") or {})}
    unknown = set(m) - set(MATCH_DEFAULTS)
    if unknown:
        raise KeyError(f"tracking.match: unknown keys {sorted(unknown)}")
    return m


def radar_to_enu_all(radar: pd.DataFrame, world, cfg=None) -> pd.DataFrame:
    """Every radar report in ENU with its noise covariance R; the true id columns are left for scoring only.

    With cfg, also R_trk: R plus the radar's registration-bias covariance from its specification
    (sensors.radar_bias_sigma x bias_scale; the realised bias of each radar is never used). Without it, two
    radars with different biases see one aircraft in two slightly different places, reports fall outside
    the gate and the tracker starts duplicate tracks."""
    site_pos = {s.site_id: s.pos for s in world.sites}
    r = radar.copy()
    sp = np.stack([site_pos[s] for s in r.sensor_id])
    xyz = sp + polar_to_enu(r.range_m.values, r.az_deg.values, r.el_deg.values)
    r["e"], r["n"], r["u"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r["site_e"], r["site_n"], r["site_u"] = sp[:, 0], sp[:, 1], sp[:, 2]
    r["R"] = list(polar_cov_to_enu(r.range_m.values, r.az_deg.values, r.el_deg.values,
                                   r.sigma_r_m.values, r.sigma_az_deg.values, r.sigma_el_deg.values))
    if cfg is not None:
        s = cfg["sensors"]; b = s["radar_bias_sigma"]
        scale = r.sensor_type.map({k: v["bias_scale"] for k, v in s["radar_types"].items()}).to_numpy(float)
        sr = np.hypot(r.sigma_r_m.values, b["range_m"] * scale)
        sa = np.hypot(r.sigma_az_deg.values, b["az_deg"] * scale)
        r["R_trk"] = list(polar_cov_to_enu(r.range_m.values, r.az_deg.values, r.el_deg.values,
                                           sr, sa, r.sigma_el_deg.values))
    return r


def match_tracks(est: pd.DataFrame, truth, cfg) -> pd.DataFrame:
    """One-to-one track-to-object match at every grid time (Hungarian on distance, with a bonus for
    keeping the previous pairing). Gate: max(match_gate_m, match_gate_sigmas * pos_sigma_m).
    Returns est rows' (eid, t) with truth_eid (-1 = no match) and dist_m."""
    m = match_settings(cfg)
    tru_eid = truth.eid
    out_truth = np.full(len(est), -1, np.int64); out_d = np.full(len(est), np.nan)
    prev = {}                                              # track -> true object at the previous grid time
    est = est.reset_index(drop=True)
    for t, idx in est.groupby("t", sort=True).indices.items():
        p, _, alive = truth.at(float(t))
        cand = np.flatnonzero(alive)
        if not len(cand):
            continue
        X = est.loc[idx, ["e", "n", "u"]].to_numpy(); tids = est.eid.to_numpy()[idx]
        D = np.linalg.norm(X[:, None, :] - p[cand][None], axis=2)
        gate = np.maximum(m["match_gate_m"], m["match_gate_sigmas"] * est.pos_sigma_m.to_numpy()[idx])[:, None]
        ok = D <= gate
        if not ok.any():
            prev = {}
            continue
        C = np.where(ok, D / gate, 1e6)
        for a, tid in enumerate(tids):                      # hysteresis: keep a pairing that still gates
            j = prev.get(tid)
            if j is not None:
                k = np.searchsorted(cand, j)
                if k < len(cand) and cand[k] == j and ok[a, k]:
                    C[a, k] -= 0.5
        rows, cols = linear_sum_assignment(C)
        good = ok[rows, cols]
        rows, cols = rows[good], cols[good]
        out_truth[idx[rows]] = tru_eid[cand[cols]]; out_d[idx[rows]] = D[rows, cols]
        prev = {tids[a]: cand[b] for a, b in zip(rows, cols)}
    return pd.DataFrame({"eid": est.eid.values, "t": est.t.values, "truth_eid": out_truth, "dist_m": out_d})


def _est_at(est_t: pd.DataFrame, t: float):
    """Track positions at time t from the latest grid estimate at or before t (constant velocity)."""
    dt = t - est_t.t.to_numpy()
    return est_t[["e", "n", "u"]].to_numpy() + est_t[["ve", "vn", "vu"]].to_numpy() * dt[:, None]


def relabel_iff(iff: pd.DataFrame, est: pd.DataFrame, world, cfg, grid_dt: float) -> pd.DataFrame:
    """Gate each IFF reply to the nearest track by measured range and azimuth (2-D chi-square)."""
    if not len(iff):
        return iff.assign(eid=pd.Series(dtype=np.int64))
    gate = match_settings(cfg)["iff_gate_chi2"]
    site_pos = {s.site_id: s.pos for s in world.sites}
    by_t = {t: g for t, g in est.groupby("t", sort=False)}
    tid = np.full(len(iff), -1, np.int64)
    ii = iff.reset_index(drop=True)
    for (t, sid), idx in ii.groupby(["time", "sensor_id"]).indices.items():
        tg = np.floor(t / grid_dt + 1e-9) * grid_dt
        g = by_t.get(tg)
        if g is None:
            continue
        P = _est_at(g, t)
        r, az, _ = enu_to_polar(P - site_pos[sid][None])
        sig = g.pos_sigma_m.to_numpy()
        zr = ii.range_m.to_numpy()[idx]; za = ii.az_deg.to_numpy()[idx]
        dr = zr[:, None] - r[None]
        daz = (za[:, None] - az[None] + 180.0) % 360.0 - 180.0
        dx = np.deg2rad(daz) * r[None]
        sr2 = 50.0 ** 2 + sig[None] ** 2
        sx2 = (np.deg2rad(0.5) * r[None]) ** 2 + sig[None] ** 2
        d2 = dr ** 2 / sr2 + dx ** 2 / sx2
        best = np.argmin(d2, axis=1)
        okk = d2[np.arange(len(idx)), best] <= gate
        tid[idx[okk]] = g.eid.to_numpy()[best[okk]]
    out = ii.assign(eid=tid)
    return out[out.eid >= 0].reset_index(drop=True)


def relabel_cued(df: pd.DataFrame, match: pd.DataFrame, grid_dt: float) -> pd.DataFrame:
    """A cued report goes to the track following the object the sensor was cued to (its 'eid')."""
    if df is None or not len(df):
        return df
    mm = match[match.truth_eid >= 0]
    key = dict(zip(zip(mm.truth_eid.to_numpy(), np.round(mm.t.to_numpy() / grid_dt).astype(int)), mm.eid.to_numpy()))
    steps = np.floor(df.time.to_numpy() / grid_dt + 1e-9).astype(int)
    tid = np.array([key.get((int(e), int(s)), -1) for e, s in zip(df.eid.to_numpy(), steps)], np.int64)
    out = df.assign(eid=tid)
    return out[out.eid >= 0].reset_index(drop=True)


def track_labels(cfg, world, ents, feats, match, radar, iff, jem, eoir, filter_tier: dict) -> pd.DataFrame:
    """Labels per (track, t): the matched object's labels; unmatched rows are false-track rows."""
    k = feats[["eid", "t"]].merge(match[["eid", "t", "truth_eid"]], on=["eid", "t"], how="left")
    k["truth_eid"] = k.truth_eid.fillna(-1).astype(np.int64)
    hit = k[k.truth_eid >= 0]
    lab = build_labels(cfg, world, ents, pd.DataFrame({"eid": hit.truth_eid.values, "t": hit.t.values}),
                       radar, iff, jem, eoir)
    lab["truth_eid"] = lab.eid.values
    lab["eid"] = hit.eid.values                             # build_labels keeps row order
    miss = k[k.truth_eid < 0]
    fl = pd.DataFrame({"eid": miss.eid.values, "t": miss.t.values, "allegiance": -1, "cls": "FALSE_TRACK",
                       "variant": "", "role": "false_track", "group_id": -1,
                       "tier": [filter_tier.get(e, 2) for e in miss.eid.values],
                       "is_object_of_interest": False, "intent_phase": "none", "threat_priority": 0.0,
                       "multi_aircraft": False, "member_count": 0, "evidence_sufficiency": 0.0,
                       "truth_eid": -1})
    lab = pd.concat([lab, fl[lab.columns]], ignore_index=True)
    lab["is_false_track"] = lab.truth_eid < 0
    return lab.set_index(["eid", "t"]).loc[pd.MultiIndex.from_frame(feats[["eid", "t"]])].reset_index()


def edge_tier_of(match: pd.DataFrame, ents, filter_tier: dict) -> dict:
    """Tier per track for the snapshot-edge gate: the tier of the object it followed most often
    (the label-mode run uses the true tier too); false tracks use their filter tier."""
    tier = {e.eid: e.tier for e in ents}
    mm = match[match.truth_eid >= 0]
    maj = mm.groupby("eid").truth_eid.agg(lambda s: s.value_counts().index[0])
    out = {int(k): tier[int(v)] for k, v in maj.items()}
    for e, ft in filter_tier.items():
        out.setdefault(int(e), int(ft))
    return out


def tracker_summary(match: pd.DataFrame, radar: pd.DataFrame, ents, tracks: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-object table (objects detected by radar at least once) and a one-row run summary."""
    dt = cfg["run"]["feature_dt_s"]
    ooi = {e.eid: e for e in ents}
    det = radar[~radar.false_alarm & (radar.eid >= 0)]
    first_det = det.groupby("eid").time.min()
    last_det = det.groupby("eid").time.max()
    mm = match[match.truth_eid >= 0].sort_values(["truth_eid", "t"])
    rows = []
    for e, t0 in first_det.items():
        g = mm[mm.truth_eid == e]
        span = np.arange(np.ceil(t0 / dt) * dt, last_det[e] + 1e-9, dt)
        covered = np.isin(np.round(span / dt), np.round(g.t.to_numpy() / dt)).mean() if len(span) else np.nan
        ids = g.eid.to_numpy()
        n_tracks = len(np.unique(ids))
        switches = int((ids[1:] != ids[:-1]).sum()) if len(ids) > 1 else 0
        ent = ooi[int(e)]
        rows.append(dict(truth_eid=int(e), tier=ent.tier, allegiance=ent.allegiance, cls=ent.cls,
                         is_ooi=ent.is_ooi, first_detect_s=float(t0),
                         first_track_s=float(g.t.min()) if len(g) else np.nan,
                         confirm_delay_s=float(g.t.min() - t0) if len(g) else np.nan,
                         coverage=float(covered), n_tracks=n_tracks, id_switches=switches,
                         mean_err_m=float(g.dist_m.mean()) if len(g) else np.nan))
    per_obj = pd.DataFrame(rows)
    # swaps: a track whose matched object changes from one object to another
    mt = match.sort_values(["eid", "t"])
    sw = 0
    for _, g in mt[mt.truth_eid >= 0].groupby("eid"):
        v = g.truth_eid.to_numpy(); sw += int((v[1:] != v[:-1]).sum())
    o = per_obj[per_obj.is_ooi] if len(per_obj) else per_obj
    summ = dict(n_objects_detected=int(len(o)), n_tracks_confirmed=int(len(tracks)),
                n_tracks_never_matched=int((~tracks.eid.isin(mm.eid.unique())).sum()),
                track_rows=int(len(match)), false_track_rows=int((match.truth_eid < 0).sum()),
                false_track_row_frac=float((match.truth_eid < 0).mean()) if len(match) else np.nan,
                coverage_mean=float(o.coverage.mean()) if len(o) else np.nan,
                tracks_per_object_mean=float(o.n_tracks.mean()) if len(o) else np.nan,
                id_switches=int(o.id_switches.sum()) if len(o) else 0, track_swaps=int(sw),
                confirm_delay_median_s=float(o.confirm_delay_s.median()) if len(o) else np.nan,
                pos_err_median_m=float(mm.dist_m.median()) if len(mm) else np.nan)
    for t in (1, 2, 3):
        ot = o[o.tier == t] if len(o) else o
        summ[f"coverage_tier{t}"] = float(ot.coverage.mean()) if len(ot) else np.nan
        summ[f"tracks_per_object_tier{t}"] = float(ot.n_tracks.mean()) if len(ot) else np.nan
    return per_obj, pd.DataFrame([summ])
