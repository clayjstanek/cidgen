"""Per-track state estimation from radar reports (constant-velocity Kalman filter in ENU).

The generator does not solve association: radar reports carry the eid of the strongest
contributor (merged reports update only that track). Features are computed from these
filtered estimates, never from truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geometry import polar_cov_to_enu, polar_to_enu

# ----------------------------------------------------------------------------- per-tier tracker settings
# Kalman filter settings per tier. These defaults reproduce v0.6.1 (tiers 1-2) and the hypersonic
# Tier 3 defaults exactly. Override any of them in a scenario YAML:
#
#   tracking:
#     tier_source: truth            # truth (default) | observed: tier judged from the first radar reports
#     tiers:
#       3: {q: 800.0, vel_sigma_mps: 2000.0, coast_s: 6.0}
#     acquisition: {window_s: 10.0}  # observed mode only; thresholds below
#
# q: white-acceleration PSD (m^2/s^3). vel_sigma_mps: initial velocity sigma, one value or [e, n, u].
# coast_s: drop the track after this long without a radar update.
TRACKER_DEFAULTS = {
    1: {"q": 3.0, "vel_sigma_mps": [15.0, 15.0, 5.0], "coast_s": 6.0},
    2: {"q": 30.0, "vel_sigma_mps": [100.0, 100.0, 30.0], "coast_s": 12.0},
    3: {"q": 1000.0, "vel_sigma_mps": [2000.0, 2000.0, 2000.0], "coast_s": 6.0},
}
ACQUISITION_DEFAULTS = {"window_s": 10.0, "tier3_min_speed_mps": 1000.0,
                        "tier1_max_speed_mps": 60.0, "tier1_max_alt_m": 1500.0, "tier1_max_rcs_dbsm": -5.0}


def tracker_settings(cfg: dict) -> dict:
    """Per-tier settings: defaults, then the older hypersonic.tracker / track_coast_s keys (Tier 3),
    then tracking.tiers. Returns {tier: {"q", "vel_sigma_mps" (3 values), "coast_s"}}."""
    out = {t: {**v, "vel_sigma_mps": list(v["vel_sigma_mps"])} for t, v in TRACKER_DEFAULTS.items()}
    hc = cfg.get("hypersonic") or {}
    legacy = dict(hc.get("tracker") or {})
    if "track_coast_s" in hc:
        legacy["coast_s"] = hc["track_coast_s"]
    layers = [{3: legacy}, {int(k): v for k, v in ((cfg.get("tracking") or {}).get("tiers") or {}).items()}]
    for layer in layers:
        for t, v in layer.items():
            if t not in out:
                raise KeyError(f"tracking.tiers: unknown tier {t} (have {sorted(out)})")
            unknown = set(v) - {"q", "vel_sigma_mps", "coast_s"}
            if unknown:
                raise KeyError(f"tracking.tiers.{t}: unknown keys {sorted(unknown)}")
            for k, x in v.items():
                if k == "vel_sigma_mps":
                    x = [float(x)] * 3 if np.isscalar(x) else [float(y) for y in x]
                    if len(x) != 3:
                        raise ValueError(f"tracking.tiers.{t}.vel_sigma_mps needs 1 or 3 values")
                else:
                    x = float(x)
                out[t][k] = x
    return out


def acquisition_tiers(rep: pd.DataFrame, cfg: dict) -> dict:
    """Tier of each track judged only from its first radar reports (no truth): speed from the
    displacement across the acquisition window, median altitude and median measured RCS, with the
    same rules as models.bn.observable_tier. Returns {eid: tier}."""
    a = {**ACQUISITION_DEFAULTS, **((cfg.get("tracking") or {}).get("acquisition") or {})}
    out = {}
    for eid, g in rep.groupby("eid", sort=False):
        ts = g.time.values
        w = ts <= ts[0] + a["window_s"]
        if w.sum() < 2 and len(ts) >= 2:
            w[:2] = True                                   # too few reports in the window: use the first two
        gw = g[w]
        dt = gw.time.values[-1] - gw.time.values[0]
        if dt > 0:
            d = gw[["e", "n", "u"]].values[-1] - gw[["e", "n", "u"]].values[0]
            sp = float(np.linalg.norm(d) / dt)
        else:
            sp = np.nan                                    # a single report: speed unknown
        alt = float(np.median(gw.u.values))
        rcs = float(np.nanmedian(gw.rcs_dbsm.values)) if gw.rcs_dbsm.notna().any() else 0.0
        if np.isfinite(sp) and sp >= a["tier3_min_speed_mps"]:
            out[eid] = 3
        elif (not np.isfinite(sp) or sp < a["tier1_max_speed_mps"]) and alt < a["tier1_max_alt_m"] \
                and rcs < a["tier1_max_rcs_dbsm"]:
            out[eid] = 1
        else:
            out[eid] = 2
    return out


def radar_to_enu(radar: pd.DataFrame, world) -> pd.DataFrame:
    site_pos = {s.site_id: s.pos for s in world.sites}
    r = radar[~radar.false_alarm & (radar.eid >= 0)].copy()
    sp = np.stack([site_pos[s] for s in r.sensor_id])
    xyz = sp + polar_to_enu(r.range_m.values, r.az_deg.values, r.el_deg.values)
    r["e"], r["n"], r["u"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    R = polar_cov_to_enu(r.range_m.values, r.az_deg.values, r.el_deg.values,
                         r.sigma_r_m.values, r.sigma_az_deg.values, r.sigma_el_deg.values)
    r["R"] = list(R)
    return r.sort_values(["eid", "time"])


def kalman_tracks(rep: pd.DataFrame, grid: np.ndarray, tier_of: dict, settings: dict | None = None) -> pd.DataFrame:
    """Run a CV Kalman filter per eid and sample the estimate on the feature grid.

    tier_of: {eid: tier} used to choose each track's filter settings (true tier, or the tier judged
    at acquisition; see generate.py). settings: tracker_settings(cfg); None = TRACKER_DEFAULTS.
    """
    settings = settings or tracker_settings({})
    out = []
    H = np.hstack([np.eye(3), np.zeros((3, 3))])
    for eid, g in rep.groupby("eid", sort=False):
        tier = tier_of[eid]
        st = settings[tier]
        q = st["q"]                                        # white-acceleration PSD (m^2/s^3)
        ts = g.time.values; zs = g[["e", "n", "u"]].values; Rs = g.R.values
        x = np.r_[zs[0], 0.0, 0.0, 0.0]
        P = np.diag([*np.diag(Rs[0]), *(v ** 2 for v in st["vel_sigma_mps"])])
        t_prev = ts[0]
        gi = np.searchsorted(grid, ts[0])
        k = 1
        n = len(ts)
        last_upd = ts[0]
        while gi < len(grid):
            tg = grid[gi]
            # process all measurements up to the grid time
            while k < n and ts[k] <= tg:
                dt = ts[k] - t_prev
                if dt > 0:
                    F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
                    Q = q * np.block([[np.eye(3) * dt ** 3 / 3, np.eye(3) * dt ** 2 / 2],
                                      [np.eye(3) * dt ** 2 / 2, np.eye(3) * dt]])
                    x = F @ x; P = F @ P @ F.T + Q
                S = H @ P @ H.T + Rs[k]
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ (zs[k] - H @ x)
                P = (np.eye(6) - K @ H) @ P
                t_prev = ts[k]; last_upd = ts[k]; k += 1
            age = tg - last_upd
            if age > st["coast_s"]:
                if k >= n:
                    break
                gi = np.searchsorted(grid, ts[k]); continue
            dt = tg - t_prev
            F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
            xs = F @ x
            Ps = F @ P @ F.T + q * np.block([[np.eye(3) * dt ** 3 / 3, np.eye(3) * dt ** 2 / 2],
                                             [np.eye(3) * dt ** 2 / 2, np.eye(3) * dt]])
            out.append((eid, tg, *xs, float(np.sqrt(np.trace(Ps[:3, :3]))), float(np.sqrt(np.trace(Ps[3:, 3:]))), age))
            gi += 1
    cols = ["eid", "t", "e", "n", "u", "ve", "vn", "vu", "pos_sigma_m", "vel_sigma_mps", "age_s"]
    return pd.DataFrame(out, columns=cols)
