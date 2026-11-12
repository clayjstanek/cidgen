"""IFF interrogation (spec Section 4.2): Mode 3/C, Mode 5 Level 1 and Level 2 (with
self-reported position), spoofed and replayed replies, and garble in dense formations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry import angle_diff_deg
from .common import Truth, visible


def simulate_iff(cfg, world, truth: Truth, rng: np.random.Generator) -> pd.DataFrame:
    s, c = cfg["sensors"], cfg["iff"]
    g = cfg["relational"]["iff_garble_coeff"]
    hosts = world.sites_of("R_S") + (world.sites_of("R_P") if c["iff_on_patriot"] else [])
    friend_idx = np.flatnonzero((truth.allegiance == 0) & np.isin(truth.transponder, ["mode5_l1", "mode5_l2"]))
    equipped = truth.transponder != "none"
    rows = []
    for site in hosts:
        rt = s["radar_types"][site.kind]
        r_max = rt["ref_range_km"] * 1000.0 * 1.5
        for t in np.arange(0.0, cfg["run"]["horizon_s"], rt["update_s"]):
            p, v, alive = truth.at(t)
            ok, rg, az, el, _ = visible(p, alive, site.pos, r_max)
            idx = np.flatnonzero(ok & equipped)
            if not len(idx):
                continue
            step = truth.step(t)
            for i in idx:
                if rng.random() > c["p_reply"]:
                    continue
                up = truth.ents[i].iff_up
                if up is not None and not up[step]:
                    continue                                       # transponder outage
                tp = truth.transponder[i]
                others = idx[(idx != i) & (np.abs(angle_diff_deg(az[idx], az[i])) < 2.0) &
                             (np.abs(rg[idx] - rg[i]) < 2000.0)]
                assoc = i
                if len(others) and g > 0:
                    p_loss = 1 - (1 - g) ** len(others)
                    if rng.random() < p_loss:
                        mis = 0.5 * (0.2 if tp == "mode5_l2" else 1.0)
                        if rng.random() < mis:
                            assoc = int(rng.choice(others))       # reply attributed to the wrong aircraft
                        else:
                            continue                               # reply lost
                if rng.random() < cfg["ambiguity"]["p_misassoc_base"] and len(others):
                    assoc = int(rng.choice(others))
                mode3c = tp in ("mode3c", "mode5_l1", "mode5_l2", "spoof_mode3c")
                mode5 = tp in ("mode5_l1", "mode5_l2", "spoof_replay")
                valid = tp in ("mode5_l1", "mode5_l2", "spoof_replay")   # a replay carries a valid code
                level2 = tp in ("mode5_l2", "spoof_replay")
                l2 = (np.nan, np.nan, np.nan)
                if tp == "mode5_l2":
                    l2 = tuple(p[i] + rng.normal(0, c["l2_position_error_m"], 3))
                elif tp == "spoof_replay":
                    # replay of a real friendly's reply recorded 30-120 s earlier: stale, elsewhere
                    if len(friend_idx):
                        f = int(rng.choice(friend_idx))
                        ts = truth.step(max(t - rng.uniform(30, 120), 0))
                        src = truth.P[f, ts] if truth.alive[f, ts] else p[i] + rng.normal(0, 20_000, 3)
                    else:
                        src = p[i] + rng.normal(0, 20_000, 3)
                    l2 = tuple(src + rng.normal(0, c["l2_position_error_m"], 3))
                rows.append((t, site.site_id, int(truth.eid[assoc]), int(truth.eid[i]),
                             rg[i] + rng.normal(0, 50.0), (az[i] + rng.normal(0, 0.5)) % 360.0,
                             mode3c, mode5, valid, level2, *l2))
    cols = ["time", "sensor_id", "eid", "eid_true", "range_m", "az_deg", "mode3c", "mode5", "mode5_valid",
            "level2", "l2_e", "l2_n", "l2_u"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "report_id", np.arange(len(df)))
    return df
