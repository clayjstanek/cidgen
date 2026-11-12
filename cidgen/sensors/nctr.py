"""NCTR-type sensors (spec Sections 4.4-4.6): JEM / rotor modulation and LIDAR hosted on the
Sentinel-class radar, and one EO/IR camera cued by the counter-UAS radar."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..library import ENGINE_CODES, EOIR_CLASSES, PLATFORMS
from .common import Cuer, Truth, aspect_deg, closure_to_assets, misassociate, visible

# notional blade-passage feature centre per engine / rotor code (arbitrary units)
BPF = {c: 1.0 + 0.35 * k for k, c in enumerate(ENGINE_CODES)}


def _engine_posterior(code, rng, p_correct=0.85):
    codes = ENGINE_CODES
    top = code if rng.random() < p_correct else rng.choice([c for c in codes if c != code])
    alpha = np.ones(len(codes)); alpha[codes.index(top)] += 12.0
    return str(top), rng.dirichlet(alpha)


def simulate_jem(cfg, world, truth: Truth, rng):
    s = cfg["sensors"]; j = s["jem"]
    win = s["jem_aspect_window_deg"]
    eng = np.array([e.engine_obs or "" for e in truth.ents])
    ok_family = np.isin(truth.nctr, ["jem", "rotor", "prop"])
    assets_xy = np.stack([a.pos[:2] for a in world.assets])
    rows = []
    for site in world.sites_of("R_S"):
        cuer = Cuer(s["cueing_policy"] if s["cueing_policy"] != "none" else "random", truth.E, rng)
        for t in np.arange(0.0, cfg["run"]["horizon_s"], j["period_s"]):
            p, v, alive = truth.at(t)
            ok, rg, az, el, _ = visible(p, alive & ok_family & (eng != ""), site.pos, j["max_range_km"] * 1000.0)
            asp = aspect_deg(p, np.nan_to_num(v), site.pos)
            ok &= (asp < win) | (asp > 180 - win)
            cand = np.flatnonzero(ok)
            sel = cuer.pick(cand, closure_to_assets(p, v, assets_xy), t, k=3)
            sel = sel[rng.random(len(sel)) < j["p_available"]]
            if not len(sel):
                continue
            assoc = misassociate(sel, p, alive, truth.tier, rng, cfg["ambiguity"]["p_misassoc_base"])
            for i, a in zip(sel, assoc):
                top, post = _engine_posterior(eng[i], rng)
                rows.append((t, site.site_id, int(truth.eid[a]), int(truth.eid[i]), float(asp[i]), top,
                             float(post.max()), BPF[eng[i]] + rng.normal(0, 0.12), *post))
    cols = ["time", "sensor_id", "eid", "eid_true", "aspect_deg", "top_engine", "top_conf", "bpf_feature"] + \
           [f"p_{c}" for c in ENGINE_CODES]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "report_id", np.arange(len(df)))
    return df


def simulate_lidar(cfg, world, truth: Truth, rng):
    s = cfg["sensors"]; L = s["lidar"]
    rs = world.sites_of("R_S")
    if not rs or s["cueing_policy"] == "none":
        return pd.DataFrame(columns=["report_id", "time", "sensor_id", "eid", "eid_true", "aspect_deg",
                                     "len_fuselage_m", "len_span_m", "len_tail_m"])
    site = rs[0]
    dims = np.array([e.dims for e in truth.ents])
    assets_xy = np.stack([a.pos[:2] for a in world.assets])
    cuer = Cuer(s["cueing_policy"], truth.E, rng)
    rows = []
    for t in np.arange(0.0, cfg["run"]["horizon_s"], L["period_s"]):
        p, v, alive = truth.at(t)
        ok, rg, *_ = visible(p, alive & (truth.cls != "K"), site.pos, L["max_range_km"] * 1000.0)
        sel = cuer.pick(np.flatnonzero(ok), closure_to_assets(p, v, assets_xy), t, k=1)
        if not len(sel):
            continue
        assoc = misassociate(sel, p, alive, truth.tier, rng, cfg["ambiguity"]["p_misassoc_base"])
        for i, a in zip(sel, assoc):
            asp = float(aspect_deg(p[i:i + 1], np.nan_to_num(v[i:i + 1]), site.pos)[0])
            sa, ca = abs(np.sin(np.deg2rad(asp))), abs(np.cos(np.deg2rad(asp)))
            fore = np.array([np.sqrt(sa ** 2 + 0.15 * ca ** 2), np.sqrt(ca ** 2 + 0.3 * sa ** 2), np.sqrt(ca ** 2 + 0.3 * sa ** 2)])
            meas = dims[i] * fore * (1 + rng.normal(0, L["noise_frac"], 3))
            meas[rng.random(3) < L["p_segment_drop"]] = np.nan
            rows.append((t, site.site_id, int(truth.eid[a]), int(truth.eid[i]), asp, *meas))
    df = pd.DataFrame(rows, columns=["time", "sensor_id", "eid", "eid_true", "aspect_deg",
                                     "len_fuselage_m", "len_span_m", "len_tail_m"])
    df.insert(0, "report_id", np.arange(len(df)))
    return df


def simulate_eoir(cfg, world, truth: Truth, rng):
    s = cfg["sensors"]; E = s["eoir"]
    vis = cfg["world"]["visibility"]
    cams = world.sites_of("EOIR")
    cols = ["report_id", "time", "sensor_id", "eid", "eid_true", "range_m", "top_class", "top_conf",
            "friend_variant_recognized", "ir_intensity"] + [f"p_{c}" for c in EOIR_CLASSES]
    if not cams or s["cueing_policy"] == "none" or vis <= 0:
        return pd.DataFrame(columns=cols)
    site = cams[0]
    rmax = E["max_range_km"] * 1000.0 * vis
    assets_xy = np.stack([a.pos[:2] for a in world.assets])
    cuer = Cuer(s["cueing_policy"], truth.E, rng)
    variants = np.array([e.variant for e in truth.ents])
    rows = []
    for t in np.arange(0.0, cfg["run"]["horizon_s"], E["period_s"]):
        p, v, alive = truth.at(t)
        ok, rg, *_ = visible(p, alive & (truth.tier == 1), site.pos, rmax)
        sel = cuer.pick(np.flatnonzero(ok), closure_to_assets(p, v, assets_xy), t, k=1)
        if not len(sel):
            continue
        assoc = misassociate(sel, p, alive, truth.tier, rng, cfg["ambiguity"]["p_misassoc_base"])
        for i, a in zip(sel, assoc):
            c = truth.cls[i] if truth.cls[i] in EOIR_CLASSES else "other"
            conc = 1.0 + 25.0 * max(1 - rg[i] / rmax, 0.0)
            alpha = np.ones(len(EOIR_CLASSES)); alpha[EOIR_CLASSES.index(c)] += conc
            post = rng.dirichlet(alpha)
            fv = bool(variants[i] == "blue" and rng.random() < E["p_friend_variant_recog"])
            ir = {"U1": 0.3, "U2": 0.4, "K": 0.5}.get(truth.cls[i], 1.0) * rng.lognormal(0, 0.2)
            rows.append((t, site.site_id, int(truth.eid[a]), int(truth.eid[i]), float(rg[i]),
                         EOIR_CLASSES[int(np.argmax(post))], float(post.max()), fv, ir, *post))
    df = pd.DataFrame(rows, columns=cols[1:])
    df.insert(0, "report_id", np.arange(len(df)))
    return df
