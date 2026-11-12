"""ESM / ELINT (spec Section 4.3): three sites reporting azimuth + elevation angle of arrival
with an emitter-class posterior, and multi-site triangulated fixes including ghosts.

Reports are NOT pre-associated to tracks. Fixes carry `is_ghost` and `eid_true` as labels only.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..geometry import angle_diff_deg, bearing_vec
from ..library import EMITTER_CODES, EMITTERS
from .common import Truth, visible


def _jam_bearings(truth, p, step, site_pos):
    out = []
    for j, e in enumerate(truth.ents):
        if e.flags.get("jammer") and truth.emitting[j, step] and np.isfinite(p[j]).all():
            d = p[j] - site_pos
            out.append(np.rad2deg(np.arctan2(d[0], d[1])) % 360.0)
    return np.array(out)


def _posterior(true_code, p_correct, rng):
    codes = EMITTER_CODES
    band = EMITTERS[true_code][0]
    if rng.random() < p_correct:
        top = true_code
    else:   # confusions favour emitters in the same band
        same = [c for c in codes if EMITTERS[c][0] == band and c != true_code]
        top = rng.choice(same) if same and rng.random() < 0.7 else rng.choice([c for c in codes if c != true_code])
    alpha = np.ones(len(codes))
    alpha[codes.index(top)] += 15.0
    return str(top), rng.dirichlet(alpha)


def _intersect(points, bearings):
    """Least-squares intersection of horizontal bearing lines."""
    A = np.zeros((2, 2)); b = np.zeros(2)
    for p, brg in zip(points, bearings):
        u = bearing_vec(brg)
        M = np.eye(2) - np.outer(u, u)
        A += M; b += M @ p
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    return np.linalg.solve(A, b)


def simulate_esm(cfg, world, truth: Truth, rng: np.random.Generator):
    s = cfg["sensors"]
    e_cfg = s["esm"]
    sig_az, sig_el = cfg["ambiguity"]["esm_aoa_sigma_deg"]
    p_corr = cfg["ambiguity"]["p_ESM_correct"]
    jam_factor = cfg["ambiguity"].get("esm_jam_factor", 1.0)
    jam_hw = cfg["ambiguity"].get("esm_jam_halfwidth_deg", 10.0)
    sites = world.sites_of("ESM")
    emitter_code = np.array([e.emitter_obs or "" for e in truth.ents])
    # each emitter instance has its own RF; drone datalinks share a few narrow channels
    rf_true = np.array([1000.0 * EMITTERS[c][0] + (rng.choice([0.0, 5.0, 10.0]) if c in ("E12", "E13", "E14")
                        else rng.normal(0, 50.0)) if c else np.nan for c in emitter_code])
    sig_f = e_cfg.get("rf_sigma_mhz", 2.0)
    rows = []
    fixes = []
    gate = e_cfg["fix_gate_deg"]
    for t in np.arange(0.0, cfg["run"]["horizon_s"], s["esm_dwell_s"]):
        p, v, alive = truth.at(t)
        step = truth.step(t)
        on = alive & truth.emitting[:, step] & (emitter_code != "")
        per_site = {}
        for site in sites:
            ok, rg, az, el, _ = visible(p, on, site.pos, e_cfg["max_range_km"] * 1000.0)
            idx = np.flatnonzero(ok)
            idx = idx[rng.random(len(idx)) < e_cfg["p_detect"]]
            jb = _jam_bearings(truth, p, step, site.pos)
            recs = []
            for i in idx:
                code = emitter_code[i]
                pc = p_corr
                if len(jb) and code != "EJ" and np.min(np.abs(angle_diff_deg(jb, az[i]))) < jam_hw:
                    pc = p_corr * jam_factor                       # confidently wrong inside the jam strobe
                top, post = _posterior(code, pc, rng)
                band, pri, scan = EMITTERS[code]
                band_meas = band if rng.random() < 0.95 else int(np.clip(band + rng.choice([-1, 1]), 1, 5))
                rid = len(rows)
                rec = (rid, t, site.site_id, (az[i] + rng.normal(0, sig_az)) % 360.0, el[i] + rng.normal(0, sig_el),
                       band_meas, pri, scan, top, float(post.max()), int(truth.eid[i]),
                       rf_true[i] + rng.normal(0, sig_f), *post)
                rows.append(rec)
                recs.append(rec)
            per_site[site.site_id] = (site, recs)
        # triangulation: pair the first two available sites, confirm with the third
        live = [k for k, (_, r) in per_site.items() if r]
        if len(live) < 2:
            continue
        sa, sb = per_site[live[0]], per_site[live[1]]
        sc = per_site[live[2]] if len(live) > 2 else None
        n_combo = 0
        key = lambda r: (r[5], r[6], r[7])        # band, PRI class, scan type: the emitter signature
        f_gate = 3.0 * np.sqrt(2.0) * sig_f
        match = lambda r1, r2: key(r1) == key(r2) and abs(r1[11] - r2[11]) < f_gate
        for ra, rb in itertools.product(sa[1], sb[1]):
            if not match(ra, rb) or n_combo > 400:
                continue
            # elevation consistency: both sites must imply a similar altitude (3-sigma)
            xy0 = _intersect([sa[0].pos[:2], sb[0].pos[:2]], [ra[3], rb[3]])
            if xy0 is None:
                continue
            ra_h = np.linalg.norm(xy0 - sa[0].pos[:2]); rb_h = np.linalg.norm(xy0 - sb[0].pos[:2])
            za = sa[0].pos[2] + ra_h * np.tan(np.deg2rad(ra[4]))
            zb = sb[0].pos[2] + rb_h * np.tan(np.deg2rad(rb[4]))
            if abs(za - zb) > 3 * np.deg2rad(sig_el) * np.hypot(ra_h, rb_h):
                continue
            xy = _intersect([sa[0].pos[:2], sb[0].pos[:2]], [ra[3], rb[3]])
            if xy is None:
                continue
            da = xy - sa[0].pos[:2]; db = xy - sb[0].pos[:2]
            # the fix must lie in front of both sites, not behind them
            if np.dot(da, bearing_vec(ra[3])) <= 0 or np.dot(db, bearing_vec(rb[3])) <= 0:
                continue
            members = [ra, rb]; sites_used = [sa[0], sb[0]]
            if sc is not None:
                brg_c = np.rad2deg(np.arctan2(*(xy - sc[0].pos[:2]))) % 360.0
                same = [r for r in sc[1] if match(r, ra)]
                cands = [r for r in same if abs(angle_diff_deg(r[3], brg_c)) < gate]
                if cands:
                    rc = min(cands, key=lambda r: abs(angle_diff_deg(r[3], brg_c)))
                    members.append(rc); sites_used.append(sc[0])
                    xy = _intersect([x.pos[:2] for x in sites_used], [m[3] for m in members])
                elif same:
                    continue    # third site heard this signature but not on this bearing: reject
            n_combo += 1
            ranges = [np.linalg.norm(xy - x.pos[:2]) for x in sites_used]
            ua = bearing_vec(members[0][3]); ub = bearing_vec(members[1][3])
            cross = max(abs(ua[0] * ub[1] - ua[1] * ub[0]), 0.05)
            sigma_m = float(np.mean(ranges) * np.deg2rad(sig_az) / cross)
            if sigma_m > e_cfg.get("max_fix_sigma_km", 15.0) * 1000.0:
                continue     # near-parallel bearings: geometry too poor for a usable fix
            alt = float(np.mean([x.pos[2] + rr * np.tan(np.deg2rad(m[4])) for x, rr, m in zip(sites_used, ranges, members)]))
            true_ids = {m[10] for m in members}
            fixes.append((t, xy[0], xy[1], alt, sigma_m, len(members), ra[5], ra[8], float(np.mean([m[9] for m in members])),
                          ";".join(str(m[0]) for m in members), len(true_ids) > 1,
                          members[0][10] if len(true_ids) == 1 else -1))
    rcols = ["report_id", "time", "sensor_id", "az_deg", "el_deg", "band", "pri_class", "scan_type", "top_emitter",
             "top_conf", "eid_true", "rf_mhz"] + [f"p_{c}" for c in EMITTER_CODES]
    fcols = ["time", "e", "n", "u", "sigma_m", "n_sites", "band", "top_emitter", "conf", "report_ids", "is_ghost", "eid_true"]
    rep = pd.DataFrame(rows, columns=rcols)
    fx = pd.DataFrame(fixes, columns=fcols)
    fx.insert(0, "fix_id", np.arange(len(fx)))
    return rep, fx
