"""Radar models (spec Section 4.1): R-C counter-UAS, R-S 3-D surveillance, R-P long-range sector.

Detection: Swerling I/III Pd = Pfa^(1/(1+SNR)), SNR scaled from a reference range and RCS.
Includes aspect-dependent RCS, registration bias, radar horizon, sector coverage, standoff
jamming, resolution-cell merging (unresolved formations), clutter, micro-Doppler features
(R-C) and a track-capacity limit (R-P).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..geometry import angle_diff_deg, enu_to_polar
from .common import Truth, aspect_deg, aspect_rcs_mult, closure_to_assets, visible

# micro-Doppler feature centres by NCTR family: (blade-flash rate, spectral spread, periodicity)
MD_CENTRES = {"micro": (0.9, 0.7, 0.9), "wingbeat": (0.15, 0.35, 0.45), "rotor": (0.5, 0.6, 0.85),
              "prop": (0.65, 0.4, 0.9), "jem": (0.8, 0.5, 0.9), "none": (0.0, 0.1, 0.2)}


def _snr0(pd_ref, pfa):
    # Swerling I: Pd = Pfa^(1/(1+S))  ->  S = ln(Pfa)/ln(Pd) - 1
    return np.log(pfa) / np.log(pd_ref) - 1.0


def simulate_radars(cfg, world, truth: Truth, rng: np.random.Generator) -> pd.DataFrame:
    s = cfg["sensors"]
    horizon = cfg["run"]["horizon_s"]
    snr0 = _snr0(s["pd_ref"], s["pfa"])
    noise = s["radar_noise_scale"]
    assets_xy = np.stack([a.pos[:2] for a in world.assets])
    jammer_idx = np.array([i for i, e in enumerate(truth.ents) if e.flags.get("jammer")], int)
    ov = cfg["ambiguity"]["micro_doppler_overlap"]
    rows = []
    for site in [x for x in world.sites if x.kind in ("R_C", "R_S", "R_P")]:
        rt = s["radar_types"][site.kind]
        bias_az = rng.normal(0, s["radar_bias_sigma"]["az_deg"] * rt["bias_scale"])
        bias_r = rng.normal(0, s["radar_bias_sigma"]["range_m"] * rt["bias_scale"])
        r_ref = rt["ref_range_km"] * 1000.0
        r_max = r_ref * rt["max_range_factor"]            # instrumented range
        held: dict[int, float] = {}                       # R-P track capacity bookkeeping
        cap = int(s["r_p_track_capacity"]) if site.kind == "R_P" else None
        for t in np.arange(0.0, horizon, rt["update_s"]):
            p, v, alive = truth.at(t)
            ok, rg, az, el, rel = visible(p, alive, site.pos, r_max)
            if rt["sector"] == "threat":
                ok &= np.abs(angle_diff_deg(az, world.threat_axis_deg)) <= s["r_p_sector_halfwidth_deg"]
            ok &= el > -2.0
            idx = np.flatnonzero(ok)
            if len(idx) == 0 and s["clutter_rate"] <= 0:
                continue
            asp = aspect_deg(p[idx], v[idx], site.pos)
            sig_mean = truth.rcs[idx] * aspect_rcs_mult(asp, truth.aspect_tab[idx])
            if truth.rcs_mult is not None:               # hypersonic (Tier 3) profile RCS over time
                sig_mean = sig_mean * truth.rcs_mult[idx, truth.step(t)]
            snr = snr0 * (sig_mean / rt["ref_rcs_m2"]) * (r_ref / np.maximum(rg[idx], 1.0)) ** 4
            # standoff jamming (R-S and R-P only): noise floor up in the jammer's azimuth sector
            jn = np.zeros(len(idx))
            if site.kind != "R_C" and len(jammer_idx):
                step = truth.step(t)
                for j in jammer_idx:
                    if not truth.emitting[j, step]:
                        continue
                    rj, azj, _ = enu_to_polar(p[j] - site.pos)
                    insect = np.abs(angle_diff_deg(az[idx], azj)) < 8.0
                    jn += insect * 30.0 * (100_000.0 / max(rj, 1000.0)) ** 2
            snr_eff = snr / (1.0 + jn)
            pd_ = s["pfa"] ** (1.0 / (1.0 + snr_eff))
            det = rng.random(len(idx)) < pd_ * (1 - s["p_drop"])
            di = idx[det]
            if len(di):
                d_asp, d_jn, d_sig = asp[det], jn[det], sig_mean[det]
                # per-dwell fluctuating RCS
                sw3 = truth.swerling[di] == 3
                fluct = np.where(sw3, rng.gamma(2.0, 0.5, len(di)), rng.exponential(1.0, len(di)))
                sig_draw = d_sig * fluct
                # resolution-cell merging (greedy, in range order)
                order = np.argsort(rg[di])
                clusters, cur = [], []
                for k in order:
                    if cur:
                        k0 = cur[0]
                        if (abs(rg[di[k]] - rg[di[k0]]) < rt["res_r_m"] and
                                abs(angle_diff_deg(az[di[k]], az[di[k0]])) < rt["res_az_deg"] and
                                abs(el[di[k]] - el[di[k0]]) < rt["res_az_deg"] * 1.5):
                            cur.append(k); continue
                        clusters.append(cur)
                    cur = [k]
                if cur:
                    clusters.append(cur)
                # R-P capacity: expire stale holds, admit by policy
                if cap is not None:
                    for e_ in [e for e, tl in held.items() if t - tl > 10.0]:
                        held.pop(e_)
                    cl_lead = [di[c[int(np.argmax(sig_draw[c]))]] for c in clusters]
                    clos = closure_to_assets(p[cl_lead], v[cl_lead], assets_xy)
                    prio = np.maximum(clos, 0) / np.maximum(rg[cl_lead], 1.0)
                    keep = np.ones(len(clusters), bool)
                    new = [m for m, i in enumerate(cl_lead) if truth.eid[i] not in held]
                    room = cap - len([1 for i in cl_lead if truth.eid[i] in held])
                    if len(new) > room:
                        pol = s["capacity_policy"]
                        if pol == "refuse_new" or pol == "drop_oldest":
                            rank = new                      # admit in arrival (range) order
                        else:
                            rank = sorted(new, key=lambda m: -prio[m])
                        for m in rank[max(room, 0):]:
                            keep[m] = False
                    clusters = [c for c, kp in zip(clusters, keep) if kp]
                    for c in clusters:
                        held[int(truth.eid[di[c[int(np.argmax(sig_draw[c]))]]])] = t
                for c in clusters:
                    w = sig_draw[c]
                    lead = di[c[int(np.argmax(w))]]
                    if len(c) > 1:
                        cen = (p[di[c]] * (w / w.sum())[:, None]).sum(0)
                        r_c, az_c, el_c = enu_to_polar(cen - site.pos)
                        vcen = v[di[c]].mean(0)
                        rel_c = cen - site.pos
                    else:
                        r_c, az_c, el_c, vcen, rel_c = rg[lead], az[lead], el[lead], v[lead], rel[lead]
                    jn_c = d_jn[c].max()
                    inflate = (1.0 + jn_c) ** 0.25
                    sr = rt["sigma_r_m"] * noise * inflate
                    sa = rt["sigma_az_deg"] * noise * inflate
                    se = rt["sigma_el_deg"] * noise * inflate
                    sv = rt["sigma_vr_mps"] * noise * inflate
                    vr = float(np.dot(np.nan_to_num(vcen), rel_c) / max(np.linalg.norm(rel_c), 1.0))
                    rcs_rep = 10 * np.log10(max(w.sum(), 1e-6)) + rng.normal(0, s["rcs_cal_error_dB"])
                    md = (np.nan, np.nan, np.nan)
                    if rt["micro_doppler"]:
                        cen_md = np.array(MD_CENTRES.get(truth.nctr[lead], MD_CENTRES["none"]))
                        md = tuple(cen_md * (1 - ov) + 0.5 * ov + rng.normal(0, 0.08, 3))
                    rows.append((t, site.site_id, site.kind, r_c + rng.normal(0, sr) + bias_r,
                                 (az_c + rng.normal(0, sa) + bias_az) % 360.0, el_c + rng.normal(0, se),
                                 vr + rng.normal(0, sv), rcs_rep, sr, sa, se, sv,
                                 10 * np.log10(max(snr_eff[det][c].max(), 1e-9)), bool(jn_c > 1.0),
                                 int(truth.eid[lead]), False, len(c),
                                 ";".join(str(int(truth.eid[di[k]])) for k in c), *md))
            # clutter false alarms
            n_fa = rng.poisson(s["clutter_rate"])
            for _ in range(n_fa):
                a0 = world.threat_axis_deg if rt["sector"] == "threat" else 0.0
                hw = s["r_p_sector_halfwidth_deg"] if rt["sector"] == "threat" else 180.0
                md = (np.nan, np.nan, np.nan)
                if rt["micro_doppler"]:
                    md = tuple(rng.uniform(0, 0.6, 3))
                rows.append((t, site.site_id, site.kind, rng.uniform(0.05, 1.0) * r_ref,
                             (a0 + rng.uniform(-hw, hw)) % 360.0, rng.uniform(0, 3.0), rng.normal(0, 2.0),
                             rng.normal(-20, 5), rt["sigma_r_m"], rt["sigma_az_deg"], rt["sigma_el_deg"],
                             rt["sigma_vr_mps"], rng.uniform(13, 16), False, -1, True, 1, "", *md))
    cols = ["time", "sensor_id", "sensor_type", "range_m", "az_deg", "el_deg", "vr_mps", "rcs_dbsm",
            "sigma_r_m", "sigma_az_deg", "sigma_el_deg", "sigma_vr_mps", "snr_db", "jam_flag",
            "eid", "false_alarm", "n_merged", "merged_eids", "md_rate", "md_spread", "md_period"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "report_id", np.arange(len(df)))
    return df
