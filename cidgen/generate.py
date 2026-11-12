"""Run one scenario end to end.

    python -m cidgen.generate --scenario S0 --seed 1 --out runs
    python -m cidgen.generate --scenario S1 --seed 1 --out runs --horizon 900
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .config import apply_randomize, config_hash, load_config
from .entities import build_population
from .features import snapshot_edges, track_features
from .io import write_tables
from .labels import build_labels, entity_table, truth_states
from .sensors.common import Truth
from .sensors.esm import simulate_esm
from .sensors.iff import simulate_iff
from .sensors.nctr import simulate_eoir, simulate_jem, simulate_lidar
from .sensors.radar import simulate_radars
from .tracking import acquisition_tiers, kalman_tracks, radar_to_enu, tracker_settings
from .trackers import make_tracker
from .trackers.integrate import (edge_tier_of, match_tracks, radar_to_enu_all, relabel_cued, relabel_iff,
                                 track_labels, tracker_summary)
from .world import build_world


def run_scenario(scenario: str, seed: int, out_root="runs", overrides=None, fmt="auto", verbose=True, debug=0):
    t_start = time.time()
    cfg = apply_randomize(load_config(scenario, overrides), seed)
    log = (lambda *a: print(f"[{time.time() - t_start:6.1f}s]", *a)) if verbose else (lambda *a: None)
    # independent streams per stage, so changing one sensor model does not reshuffle the rest
    ss = np.random.SeedSequence(seed)
    r_world, r_ent, r_rad, r_iff, r_esm, r_jem, r_lid, r_eo = [np.random.default_rng(s) for s in ss.spawn(8)]

    world = build_world(cfg, r_world)
    ents, tier3_segments = build_population(cfg, world, r_ent, return_tier3=True)
    log(f"{cfg['scenario_name']}: {len(ents)} entities"
        + (f" | drawn: {', '.join(f'{k.split(chr(46))[-1]}={v:.3g}' if isinstance(v, float) else f'{k.split(chr(46))[-1]}={v}' for k, v in cfg['randomized_values'].items())}"
           if cfg.get("randomized_values") else ""))
    truth = Truth(ents, cfg["run"]["dt_truth_s"])
    radar = simulate_radars(cfg, world, truth, r_rad); log(f"radar reports {len(radar):,}")
    iff = simulate_iff(cfg, world, truth, r_iff); log(f"IFF replies {len(iff):,}")
    esm_rep, esm_fix = simulate_esm(cfg, world, truth, r_esm); log(f"ESM reports {len(esm_rep):,}, fixes {len(esm_fix):,}")
    jem = simulate_jem(cfg, world, truth, r_jem)
    lidar = simulate_lidar(cfg, world, truth, r_lid)
    eoir = simulate_eoir(cfg, world, truth, r_eo); log(f"JEM {len(jem)}, LIDAR {len(lidar)}, EO/IR {len(eoir)}")

    tier_of = {e.eid: e.tier for e in ents}
    grid = np.arange(0.0, cfg["run"]["horizon_s"], cfg["run"]["feature_dt_s"])
    method = str((cfg.get("tracking") or {}).get("method", "label"))
    if method not in ("label", "gnn", "jpda", "vb", "vbm", "mfa"):
        raise ValueError(f"tracking.method must be 'label', 'gnn', 'jpda', 'vb', 'vbm' or 'mfa', got {method!r}")
    trk_tables = {}
    radar_out = radar.drop(columns=[])     # merged_eids / n_merged are LABEL columns, see README
    if method == "label":
        # v0.6.1 stand-in: every report carries its true object id; one Kalman filter per id
        renu = radar_to_enu(radar, world)
        # tracker settings per tier; the tier comes from truth (default) or from the first radar reports
        tsrc = str((cfg.get("tracking") or {}).get("tier_source", "truth"))
        if tsrc not in ("truth", "observed"):
            raise ValueError(f"tracking.tier_source must be 'truth' or 'observed', got {tsrc!r}")
        filter_tier = tier_of if tsrc == "truth" else acquisition_tiers(renu, cfg)
        est = kalman_tracks(renu, grid, filter_tier, tracker_settings(cfg)); log(f"track estimates {len(est):,}")
        feats = track_features(cfg, world, est, renu, iff, jem, lidar, eoir, esm_fix); log("features done")
        labels = build_labels(cfg, world, ents, feats, radar, iff, jem, eoir)
        labels = labels.set_index(["eid", "t"]).loc[pd.MultiIndex.from_frame(feats[["eid", "t"]])].reset_index()
        e_tt, e_ta, e_et = snapshot_edges(cfg, world, feats, esm_fix, tier_of); log(f"edges tt {len(e_tt):,}")
        if tsrc == "observed":                             # diagnostic: filter tier chosen at acquisition vs truth
            trk_tables["track_filter_tier"] = pd.DataFrame({"eid": list(filter_tier), "filter_tier": list(filter_tier.values()),
                                                            "true_tier": [tier_of[e] for e in filter_tier]})
    else:
        # real tracker: all radar reports (clutter included), true ids hidden; tables keyed by track number
        fdt = cfg["run"]["feature_dt_s"]
        rall = radar_to_enu_all(radar, world, cfg)
        tlog = None
        if debug == 1:                                   # dated per-block timing log in the output folder
            from .trackers.timing import Timing
            tlog = Timing.stamped_path(out_root, f"{cfg['scenario_name']}_seed{seed}")
        gt = make_tracker(cfg, method, debug, tlog)
        est, rep_tid, tracks = gt.run(rall, grid)
        log(f"{method.upper()}: {len(tracks):,} confirmed tracks ({gt.n_merged} duplicates dropped), "
            f"track estimates {len(est):,}" + (f", {gt.stats}" if hasattr(gt, "stats") else ""))
        radar_out = radar.assign(track_id=rep_tid.values.astype(np.int64))
        renu = rall.assign(eid=rep_tid.values.astype(np.int64))
        renu = renu[renu.eid >= 0].sort_values(["eid", "time"])
        match = match_tracks(est, truth, cfg)
        filter_tier = dict(zip(tracks.eid.astype(int), tracks.filter_tier.astype(int)))
        iff_t = relabel_iff(iff, est, world, cfg, fdt)
        jem_t, lidar_t, eoir_t = (relabel_cued(d, match, fdt) for d in (jem, lidar, eoir))
        feats = track_features(cfg, world, est, renu, iff_t, jem_t, lidar_t, eoir_t, esm_fix); log("features done")
        labels = track_labels(cfg, world, ents, feats, match, radar, iff, jem, eoir, filter_tier)
        e_tt, e_ta, e_et = snapshot_edges(cfg, world, feats, esm_fix, edge_tier_of(match, ents, filter_tier))
        log(f"edges tt {len(e_tt):,}")
        per_obj, summ = tracker_summary(match, radar, ents, tracks, cfg)
        summ["duplicate_tracks_dropped"] = gt.n_merged
        trk_tables = dict(track_truth=match, tracks=tracks, tracker_objects=per_obj, tracker_summary=summ)
        s0 = summ.iloc[0]
        log(f"tracker: coverage {s0.coverage_mean:.3f}, tracks/object {s0.tracks_per_object_mean:.2f}, "
            f"false-track rows {s0.false_track_row_frac:.3f}, swaps {s0.track_swaps}")

    assets = pd.DataFrame([dict(asset_id=a.asset_id, e=a.pos[0], n=a.pos[1], value=a.value, radius_m=a.radius_m)
                           for a in world.assets])
    sites = pd.DataFrame([dict(site_id=s.site_id, kind=s.kind, e=s.pos[0], n=s.pos[1], u=s.pos[2]) for s in world.sites])
    airways = pd.DataFrame([dict(airway_id=a.airway_id, ae=a.a[0], an=a.a[1], be=a.b[0], bn=a.b[1],
                                 alt_lo=a.alt_band_m[0], alt_hi=a.alt_band_m[1]) for a in world.airways])
    corridor = pd.DataFrame(world.corridor, columns=["e", "n"])
    tables = dict(entities=entity_table(ents), truth_states=truth_states(ents), assets=assets, sites=sites,
                  airways=airways, corridor=corridor, reports_radar=radar_out, reports_iff=iff,
                  reports_esm=esm_rep, esm_fixes=esm_fix, reports_jem=jem, reports_lidar=lidar,
                  reports_eoir=eoir, features_track=feats, labels=labels, edges_track_track=e_tt,
                  edges_track_asset=e_ta, edges_esm_track=e_et)
    if len(tier3_segments):                                # hypersonic (Tier 3): LABEL table, never a model input
        tables["hypersonic_segments"] = tier3_segments
    tables.update(trk_tables)
    out_dir = Path(out_root) / f"{cfg['scenario_name']}_seed{seed}"
    meta = dict(scenario=cfg["scenario_name"], seed=seed, generator_version=__version__,
                config_hash=config_hash(cfg), config=cfg, threat_axis_deg=world.threat_axis_deg)
    ext = write_tables(out_dir, tables, meta, fmt)
    (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    log(f"wrote {out_dir} ({ext})")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="S0")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1, help="run this many consecutive seeds")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--horizon", type=float, default=None, help="override run.horizon_s")
    ap.add_argument("--format", default="auto", choices=["auto", "parquet", "csv"])
    ap.add_argument("--debug", type=int, default=0,
                    help="1: time the tracker's update cycles (gnn/jpda/vb/vbm) and write a dated per-block "
                         "timing log plus a summary to the --out folder; anything else: no timing")
    a = ap.parse_args()
    ov = {"run": {"horizon_s": a.horizon}} if a.horizon else None
    for s in range(a.seed, a.seed + a.seeds):
        run_scenario(a.scenario, s, a.out, ov, a.format, debug=a.debug)


if __name__ == "__main__":
    main()
