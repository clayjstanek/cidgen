"""Run only the tracker on one scenario seed: world, radar reports, tracker, track quality. No features,
labels or graph, so it is the quick loop for timing work.

    python -m cidgen.trackers.run_tracker --scenario S12_vb --seed 20 --debug 1
    python -m cidgen.trackers.run_tracker --scenario S12_vb --seed 20 --horizon 300 --debug 1 --out timing

--debug 1 times the tracker's update cycles and blocks: a summary table is printed, and every timed block
writes one line (wall time, simulation time, section, milliseconds) to
    <out>/<scenario>_seed<N>_timingLog_<YYYYMMDD_HHMMSS>.csv
with the summary beside it (..._summary.txt and ..._summary.csv). Any other --debug value: no timing.
--method overrides the scenario's tracking.method (gnn, jpda, vb, vbm).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..config import apply_randomize, load_config
from ..entities import build_population
from ..sensors.common import Truth
from ..sensors.radar import simulate_radars
from ..world import build_world
from . import make_tracker
from .integrate import match_tracks, radar_to_enu_all, tracker_summary
from .timing import Timing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--method", default=None, help="gnn | jpda | vb | vbm (default: the scenario's)")
    ap.add_argument("--horizon", type=float, default=None, help="stop the radar reports at this time (s)")
    ap.add_argument("--debug", type=int, default=0, help="1: time the tracker and write a dated timing log")
    ap.add_argument("--out", default="timing", help="folder for the timing log and summary")
    a = ap.parse_args()
    ov = {"tracking": {"method": a.method}} if a.method else None
    cfg = apply_randomize(load_config(a.scenario, ov), a.seed)
    method = (cfg.get("tracking") or {}).get("method", "label")
    if method == "label":
        raise SystemExit("this scenario uses association by label; pick a tracker scenario or --method")
    # same random streams as cidgen.generate, so the reports match a full run of this seed
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(a.seed).spawn(8)]
    world = build_world(cfg, rngs[0])
    ents, _ = build_population(cfg, world, rngs[1], return_tier3=True)
    truth = Truth(ents, cfg["run"]["dt_truth_s"])
    radar = simulate_radars(cfg, world, truth, rngs[2])
    horizon = a.horizon or cfg["run"]["horizon_s"]
    radar = radar[radar.time < horizon]
    grid = np.arange(0.0, horizon, cfg["run"]["feature_dt_s"])
    rall = radar_to_enu_all(radar, world, cfg)
    tlog = Timing.stamped_path(a.out, f"{cfg['scenario_name']}_seed{a.seed}") if a.debug == 1 else None
    tr = make_tracker(cfg, method, a.debug, tlog)
    t0 = time.perf_counter()
    est, _, tracks = tr.run(rall, grid)
    print(f"{method}: {len(radar):,} reports, {len(tracks):,} confirmed tracks, {time.perf_counter() - t0:.1f} s")
    if getattr(tr, "stats", None):
        print("tracker stats:", tr.stats)
    m = match_tracks(est, truth, cfg)
    _, su = tracker_summary(m, radar, ents, tracks, cfg)
    s0 = su.iloc[0]
    print(f"coverage {s0.coverage_mean:.3f}  tracks/object {s0.tracks_per_object_mean:.2f}  "
          f"false-track rows {s0.false_track_row_frac:.3f}")


if __name__ == "__main__":
    main()
