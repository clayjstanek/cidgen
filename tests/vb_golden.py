"""Pinned VB results on a real scenario: make the golden file, or compare against it.

The golden file records what the VB tracker produced on one S12 seed, on THIS machine and with THIS scenario
configuration, so any later code change that moves the results is caught (tests/test_vb_regression.py).

    python tests/vb_golden.py                    # write tests/golden/vb_<scenario>_seed<N>_h<H>.json
    python tests/vb_golden.py --check            # compare a fresh run with it, print the differences
    python tests/vb_golden.py --scenario S12_vb --seed 18 --horizon 300

Make the golden file once, on the machine and scenario files you will test with, from code you trust (for
example right after a git commit). Re-make it only when a change is MEANT to move the results, and say so
in the commit. The file stores a fingerprint of the resolved scenario configuration: if the scenario files
change (a new platform class, a new raid), the test stops with "scenario changed" instead of a false failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
DEFAULT = dict(scenario="S12_vbp", seed=20, horizon=120.0)


def golden_path(scenario, seed, horizon):
    return GOLDEN_DIR / f"vb_{scenario}_seed{seed}_h{int(horizon)}.json"


def _fingerprint(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def compute(scenario, seed, horizon):
    """Run world -> radar -> VB tracker -> truth match, exactly as cidgen.generate does, and summarise."""
    from cidgen.config import apply_randomize, load_config
    from cidgen.entities import build_population
    from cidgen.sensors.common import Truth
    from cidgen.sensors.radar import simulate_radars
    from cidgen.trackers import make_tracker
    from cidgen.trackers.integrate import match_tracks, radar_to_enu_all, tracker_summary
    from cidgen.world import build_world

    cfg = apply_randomize(load_config(scenario), seed)
    method = (cfg.get("tracking") or {}).get("method", "label")
    if method not in ("vb", "vbm"):
        raise SystemExit(f"{scenario} uses tracking.method {method!r}; pick a VB scenario (S12_vb, S12_vbp, S12_vbm)")
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(8)]
    world = build_world(cfg, rngs[0])
    ents, _ = build_population(cfg, world, rngs[1], return_tier3=True)
    truth = Truth(ents, cfg["run"]["dt_truth_s"])
    radar = simulate_radars(cfg, world, truth, rngs[2])
    radar = radar[radar.time < horizon]
    grid = np.arange(0.0, horizon, cfg["run"]["feature_dt_s"])
    tr = make_tracker(cfg, method)
    est, rep, tracks = tr.run(radar_to_enu_all(radar, world, cfg), grid)
    m = match_tracks(est, truth, cfg)
    _, su = tracker_summary(m, radar, ents, tracks, cfg)
    s0 = su.iloc[0]
    pos = est[["e", "n", "u"]].to_numpy()
    return {
        "scenario": scenario, "seed": seed, "horizon": horizon, "method": method,
        "config_fingerprint": _fingerprint(cfg),
        "counts": {
            "radar_reports": int(len(radar)),
            "track_rows": int(len(est)),
            "confirmed_tracks": int(len(tracks)),
            "reports_on_confirmed_tracks": int((rep >= 0).sum()),
            **{f"stats_{k}": int(v) for k, v in sorted(tr.stats.items()) if isinstance(v, (int, np.integer))},
        },
        "values": {
            "coverage_mean": float(s0.coverage_mean),
            "tracks_per_object_mean": float(s0.tracks_per_object_mean),
            "false_track_row_frac": float(s0.false_track_row_frac),
            "pos_err_median_m": float(s0.pos_err_median_m),
            "est_pos_mean_e": float(pos[:, 0].mean()), "est_pos_mean_n": float(pos[:, 1].mean()),
            "est_pos_mean_u": float(pos[:, 2].mean()),
            "est_pos_abs_sum": float(np.abs(pos).sum()),
            "est_vel_abs_sum": float(np.abs(est[["ve", "vn", "vu"]].to_numpy()).sum()),
            "est_pos_sigma_sum": float(est.pos_sigma_m.sum()),
        },
    }


def compare(new, old, rtol=1e-6):
    """List of differences (empty = same). Counts must match exactly, values to rtol."""
    diffs = []
    for k in ("scenario", "seed", "horizon", "method"):
        if new[k] != old[k]:
            diffs.append(f"{k}: {new[k]} vs golden {old[k]}")
    for k in sorted(set(new["counts"]) | set(old["counts"])):
        a, b = new["counts"].get(k), old["counts"].get(k)
        if a != b:
            diffs.append(f"count {k}: {a} vs golden {b}")
    for k in sorted(set(new["values"]) | set(old["values"])):
        a, b = new["values"].get(k), old["values"].get(k)
        if a is None or b is None or not np.isclose(a, b, rtol=rtol, atol=1e-9):
            diffs.append(f"value {k}: {a} vs golden {b}")
    return diffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=DEFAULT["scenario"])
    ap.add_argument("--seed", type=int, default=DEFAULT["seed"])
    ap.add_argument("--horizon", type=float, default=DEFAULT["horizon"])
    ap.add_argument("--check", action="store_true", help="compare with the golden file instead of writing it")
    a = ap.parse_args()
    p = golden_path(a.scenario, a.seed, a.horizon)
    new = compute(a.scenario, a.seed, a.horizon)
    if a.check:
        old = json.loads(p.read_text())
        if new["config_fingerprint"] != old["config_fingerprint"]:
            print("scenario changed since the golden file was made (config fingerprint differs)")
        d = compare(new, old)
        print("\n".join(d) if d else f"same as {p.name}")
        sys.exit(1 if d else 0)
    GOLDEN_DIR.mkdir(exist_ok=True)
    p.write_text(json.dumps(new, indent=2) + "\n")
    print(f"wrote {p}")
    print(json.dumps(new["counts"], indent=1)); print(json.dumps(new["values"], indent=1))


if __name__ == "__main__":
    main()
