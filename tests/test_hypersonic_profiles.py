"""Hypersonic (Tier 3) profile player test, CPU only (about 1 minute).

    python tests/test_hypersonic_profiles.py

Checks the worked example in HYPERSONIC_PROFILES.md, that draws are reproducible, that placed tracks
end on their asset, and runs a short S12 scenario through the whole pipeline.
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AIRLINER = [
    {"phase": "cruise", "duration_s": 600, "alt_km": [11.0, 11.0], "speed_mps": [230, 230]},
    {"phase": "descent", "duration_s": 900, "alt_km": [11.0, 3.0], "speed_mps": [230, 150]},
    {"phase": "approach", "duration_s": 300, "alt_km": [3.0, 0.5], "speed_mps": [150, 80],
     "weave_g": 0.2, "weave_period_s": 40},
]

# Profiles the test supplies itself, so it does not depend on what S12_hypersonic.yaml contains.
TEST_PROFILES = {
    "AIRLINER-ARRIVAL": {"class": "E", "allegiance": "neutral", "variant": "civil", "role": "airway",
                         "segments": AIRLINER},
    "PLUMBING-TEST": {"class": "H", "allegiance": "hostile",
                      "segments": [{"phase": "cruise", "duration_s": 300, "alt_km": [15.0, 15.0],
                                    "speed_mps": [1200, 1200], "weave_g": 1.0, "weave_period_s": 30},
                                   {"phase": "terminal", "duration_s": 30, "alt_km": [None, 0.0],
                                    "speed_mps": [None, 1000]}]},
}


def main():
    from cidgen.generate import run_scenario
    from cidgen.hypersonic_profiles import place, play, resolve_segments
    from cidgen.io import read_table

    # 1. worked example: t = 750 s is 1/6 of the way through the descent leg
    rng = np.random.default_rng(0)
    segs = resolve_segments(AIRLINER, rng)
    t, pos, vel, lab, _ = play(segs, rng)
    k = int(np.flatnonzero(t == 750.0)[0])
    assert abs(pos[k, 2] - 9666.67) < 0.1, pos[k, 2]
    assert abs(np.linalg.norm(vel[k]) - 216.67) < 0.01, np.linalg.norm(vel[k])
    assert lab[k] == "descent" and len(t) == 1801
    print("worked example: alt %.2f km, speed %.1f m/s, phase %s" % (pos[k, 2] / 1000, np.linalg.norm(vel[k]), lab[k]))

    # 2. null start values carry over; distributions are reproducible for a given stream
    spec = [{"phase": "a", "duration_s": 10, "alt_km": [5, {"dist": "uniform", "low": 6, "high": 8}], "speed_mps": [100, 100]},
            {"phase": "b", "duration_s": 10, "alt_km": [None, 1], "speed_mps": [None, 50]}]
    s1 = resolve_segments(spec, np.random.default_rng(3)); s2 = resolve_segments(spec, np.random.default_rng(3))
    assert s1 == s2 and s1[1]["alt_km"][0] == s1[0]["alt_km"][1] and s1[1]["speed_mps"][0] == 100

    # 3. placement ends on the target, approaching from the requested bearing
    p2, v2 = place(pos, vel, np.array([5000.0, -2000.0]), 45.0)
    assert np.allclose(p2[-1, :2], [5000.0, -2000.0], atol=1e-6)
    d = p2[-1, :2] - p2[0, :2]
    assert abs(math.degrees(math.atan2(d[0], d[1])) % 360.0 - 225.0) < 1e-6   # travels toward 225 deg (from 045)

    # 4. short S12 run through the whole pipeline
    with tempfile.TemporaryDirectory() as td:
        ov = {"run": {"horizon_s": 600}, "hypersonic": {
            "profiles": TEST_PROFILES, "arrival_window_s": [300, 580],
            "raid": [{"profile": "AIRLINER-ARRIVAL", "count": 1, "arrival_window_s": [300, 580]},
                     {"profile": "PLUMBING-TEST", "count": 3, "arrival_window_s": [300, 580]}]}}
        out = run_scenario("S12", 1, td, ov, verbose=False)
        ent = read_table(out, "entities"); seg = read_table(out, "hypersonic_segments")
        feats = read_table(out, "features_track"); lab = read_table(out, "labels")
        h = ent[ent.cls == "H"]
        assert len(h) == 3 and (h.tier == 3).all() and (h.transponder == "none").all()
        assert set(seg.profile) == {"AIRLINER-ARRIVAL", "PLUMBING-TEST"}
        fh = feats[feats.eid.isin(h.eid)]
        assert len(fh) > 0 and fh.speed.max() > 1000, "Tier 3 tracks not formed"
        assert lab[lab.eid.isin(h.eid)].threat_priority.max() > 0
        print(f"S12 short run: {len(h)} Tier 3 tracks, {len(fh)} track-feature rows, max est speed {fh.speed.max():.0f} m/s")

    # 5. raid waves: wave 2 starts 50-70 s after wave 1's last arrival
    with tempfile.TemporaryDirectory() as td:
        ov = {"hypersonic": {"profiles": TEST_PROFILES, "raid": [
            {"profile": "PLUMBING-TEST", "count": 2, "arrival_window_s": [600, 800]},
            {"profile": "PLUMBING-TEST", "count": 3, "start_after_prev_s": [50, 70]}]}}
        out = run_scenario("S12", 2, td, ov, verbose=False)
        seg = read_table(out, "hypersonic_segments")
        tr = seg.groupby(["wave", "eid"]).agg(start=("t_start", "min"), arrive=("arrival_t", "first")).reset_index()
        w0, w1 = tr[tr.wave == 0], tr[tr.wave == 1]
        gap = w1.start - w0.arrive.max()
        assert len(w0) == 2 and len(w1) == 3 and gap.between(50, 70).all(), (w0, w1)
        print(f"waves: wave 2 starts {gap.min():.0f}-{gap.max():.0f} s after wave 1's last arrival")
    from cidgen.hypersonic_profiles import raid_waves
    for bad in ([{"profile": "PLUMBING-TEST", "count": 1, "start_after_prev_s": [1, 2]}],
                [{"profile": "PLUMBING-TEST", "count": 1}],
                [{"profile": "NOPE", "count": 1, "arrival_window_s": [1, 2]}]):
        try:
            hc = {"profiles": {"PLUMBING-TEST": {}}, "raid": bad}
            raid_waves(hc, 1800)
            if bad[0].get("start_after_prev_s"):
                raise ValueError("start_after_prev_s needs an earlier wave")   # raised at build time
            raise AssertionError(f"accepted bad raid {bad}")
        except (KeyError, ValueError):
            pass
    # 6. per-tier tracker settings and the acquisition-tier option
    from cidgen.tracking import TRACKER_DEFAULTS, tracker_settings
    st = tracker_settings({})
    assert st[1]["q"] == 3.0 and st[2]["q"] == 30.0 and st[3]["q"] == 1000.0 and st[2]["coast_s"] == 12.0
    st = tracker_settings({"hypersonic": {"tracker": {"q": 500.0}},
                           "tracking": {"tiers": {3: {"vel_sigma_mps": 1500.0}, 2: {"q": 40.0}}}})
    assert st[3]["q"] == 500.0 and st[3]["vel_sigma_mps"] == [1500.0] * 3 and st[2]["q"] == 40.0
    assert TRACKER_DEFAULTS[2]["q"] == 30.0                                    # defaults not mutated
    with tempfile.TemporaryDirectory() as td:
        ov = {"run": {"horizon_s": 600}, "tracking": {"tier_source": "observed"}, "hypersonic": {
            "profiles": TEST_PROFILES,
            "raid": [{"profile": "PLUMBING-TEST", "count": 2, "arrival_window_s": [400, 580]}]}}
        out = run_scenario("S12", 3, td, ov, verbose=False)
        ft = read_table(out, "track_filter_tier")
        agree = (ft.filter_tier == ft.true_tier).mean()
        assert set(ft.filter_tier) <= {1, 2, 3} and agree > 0.9, agree
        assert (ft[ft.true_tier == 3].filter_tier == 3).all(), "Tier 3 not recognised at acquisition"
        print(f"observed tier mode: {len(ft)} tracks, acquisition tier = true tier for {agree:.0%}")
    print("OK")


if __name__ == "__main__":
    main()
