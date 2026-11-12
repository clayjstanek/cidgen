"""Hypersonic (Tier 3) kinematic profile player -- segment playback for cidgen.

Plays back a track SHAPE that the user describes leg by leg. There is no aerodynamics, propulsion,
energy model or guidance law here: altitude and total speed change linearly across each segment, an
optional weave flips the turn direction at random intervals, and the finished path is rotated and
placed so that it ends on a defended asset at the chosen arrival time. Every number comes from the
scenario YAML; the module itself contains no vehicle values. See HYPERSONIC_PROFILES.md.

Scenario YAML (configs/scenarios/S12_hypersonic.yaml is a runnable example):

    hypersonic:
      profiles:
        MY-PROFILE:
          class: H                     # library.PLATFORMS code (default H, the Tier 3 placeholder)
          allegiance: hostile          # friend | neutral | hostile (default hostile)
          variant: red                 # library variant key (default red); blue = friendly, civil = neutral
          transponder: none            # optional: none | mode3c | mode5_l1 | mode5_l2 (Tier 3 default: none)
          direction: inbound           # optional: inbound (ends on an asset, default) | outbound (starts on an
                                       #   asset and flies the profile away from it, toward the ingress bearing)
          rcs_dbsm: {dist: normal, mean: <>, std: <>}   # optional; else the class median
          rcs_mod_db: [[t_s, dB], ...] # optional; added to RCS over profile time (interpolated)
          segments:
            - {phase: <label>, duration_s: <>, alt_km: [<start>, <end>], speed_mps: [<start>, <end>],
               weave_g: <>, weave_period_s: <>}       # weave optional
      raid: {MY-PROFILE: 2}            # how many of each profile per run
      arrival_window_s: [600, 1700]    # arrival time on the asset, absolute run time
      ingress_halfwidth_deg: 30        # approach bearing: threat axis +/- this

Waves: `raid` may instead be a list, played in order. Each wave names one profile and a count, and
times its tracks with exactly one of
      arrival_window_s: [lo, hi]       # absolute arrival time on the asset (as above)
      start_after_prev_s: [lo, hi]     # profile START time = previous wave's last arrival + U(lo, hi)
and may set its own ingress_halfwidth_deg (default: the block-level value):

      raid:
        - {profile: MY-PROFILE, count: 4, arrival_window_s: [600, 900]}
        - {profile: OTHER,      count: 6, start_after_prev_s: [50, 70]}

Any number may be a distribution: {dist: uniform, low, high} | {dist: normal, mean, std} |
{dist: truncnormal, mean, std, low, high} | {dist: lognormal, median, sigma_ln} |
{dist: triangular, low, mode, high} | {dist: choice, options, weights}. A segment's start value may
be null: it carries over the previous segment's end value.

Outbound profiles (direction: outbound) are the same path played backwards in time: the track is at the
asset at the drawn time and then flies out along the ingress bearing, with the segments in reverse order.
For outbound waves, arrival_window_s and the time used by a following start_after_prev_s wave are the time
the track is AT the asset (its start), not the end of the track.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import FRIEND, HOSTILE, NEUTRAL
from .geometry import bearing_vec
from .kinematics import PHASES
from .library import PLATFORMS

G0 = 9.80665
ALLEGIANCE_CODES = {"friend": FRIEND, "neutral": NEUTRAL, "hostile": HOSTILE}
TRANSPONDERS = ("none", "mode3c", "mode5_l1", "mode5_l2")
DIRECTIONS = ("inbound", "outbound")
_FIRST_DYNAMIC_PHASE = 20          # user phase labels not already in kinematics.PHASES start here


# ----------------------------------------------------------------------------- distributions
def draw(v, rng: np.random.Generator):
    """A number, None, or a distribution spec -> a concrete value."""
    if v is None or isinstance(v, (int, float)):
        return v
    if not isinstance(v, dict) or "dist" not in v:
        raise ValueError(f"expected a number, null or a {{dist: ...}} spec, got {v!r}")
    k = v["dist"]
    if k == "fixed":
        return float(v["value"])
    if k == "uniform":
        return float(rng.uniform(v["low"], v["high"]))
    if k == "normal":
        return float(rng.normal(v["mean"], v["std"]))
    if k == "truncnormal":
        for _ in range(200):
            x = rng.normal(v["mean"], v["std"])
            if v["low"] <= x <= v["high"]:
                return float(x)
        return float(np.clip(v["mean"], v["low"], v["high"]))
    if k == "lognormal":
        return float(rng.lognormal(math.log(v["median"]), v["sigma_ln"]))
    if k == "triangular":
        return float(rng.triangular(v["low"], v["mode"], v["high"]))
    if k == "choice":
        w = v.get("weights")
        p = None if w is None else np.asarray(w, float) / np.sum(w)
        return v["options"][int(rng.choice(len(v["options"]), p=p))]
    raise ValueError(f"unknown distribution '{k}'")


# ----------------------------------------------------------------------------- phases
def phase_code(label: str) -> int:
    """Integer code for a phase label; labels not in kinematics.PHASES are registered on first use
    (in profile order, so codes are deterministic for a given scenario file)."""
    if label not in PHASES:
        PHASES[label] = max(max(PHASES.values()) + 1, _FIRST_DYNAMIC_PHASE)
        from .entities import PHASE_NAMES      # labels.py reads names from here
        PHASE_NAMES[PHASES[label]] = label
    return PHASES[label]


# ----------------------------------------------------------------------------- one profile
def resolve_segments(segments: list, rng) -> list[dict]:
    """Draw every distribution and fill null start values from the previous segment's end."""
    out, prev = [], None
    for i, s in enumerate(segments):
        if "phase" not in s or "duration_s" not in s:
            raise ValueError(f"segment {i}: 'phase' and 'duration_s' are required")
        a0, a1 = (draw(x, rng) for x in s.get("alt_km", [None, None]))
        v0, v1 = (draw(x, rng) for x in s.get("speed_mps", [None, None]))
        if a0 is None:
            a0 = prev["alt_km"][1] if prev else None
        if v0 is None:
            v0 = prev["speed_mps"][1] if prev else None
        a1 = a0 if a1 is None else a1
        v1 = v0 if v1 is None else v1
        if None in (a0, v0):
            raise ValueError(f"segment {i}: the first segment needs start alt_km and speed_mps")
        r = dict(phase=str(s["phase"]), duration_s=float(draw(s["duration_s"], rng)),
                 alt_km=(float(a0), float(a1)), speed_mps=(float(v0), float(v1)),
                 weave_g=float(draw(s.get("weave_g", 0.0), rng) or 0.0),
                 weave_period_s=float(draw(s.get("weave_period_s", 60.0), rng)))
        if r["duration_s"] <= 0:
            raise ValueError(f"segment {i}: duration_s must be positive")
        out.append(r)
        prev = r
    return out


def play(segs: list[dict], rng, dt: float = 1.0):
    """Play resolved segments in a local frame starting at the origin heading north.
    Returns t (N,), pos (N,3), vel (N,3), phase labels (N,) and segment index (N,)."""
    total = sum(s["duration_s"] for s in segs)
    n = int(math.floor(total / dt)) + 1
    t = np.arange(n) * dt
    pos = np.zeros((n, 3)); vel = np.zeros((n, 3))
    labels = np.empty(n, object); seg_i = np.zeros(n, int)
    bounds = np.cumsum([0.0] + [s["duration_s"] for s in segs])
    hdg = 0.0
    sign = float(rng.choice([-1.0, 1.0]))
    next_flip = None
    x = y = 0.0
    for k in range(n):
        j = min(int(np.searchsorted(bounds, t[k], "right") - 1), len(segs) - 1)
        s = segs[j]
        f = min((t[k] - bounds[j]) / s["duration_s"], 1.0)
        alt = (s["alt_km"][0] + (s["alt_km"][1] - s["alt_km"][0]) * f) * 1000.0
        V = s["speed_mps"][0] + (s["speed_mps"][1] - s["speed_mps"][0]) * f
        vz = (s["alt_km"][1] - s["alt_km"][0]) * 1000.0 / s["duration_s"]
        vh = math.sqrt(max(V * V - vz * vz, 0.0))
        if s["weave_g"] > 0 and vh > 1.0:
            if next_flip is None or next_flip < bounds[j]:
                next_flip = bounds[j] + s["weave_period_s"] * rng.uniform(0.5, 1.5)
            if t[k] >= next_flip:
                sign = -sign
                next_flip = t[k] + s["weave_period_s"] * rng.uniform(0.5, 1.5)
            hdg += sign * s["weave_g"] * G0 / vh * dt
        if k:
            x += vh * math.sin(hdg) * dt
            y += vh * math.cos(hdg) * dt
        pos[k] = (x, y, max(alt, 0.0))
        vel[k] = (vh * math.sin(hdg), vh * math.cos(hdg), vz)
        labels[k] = s["phase"]; seg_i[k] = j
    return t, pos, vel, labels, seg_i


def place(pos, vel, target_xy, ingress_brg_deg):
    """Rotate and translate a local path so it ends on target_xy, arriving from ingress_brg_deg
    (the start point lies on that bearing from the target, at whatever distance the path covers)."""
    end = pos[-1, :2]
    d = np.linalg.norm(end)
    if d < 1.0:
        return pos + np.r_[target_xy, 0.0], vel
    want = -bearing_vec(ingress_brg_deg)                 # direction of travel, start -> target
    ang = math.atan2(want[0], want[1]) - math.atan2(end[0], end[1])
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, s], [-s, c]])                      # rotates compass bearings by +ang
    p2 = pos[:, :2] @ R.T; v2 = vel[:, :2] @ R.T
    p2 = p2 - p2[-1] + target_xy
    return np.c_[p2, pos[:, 2]], np.c_[v2, vel[:, 2]]


def rcs_multiplier(profile: dict, cls: str, t_rel, rng):
    """Linear multiplier on the class median RCS over profile time."""
    base_db = 10.0 * math.log10(PLATFORMS[cls].rcs_median_m2)
    level = draw(profile.get("rcs_dbsm"), rng)
    off = (level - base_db) if level is not None else 0.0
    mod = profile.get("rcs_mod_db")
    if mod:
        m = np.asarray(mod, float)
        return 10.0 ** ((off + np.interp(t_rel, m[:, 0], m[:, 1])) / 10.0)
    return np.full(len(t_rel), 10.0 ** (off / 10.0))


# ----------------------------------------------------------------------------- raid waves
def raid_waves(hc: dict, T: int) -> list[dict]:
    """Normalise hypersonic.raid to a list of waves. The original mapping form {PROFILE: count} becomes
    one wave per profile, all sharing the block-level arrival_window_s."""
    raid = hc.get("raid") or {}
    profiles = hc.get("profiles") or {}
    win = list(hc.get("arrival_window_s", [0.5 * T, T - 1]))
    hw = float(hc.get("ingress_halfwidth_deg", 30.0))
    if isinstance(raid, dict):
        waves = [{"profile": k, "count": v, "arrival_window_s": win} for k, v in raid.items()]
    elif isinstance(raid, list):
        waves = [dict(w) for w in raid]
    else:
        raise ValueError("hypersonic.raid must be a mapping {PROFILE: count} or a list of waves")
    out = []
    for i, w in enumerate(waves):
        unknown = set(w) - {"profile", "count", "arrival_window_s", "start_after_prev_s", "ingress_halfwidth_deg"}
        if unknown:
            raise KeyError(f"raid wave {i}: unknown keys {sorted(unknown)}")
        if w.get("profile") not in profiles:
            raise KeyError(f"raid wave {i}: profile {w.get('profile')!r} is not defined in hypersonic.profiles")
        if isinstance(raid, list):
            timing = [k for k in ("arrival_window_s", "start_after_prev_s") if k in w]
            if len(timing) != 1:
                raise ValueError(f"raid wave {i}: give exactly one of arrival_window_s or start_after_prev_s")
        for k in ("arrival_window_s", "start_after_prev_s"):
            if k in w:
                lo, hi = (float(x) for x in w[k])
                if hi < lo:
                    raise ValueError(f"raid wave {i}: {k} high < low")
                w[k] = (lo, hi)
        w["count"] = int(w.get("count", 1))
        w["ingress_halfwidth_deg"] = float(w.get("ingress_halfwidth_deg", hw))
        out.append(w)
    return out


# ----------------------------------------------------------------------------- population hook
def add_profile_tracks(pop) -> pd.DataFrame:
    """Add every raid entry of cfg['hypersonic'] to the Population. Called from Population.build()
    after all other entities, emitter timelines and IFF outages, on its own random stream, so the
    rest of the scenario's truth is unchanged by adding profiles. Returns the per-segment table."""
    hc = pop.cfg.get("hypersonic") or {}
    T = pop.T
    waves = raid_waves(hc, T)
    if not waves:
        return pd.DataFrame()
    profiles = hc.get("profiles") or {}
    rng = pop.rng.spawn(1)[0]
    saved, pop.rng = pop.rng, rng                       # _add() draws signatures from pop.rng
    rows = []
    prev_last_arrival = None
    try:
        for wi, w in enumerate(waves):                  # YAML order: deterministic phase codes
            name = w["profile"]
            prof = profiles[name]
            hw = w["ingress_halfwidth_deg"]
            if "start_after_prev_s" in w and prev_last_arrival is None:
                raise ValueError(f"raid wave {wi}: start_after_prev_s needs an earlier wave")
            arrivals = []
            cls = str(prof.get("class", "H"))
            if cls not in PLATFORMS:
                raise KeyError(f"profile {name}: class '{cls}' is not in library.PLATFORMS")
            alg = ALLEGIANCE_CODES[str(prof.get("allegiance", "hostile")).lower()]
            variant = str(prof.get("variant", "red"))
            role = str(prof.get("role", name))
            direction = str(prof.get("direction", "inbound")).lower()
            if direction not in DIRECTIONS:
                raise ValueError(f"profile {name}: direction must be one of {DIRECTIONS}, got {direction!r}")
            xpdr = prof.get("transponder")
            if xpdr is not None and str(xpdr) not in TRANSPONDERS:
                raise ValueError(f"profile {name}: transponder must be one of {TRANSPONDERS}, got {xpdr!r}")
            for _ in range(w["count"]):
                segs = resolve_segments(prof["segments"], rng)
                t_rel, pos, vel, labels, seg_i = play(segs, rng)
                ti = int(rng.integers(len(pop.world.assets)))
                tgt = pop.world.assets[ti].pos[:2]
                brg = pop.world.threat_axis_deg + rng.uniform(-hw, hw)
                pos, vel = place(pos, vel, tgt, brg)
                outbound = direction == "outbound"
                if outbound:                            # same path backwards: starts on the asset, flies out
                    pos, vel = pos[::-1].copy(), -vel[::-1].copy()
                    labels = list(labels)[::-1]
                if "start_after_prev_s" in w:
                    t0 = int(round(prev_last_arrival + rng.uniform(*w["start_after_prev_s"])))
                    t_arr = t0 if outbound else t0 + int(round(t_rel[-1]))
                else:
                    t_arr = int(round(rng.uniform(*w["arrival_window_s"])))
                    t0 = t_arr if outbound else t_arr - int(round(t_rel[-1]))
                arrivals.append(t_arr)
                k = t0 + np.arange(len(t_rel))
                ok = (k >= 0) & (k < T)
                P = np.full((T, 3), np.nan); V = np.full((T, 3), np.nan)
                A = np.zeros(T, bool); PH = np.full(T, -1, np.int16)
                codes = np.array([phase_code(l) for l in labels], np.int16)
                P[k[ok]] = pos[ok]; V[k[ok]] = vel[ok]; A[k[ok]] = True; PH[k[ok]] = codes[ok]
                rm = np.ones(T)
                if outbound:                            # RCS over profile time, played backwards too
                    rm[k[ok]] = rcs_multiplier(prof, cls, t_rel, rng)[::-1][ok]
                else:
                    rm[k[ok]] = rcs_multiplier(prof, cls, t_rel[ok], rng)
                e = pop._add(cls, variant, alg, role, pop._new_gid(), (P, V, A, PH), ti, profile=name)
                if e is None:                           # entirely outside the run horizon
                    continue
                e.flags["wave"] = wi
                if PLATFORMS[cls].tier == 3:
                    e.transponder = "none"
                if xpdr is not None:                    # explicit choice in the profile wins
                    e.transponder = str(xpdr)
                e.emitting = A.copy() if e.emitter_obs is not None else np.zeros(T, bool)
                e.rcs_mult = rm
                b = np.cumsum([0.0] + [s["duration_s"] for s in segs])
                if outbound:                            # segment j is flown backwards, late in the track first
                    b_start, b_end = b[-1] - b[1:], b[-1] - b[:-1]
                else:
                    b_start, b_end = b[:-1], b[1:]
                for j, s in enumerate(segs):
                    rows.append(dict(eid=e.eid, wave=wi, profile=name, segment=j, phase=s["phase"],
                                     t_start=t0 + b_start[j], t_end=t0 + b_end[j], alt0_km=s["alt_km"][0],
                                     alt1_km=s["alt_km"][1], speed0_mps=s["speed_mps"][0],
                                     speed1_mps=s["speed_mps"][1], weave_g=s["weave_g"],
                                     weave_period_s=s["weave_period_s"], target_asset=ti,
                                     ingress_brg_deg=brg % 360.0, arrival_t=t_arr))
            prev_last_arrival = max(arrivals) if arrivals else prev_last_arrival
    finally:
        pop.rng = saved
    return pd.DataFrame(rows)
