"""Kinematic engine: waypoint guidance with turn/climb/acceleration limits plus Singer-style
acceleration noise, and formation / swarm followers (spec Section 3.4).

All behaviors are expressed as a waypoint plan: a list of Waypoint(x, y, alt, speed, phase).
The phase of the waypoint currently being flown toward is the intent-phase label.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PHASES = {"transit": 0, "ingress": 1, "attack": 2, "egress": 3, "orbit": 4, "airway": 5,
          "loiter": 6, "wander": 7, "rtb": 8, "hover": 9}


@dataclass
class Waypoint:
    x: float
    y: float
    alt: float
    speed: float
    phase: str
    hold_s: float = 0.0      # hover / loiter time at this waypoint (Tier 1)


# Per-class kinematic limits: (max turn rate deg/s, max climb m/s, max accel m/s^2,
# heading noise deg/s, speed noise m/s, capture radius m)
LIMITS = {
    "A": (15.0, 150.0, 15.0, 0.3, 1.0, 2000.0),
    "B": (8.0, 60.0, 8.0, 0.3, 1.0, 2000.0),
    "C": (2.5, 15.0, 2.0, 0.1, 0.5, 3000.0),
    "D": (5.0, 5.0, 2.0, 0.5, 0.5, 1000.0),
    "E": (2.0, 12.0, 1.5, 0.05, 0.3, 3000.0),
    "F": (4.0, 5.0, 1.5, 0.8, 0.8, 1000.0),
    "R": (10.0, 8.0, 3.0, 1.0, 0.8, 300.0),
    "U1": (90.0, 5.0, 4.0, 6.0, 0.8, 30.0),
    "U2": (45.0, 6.0, 3.0, 3.0, 0.8, 60.0),
    "K": (60.0, 3.0, 3.0, 12.0, 1.5, 40.0),
}


def fly(plan: list[Waypoint], start: tuple[float, float, float], heading_deg: float, speed: float,
        t0: int, T: int, cls: str, rng: np.random.Generator, loop: bool = False,
        end_alive_at_last: bool = False):
    """Integrate one aircraft along its plan at 1 s steps.

    Returns pos (T,3), vel (T,3), alive (T,), phase (T,) arrays indexed by absolute time.
    """
    turn, climb, accel, hnoise, vnoise, capture = LIMITS[cls]
    pos = np.full((T, 3), np.nan)
    vel = np.full((T, 3), np.nan)
    alive = np.zeros(T, bool)
    phase = np.full(T, -1, np.int16)
    x, y, z = start
    hdg = math.radians(heading_deg)
    v = speed
    k = 0
    hold = 0.0
    hrate_noise = 0.0
    beta = 0.05                                    # Singer correlation (1/s)
    sig_h = math.radians(hnoise)
    for t in range(max(t0, 0), T):
        wp = plan[k]
        dx, dy = wp.x - x, wp.y - y
        dist = math.hypot(dx, dy)
        if dist < capture:
            if hold < wp.hold_s:
                hold += 1.0
            elif k + 1 < len(plan):
                k += 1; hold = 0.0
                wp = plan[k]; dx, dy = wp.x - x, wp.y - y; dist = math.hypot(dx, dy)
            elif loop:
                k = 0; hold = 0.0
                wp = plan[k]; dx, dy = wp.x - x, wp.y - y; dist = math.hypot(dx, dy)
            elif end_alive_at_last:
                break
        # heading toward the waypoint, rate-limited, plus correlated noise
        want = math.atan2(dx, dy)                  # compass heading in radians
        err = (want - hdg + math.pi) % (2 * math.pi) - math.pi
        max_t = math.radians(turn)
        hrate_noise += -beta * hrate_noise + sig_h * math.sqrt(2 * beta) * rng.standard_normal()
        hdg += max(-max_t, min(max_t, err)) + hrate_noise
        # speed command: hover/hold slows to the waypoint speed
        v_cmd = wp.speed if not (dist < capture and wp.hold_s > 0) else 0.0
        dv = max(-accel, min(accel, v_cmd - v))
        v = max(0.0, v + dv + vnoise * 0.3 * rng.standard_normal())
        dz = max(-climb, min(climb, wp.alt - z))
        z = max(5.0, z + dz)
        x += v * math.sin(hdg)
        y += v * math.cos(hdg)
        pos[t] = (x, y, z)
        vel[t] = (v * math.sin(hdg), v * math.cos(hdg), dz)
        alive[t] = True
        phase[t] = PHASES[wp.phase] if not (dist < capture and wp.hold_s > 0) else PHASES["hover"] \
            if cls in ("U1", "U2", "R") else PHASES[wp.phase]
    return pos, vel, alive, phase


def follow(leader_pos, leader_vel, leader_alive, leader_phase, offset_body: np.ndarray,
           jitter_m: float, rng: np.random.Generator, dalt: float = 0.0):
    """Formation / swarm follower: leader position + offset rotated into the leader's heading
    frame (x = right, y = forward) + smooth OU jitter. Velocity by finite difference."""
    T = len(leader_pos)
    pos = np.full((T, 3), np.nan)
    hdg = np.arctan2(leader_vel[:, 0], leader_vel[:, 1])
    # smooth heading through hover (speed ~ 0) so offsets don't spin
    spd = np.hypot(leader_vel[:, 0], leader_vel[:, 1])
    for t in range(1, T):
        if np.isfinite(hdg[t]) and spd[t] < 0.5 and np.isfinite(hdg[t - 1]):
            hdg[t] = hdg[t - 1]
    # low-pass the leader heading so the rotated offset does not whip around in turns
    ok = np.isfinite(hdg)
    if ok.any():
        h = np.unwrap(np.where(ok, hdg, np.interp(np.arange(T), np.flatnonzero(ok), hdg[ok])))
        for t in range(1, T):
            h[t] = h[t - 1] + 0.1 * (h[t] - h[t - 1]) if ok[t] else h[t - 1]
        hdg = h
    c, s = np.cos(hdg), np.sin(hdg)
    ox, oy = offset_body
    de = ox * c + oy * s
    dn = -ox * s + oy * c
    # smooth station-keeping jitter: OU noise, twice exponentially smoothed, rescaled to jitter_m
    raw = np.zeros((T, 3))
    beta = 0.02
    for t in range(1, T):
        raw[t] = raw[t - 1] * (1 - beta) + math.sqrt(2 * beta) * rng.standard_normal(3)
    jit = raw.copy()
    for _ in range(2):
        for t in range(1, T):
            jit[t] = jit[t - 1] + 0.1 * (jit[t] - jit[t - 1])
    jit = jit / max(jit.std(), 1e-9) * jitter_m * np.array([1.0, 1.0, 0.3])
    pos[:, 0] = leader_pos[:, 0] + de + jit[:, 0]
    pos[:, 1] = leader_pos[:, 1] + dn + jit[:, 1]
    pos[:, 2] = np.maximum(5.0, leader_pos[:, 2] + dalt + jit[:, 2])
    vel = np.full((T, 3), np.nan)
    vel[1:] = pos[1:] - pos[:-1]
    first = np.argmax(leader_alive) if leader_alive.any() else 0
    vel[first] = leader_vel[first]
    alive = leader_alive.copy()
    pos[~alive] = np.nan
    vel[~alive] = np.nan
    return pos, vel, alive, leader_phase.copy()
