"""Shared sensor utilities: stacked truth, interpolation, aspect, association error, cueing."""
from __future__ import annotations

import numpy as np

from ..geometry import enu_to_polar, horizon_range_m
from ..library import PLATFORMS


class Truth:
    """Entities stacked into (E, T, 3) arrays for vectorised sensing."""

    def __init__(self, entities, dt=1.0):
        self.ents = entities
        self.E = len(entities)
        self.T = len(entities[0].alive)
        self.dt = dt
        self.P = np.stack([e.pos for e in entities])
        self.V = np.stack([e.vel for e in entities])
        self.alive = np.stack([e.alive for e in entities])
        self.emitting = np.stack([e.emitting for e in entities])
        self.eid = np.array([e.eid for e in entities])
        self.cls = np.array([e.cls for e in entities])
        self.tier = np.array([e.tier for e in entities])
        self.rcs = np.array([PLATFORMS[e.cls].rcs_median_m2 for e in entities])
        self.aspect_tab = np.array([PLATFORMS[e.cls].aspect_rcs for e in entities])
        self.swerling = np.array([PLATFORMS[e.cls].swerling for e in entities])
        self.nctr = np.array([PLATFORMS[e.cls].nctr for e in entities])
        self.transponder = np.array([e.transponder for e in entities])
        self.allegiance = np.array([e.allegiance for e in entities])
        self.idx_of = {e.eid: i for i, e in enumerate(entities)}
        # hypersonic (Tier 3): time-varying RCS multiplier (E, T); None when no entity has one,
        # so scenarios without profiles take exactly the original code path
        rm = [getattr(e, "rcs_mult", None) for e in entities]
        self.rcs_mult = None if all(r is None for r in rm) else \
            np.stack([np.ones(self.T) if r is None else r for r in rm])

    def at(self, t: float):
        """Positions, velocities and alive mask at time t (s), linear interpolation."""
        i = int(np.floor(t / self.dt))
        f = t / self.dt - i
        if i >= self.T - 1:
            return self.P[:, -1], self.V[:, -1], self.alive[:, -1]
        p = self.P[:, i] * (1 - f) + self.P[:, i + 1] * f if f > 0 else self.P[:, i]
        v = self.V[:, i]
        a = self.alive[:, i] & (self.alive[:, i + 1] if f > 0 else True)
        return p, v, a

    def step(self, t: float) -> int:
        return min(int(round(t / self.dt)), self.T - 1)


def aspect_deg(p, v, site):
    """Aspect angle (deg) of each target as seen from site: 0 = nose-on, 180 = tail-on."""
    los = site[None, :] - p                       # target -> sensor
    vh = v[:, :2]
    lh = los[:, :2]
    num = np.einsum("ij,ij->i", vh, lh)
    den = np.linalg.norm(vh, axis=1) * np.linalg.norm(lh, axis=1)
    c = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)   # hovering -> beam
    return np.rad2deg(np.arccos(np.clip(c, -1, 1)))


def aspect_rcs_mult(asp, tab):
    """Interpolate nose/beam/tail RCS multipliers with sin^2 weighting."""
    s2 = np.sin(np.deg2rad(asp)) ** 2
    nose, beam, tail = tab[:, 0], tab[:, 1], tab[:, 2]
    front = nose + (beam - nose) * s2
    back = tail + (beam - tail) * s2
    return np.where(asp <= 90, front, back)


def visible(p, alive, site, max_range_m):
    rel = p - site[None, :]
    rng, az, el = enu_to_polar(rel)
    hz = horizon_range_m(site[2], p[:, 2])
    ok = alive & np.isfinite(rng) & (rng <= max_range_m) & (rng <= hz)
    return ok, rng, az, el, rel


def misassociate(true_idx, p, alive, tier, rng, p_base, gate_t1=150.0, gate_t2=3000.0):
    """Return the associated entity index for each report: usually the true one; with probability
    p_base * (1 + neighbours) a nearby wrong entity."""
    out = np.array(true_idx, copy=True)
    for k, i in enumerate(true_idx):
        gate = gate_t1 if tier[i] == 1 else gate_t2
        d = np.linalg.norm(p - p[i], axis=1)
        nb = np.flatnonzero(alive & (d < gate) & (np.arange(len(p)) != i))
        if len(nb) and rng.random() < p_base * (1 + len(nb)):
            out[k] = rng.choice(nb)
    return out


class Cuer:
    """Picks which target a cued sensor (LIDAR, EO/IR, JEM dwell) looks at next."""

    def __init__(self, policy, E, rng):
        self.policy, self.rng = policy, rng
        self.last = np.full(E, -1e9)

    def pick(self, cand, closure, t, k=1):
        cand = np.asarray(cand)
        if self.policy == "none" or len(cand) == 0:
            return np.array([], int)
        if self.policy == "random":
            sel = self.rng.choice(cand, size=min(k, len(cand)), replace=False)
        else:   # suspicious_first: closing targets first, with a staleness bonus for fairness
            stale = np.minimum((t - self.last[cand]) / 30.0, 1.0)
            score = np.clip(closure[cand] / 50.0, 0, 2) + 0.5 * stale + 0.05 * self.rng.random(len(cand))
            sel = cand[np.argsort(-score)[:k]]
        self.last[sel] = t
        return np.asarray(sel, int)


def closure_to_assets(p, v, assets_xy):
    """Closure rate (m/s, positive = closing) toward the nearest asset, per entity."""
    d = p[:, None, :2] - assets_xy[None]
    dist = np.linalg.norm(d, axis=2)
    j = np.nanargmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    rel = d[np.arange(len(p)), j]
    r = np.linalg.norm(rel, axis=1)
    return -np.einsum("ij,ij->i", rel, np.nan_to_num(v[:, :2])) / np.maximum(r, 1.0)
