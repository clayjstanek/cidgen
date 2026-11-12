"""World model (spec Section 2): defended assets, airways, friendly corridor, sensor sites."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import bearing_vec


@dataclass
class Asset:
    asset_id: int
    pos: np.ndarray          # ENU (m)
    value: float
    radius_m: float


@dataclass
class Airway:
    airway_id: int
    a: np.ndarray            # endpoints (2,), metres
    b: np.ndarray
    alt_band_m: tuple[float, float]


@dataclass
class SensorSite:
    site_id: str             # e.g. "R_S-1", "ESM-2"
    kind: str                # R_C | R_S | R_P | ESM | EOIR
    pos: np.ndarray          # ENU (m)


@dataclass
class World:
    threat_axis_deg: float
    assets: list[Asset]
    airways: list[Airway]
    corridor: np.ndarray     # polyline (K,2) friendly return corridor, from outside toward base
    town: np.ndarray         # (2,) populated area for hobbyist drones
    sites: list[SensorSite] = field(default_factory=list)

    def site(self, site_id):
        return next(s for s in self.sites if s.site_id == site_id)

    def sites_of(self, kind):
        return [s for s in self.sites if s.kind == kind]


def build_world(cfg: dict, rng: np.random.Generator) -> World:
    w = cfg["world"]
    s = cfg["sensors"]
    axis = w["threat_axis_deg"]
    axis = float(rng.uniform(0.0, 360.0)) if axis in (None, "random") else float(axis)

    # Assets: primary at origin, others within asset_spread_km, biased away from threat axis.
    assets = [Asset(0, np.zeros(3), 1.0, 5000.0)]
    for i in range(1, int(w["n_assets"])):
        brg = axis + 180.0 + rng.uniform(-100, 100)
        d = rng.uniform(0.4, 1.0) * w["asset_spread_km"] * 1000.0
        xy = bearing_vec(brg) * d
        assets.append(Asset(i, np.array([xy[0], xy[1], 0.0]), float(rng.uniform(0.4, 0.8)), 3000.0))

    # Airways: straight lines crossing the volume, at random offsets from the origin.
    airways = []
    lo, hi = w["airway_alt_band_km"]
    for i in range(int(w["n_airways"])):
        heading = rng.uniform(0, 180)
        offset = rng.uniform(-60_000, 60_000)
        u = bearing_vec(heading)
        nrm = np.array([u[1], -u[0]])
        c = nrm * offset
        airways.append(Airway(i, c - u * 350_000, c + u * 350_000, (lo * 1000, hi * 1000)))

    # Friendly return corridor: from 180 km out at (axis + offset) bearing, bending into base.
    cb = axis + w["corridor_offset_deg"]
    corridor = np.stack([bearing_vec(cb) * 180_000, bearing_vec(cb) * 60_000, bearing_vec(cb - 20) * 15_000,
                         np.zeros(2)])
    town = bearing_vec(axis + rng.uniform(60, 120)) * rng.uniform(3000, 6000)

    world = World(axis, assets, airways, corridor, town)

    # Sensor sites (spec Section 4): R-C at the first assets, R-S at assets, R-P near the primary.
    def at_asset(k, jitter=500.0):
        a = assets[k % len(assets)].pos[:2]
        return a + rng.uniform(-jitter, jitter, 2)

    rt = s["radar_types"]
    for i in range(int(s["n_R_C"])):
        p = at_asset(i, 200)
        world.sites.append(SensorSite(f"R_C-{i+1}", "R_C", np.array([p[0], p[1], rt["R_C"]["height_m"]])))
    for i in range(int(s["n_R_S"])):
        p = at_asset(i if i == 0 else i + 1, 800)
        world.sites.append(SensorSite(f"R_S-{i+1}", "R_S", np.array([p[0], p[1], rt["R_S"]["height_m"]])))
    for i in range(int(s["n_R_P"])):
        p = bearing_vec(axis + 180 + 30 * i) * 4000
        world.sites.append(SensorSite(f"R_P-{i+1}", "R_P", np.array([p[0], p[1], rt["R_P"]["height_m"]])))

    # ESM: site 1 co-located with R_S-1; sites 2 and 3 displaced on a baseline, either side of the axis.
    rs1 = world.site("R_S-1").pos if s["n_R_S"] > 0 else np.array([0.0, 0.0, 10.0])
    base = s["esm_baseline_km"] * 1000.0
    esm_pos = [rs1[:2], rs1[:2] + bearing_vec(axis + 90) * base, rs1[:2] + bearing_vec(axis - 90) * base]
    for i, (p, up) in enumerate(zip(esm_pos, s["esm_sites_up"])):
        if up:
            world.sites.append(SensorSite(f"ESM-{i+1}", "ESM", np.array([p[0], p[1], 10.0])))

    # One EO/IR camera, co-located with R_C-1.
    if s["n_R_C"] > 0:
        world.sites.append(SensorSite("EOIR-1", "EOIR", world.site("R_C-1").pos + np.array([20.0, 0, 0])))
    return world
