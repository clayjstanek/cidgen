"""Frames and geometry helpers. Local ENU (east, north, up) in metres, flat earth for v0.1."""
from __future__ import annotations

import numpy as np

R_EFF = 4.0 / 3.0 * 6_371_000.0   # 4/3-earth effective radius for radar horizon


def bearing_vec(deg: float) -> np.ndarray:
    """Unit horizontal vector for a compass bearing (deg clockwise from north)."""
    r = np.deg2rad(deg)
    return np.array([np.sin(r), np.cos(r)])


def enu_to_polar(d: np.ndarray):
    """d: (..., 3) relative ENU vector sensor->target. Returns range, az (deg, 0..360), el (deg)."""
    e, n, u = d[..., 0], d[..., 1], d[..., 2]
    horiz = np.hypot(e, n)
    rng = np.sqrt(horiz ** 2 + u ** 2)
    az = np.rad2deg(np.arctan2(e, n)) % 360.0
    el = np.rad2deg(np.arctan2(u, np.maximum(horiz, 1e-6)))
    return rng, az, el


def polar_to_enu(rng, az_deg, el_deg):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    h = rng * np.cos(el)
    return np.stack([h * np.sin(az), h * np.cos(az), rng * np.sin(el)], axis=-1)


def polar_cov_to_enu(rng, az_deg, el_deg, s_r, s_az_deg, s_el_deg):
    """Linearised ENU covariance (N,3,3) for polar measurement noise."""
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    sa, se = np.deg2rad(s_az_deg), np.deg2rad(s_el_deg)
    ca, sa_, ce, se_ = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    # Jacobian d(e,n,u)/d(r,az,el)
    J = np.zeros(np.shape(rng) + (3, 3))
    J[..., 0, 0] = ce * sa_;  J[..., 0, 1] = rng * ce * ca;  J[..., 0, 2] = -rng * se_ * sa_
    J[..., 1, 0] = ce * ca;   J[..., 1, 1] = -rng * ce * sa_; J[..., 1, 2] = -rng * se_ * ca
    J[..., 2, 0] = se_;       J[..., 2, 1] = 0.0;            J[..., 2, 2] = rng * ce
    D = np.zeros_like(J)
    D[..., 0, 0] = np.broadcast_to(s_r, np.shape(rng)) ** 2
    D[..., 1, 1] = np.broadcast_to(sa, np.shape(rng)) ** 2
    D[..., 2, 2] = np.broadcast_to(se, np.shape(rng)) ** 2
    return J @ D @ np.swapaxes(J, -1, -2)


def horizon_range_m(h_sensor_m, h_target_m):
    """Line-of-sight horizon range (m) on a 4/3 earth."""
    return np.sqrt(2 * R_EFF * np.maximum(h_sensor_m, 0)) + np.sqrt(2 * R_EFF * np.maximum(h_target_m, 0))


def angle_diff_deg(a, b):
    """Signed smallest difference a-b in degrees, in (-180, 180]."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def point_to_segment_dist(p, a, b):
    """Horizontal distance from points p (N,2) to segment a-b (2,), plus along-track fraction."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1), t
