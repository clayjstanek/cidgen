"""Scenario replay plots for SME review (spec Section 10, visual review): one figure per tier.

    python -m cidgen.replay runs/S0_baseline_seed1                     # -> replay_tier1.png, _tier2, _tier3
    python -m cidgen.replay runs/S0_baseline_seed1 --out S0_replay.png # -> S0_replay_tier1.png, ...

Tier 1  small UAS and birds around the defended assets (12 km)
Tier 2  the air picture (320 km), with ESM fixes and ghosts
Tier 3  hypersonic profile tracks (cidgen/hypersonic_profiles.py): plan view sized to fit the tracks,
        with each radar's maximum instrumented range, and altitude vs time. Written only when the run
        has Tier 3 entities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .io import read_meta, read_table

COL = {0: "#2b6cb0", 1: "#2f855a", 2: "#c53030", -1: "#a0aec0"}
NAME = {0: "Friend", 1: "Neutral", 2: "Hostile", -1: "Bird"}
SITE_MK = {"R_C": ("s", "R-C"), "R_S": ("^", "R-S"), "R_P": ("D", "R-P"), "ESM": ("P", "ESM"), "EOIR": ("*", "EO/IR")}
PROFILE_COL = ["#c53030", "#dd6b20", "#805ad5", "#d53f8c", "#b7791f", "#319795", "#2c5282", "#718096"]


def _load(run):
    d = {k: read_table(run, k) for k in ("truth_states", "entities", "assets", "sites", "airways", "corridor",
                                          "esm_fixes", "features_track")}
    d["truth_states"]["alg"] = d["truth_states"].eid.map(d["entities"].set_index("eid").allegiance)
    return d


def _background(ax, d, lim):
    """Airways, friendly corridor, assets and sensor sites, then axis limits."""
    for r in d["airways"].itertuples():
        ax.plot([r.ae / 1e3, r.be / 1e3], [r.an / 1e3, r.bn / 1e3], color="#cbd5e0", lw=6, alpha=0.5, zorder=0)
    ax.plot(d["corridor"].e / 1e3, d["corridor"].n / 1e3, color="#90cdf4", lw=3, ls="--", zorder=0)
    for r in d["assets"].itertuples():
        ax.add_patch(plt.Circle((r.e / 1e3, r.n / 1e3), r.radius_m / 1e3, fill=False, color="k", lw=1))
        ax.plot(r.e / 1e3, r.n / 1e3, "kx", ms=7)
    for kind, (mk, lab) in SITE_MK.items():
        s = d["sites"][d["sites"].kind == kind]
        if len(s):
            ax.scatter(s.e / 1e3, s.n / 1e3, marker=mk, s=60, facecolor="white", edgecolor="k", zorder=5, label=lab)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.set_xlabel("East (km)"); ax.set_ylabel("North (km)")


def _tracks_by_allegiance(ax, d, ids):
    sub = d["truth_states"][d["truth_states"].eid.isin(ids)]
    for _, g in sub.groupby("eid"):
        a = int(g.alg.iloc[0])
        ax.plot(g.e / 1e3, g.n / 1e3, color=COL[a], lw=0.6 if a == -1 else 1.0, alpha=0.35 if a == -1 else 0.8)
        ax.plot(g.e.iloc[-1] / 1e3, g.n.iloc[-1] / 1e3, "o", ms=2.5, color=COL[a])
    for a, n in NAME.items():
        if (sub.alg == a).any():
            ax.plot([], [], color=COL[a], label=n)


def _save(fig, out, tier):
    path = out.with_name(f"{out.stem}_tier{tier}{out.suffix or '.png'}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


def _held(d, ids):
    return d["features_track"][d["features_track"].eid.isin(ids)].eid.nunique()


def plot_tier1(d, run, out):
    ent = d["entities"]; ids = ent[ent.tier == 1].eid
    fig, ax = plt.subplots(figsize=(9, 9))
    _background(ax, d, 12)
    _tracks_by_allegiance(ax, d, ids)
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title(f"{run.name} | Tier 1 around the defended assets: {len(ids)} entities, {_held(d, ids)} held tracks",
                 loc="left", fontsize=11)
    return _save(fig, out, 1)


def plot_tier2(d, run, out):
    ent = d["entities"]; fx = d["esm_fixes"]; ids = ent[ent.tier == 2].eid
    fig, ax = plt.subplots(figsize=(10, 10))
    _background(ax, d, 320)
    _tracks_by_allegiance(ax, d, ids)
    if len(fx):
        ghost = fx.is_ghost.astype(bool)
        ax.scatter(fx.e[~ghost] / 1e3, fx.n[~ghost] / 1e3, s=2, color="#d69e2e", alpha=0.25, label="ESM fix")
        ax.scatter(fx.e[ghost] / 1e3, fx.n[ghost] / 1e3, s=2, color="#805ad5", alpha=0.25, label="ESM ghost")
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    ghosts = f" ({fx.is_ghost.astype(bool).mean():.0%} ghosts)" if len(fx) else ""
    ax.set_title(f"{run.name} | Tier 2 air picture: {len(ids)} entities, {_held(d, ids)} held tracks, "
                 f"{len(fx):,} ESM fixes{ghosts}", loc="left", fontsize=11)
    return _save(fig, out, 2)


def plot_tier3(d, run, out):
    ent = d["entities"]; ids = ent[ent.tier == 3].eid
    if not len(ids):
        return None
    tr = d["truth_states"]; sub = tr[tr.eid.isin(ids)]
    prof = ent.set_index("eid")["flag_profile"] if "flag_profile" in ent else ent.set_index("eid").cls
    held = set(d["features_track"].eid)
    names = list(dict.fromkeys(prof.loc[ids].astype(str)))
    pcol = {n: PROFILE_COL[i % len(PROFILE_COL)] for i, n in enumerate(names)}
    ext = np.nanmax(np.abs(sub[["e", "n"]].values)) / 1e3
    lim = max(320.0, float(np.ceil(ext * 1.05 / 100.0) * 100.0))

    fig, (ax, ah) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw=dict(width_ratios=[1.1, 1]))
    _background(ax, d, lim)
    try:                                                   # radar maximum instrumented range rings
        rts = read_meta(run)["config"]["sensors"]["radar_types"]
        for s in d["sites"].itertuples():
            if s.kind in rts:
                rmax = rts[s.kind]["ref_range_km"] * rts[s.kind]["max_range_factor"]
                ax.add_patch(plt.Circle((s.e / 1e3, s.n / 1e3), rmax, fill=False, color="#4a5568", lw=0.8, ls=":"))
        ax.plot([], [], color="#4a5568", lw=0.8, ls=":", label="Radar max range")
    except (KeyError, TypeError):
        pass
    for eid, g in sub.groupby("eid"):
        c = pcol[str(prof.loc[eid])]
        ls = "-" if eid in held else "--"                  # dashed: never held as a track
        ax.plot(g.e / 1e3, g.n / 1e3, color=c, lw=1.2, ls=ls, alpha=0.9)
        ax.plot(g.e.iloc[-1] / 1e3, g.n.iloc[-1] / 1e3, "o", ms=4, color=c)
        ah.plot(g.t, g.u / 1e3, color=c, lw=1.2, ls=ls, alpha=0.9)
    for n in names:
        k = (prof.loc[ids].astype(str) == n).sum()
        ax.plot([], [], color=pcol[n], lw=2, label=f"{n} ({k})")
        ah.plot([], [], color=pcol[n], lw=2, label=f"{n} ({k})")
    if not set(ids) <= held:
        ax.plot([], [], color="k", lw=1, ls="--", label="never held as a track")
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title(f"{run.name} | Tier 3 plan view: {len(ids)} entities, {_held(d, ids)} held tracks",
                 loc="left", fontsize=11)
    ah.set_xlabel("Run time (s)"); ah.set_ylabel("Altitude (km)"); ah.grid(alpha=0.3); ah.set_ylim(bottom=0)
    ah.set_title("Tier 3 altitude vs time", loc="left", fontsize=11)
    ah.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return _save(fig, out, 3)


def plot(run, out=None):
    """Write one PNG per tier; returns the paths written (Tier 3 only if the run has Tier 3 entities)."""
    run = Path(run)
    out = Path(out) if out else run / "replay.png"
    d = _load(run)
    paths = [plot_tier1(d, run, out), plot_tier2(d, run, out), plot_tier3(d, run, out)]
    return [p for p in paths if p is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None, help="base name; _tier1/_tier2/_tier3 are added")
    a = ap.parse_args()
    for p in plot(a.run, a.out):
        print(p)


if __name__ == "__main__":
    main()
