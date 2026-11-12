"""Score ranked threat lists against the generator's truth threat priority.

Truth: labels.threat_priority = capability * max_a value_a * exp(-TTG/tau) * 1{CPA < radius}, hostiles
only (0 for everything else). A "real threat" at a snapshot is a track with truth >= threat_min.

Metrics (one row per identity source x ranker)
  threat_captured3    truth threat in the top 3 / truth threat in the optimal top 3, over snapshots with a
                      real threat (1 = the top 3 holds the three most dangerous tracks)
  top_threat_rank     median rank of the single most dangerous track (1 = always first)
  top_threat_rank90   90th percentile of that rank
  order_tau           Kendall tau between list order and truth threat among real threats, snapshots with
                      at least 3 (1 = threats perfectly ordered among themselves)
  ndcg10              ranking quality of the top 10 against truth threat (1 = perfect order)
  top1_in_top3        share of snapshots where the truly most threatening track is in the top 3
  threat_recall3      share of real-threat track-snapshots that sit in the top 3
  first_top3_delay_s  per real-threat episode: seconds from becoming a real threat to first reaching the
                      top 3 (0 if already there); median over episodes
  never_top3          share of real-threat episodes that never reached the top 3
  friend_above_threat share of snapshots with a real threat where a friendly track is ranked above the
                      most threatening real threat (operator attention drawn to a friend first)
  friend_in_top3      share of snapshots with a friendly track in the top 3 (counts quiet periods too)
  friend_engage       share of friendly track-snapshots placed in the ENGAGE tier
  airliner_in_top3    share of snapshots with an airliner in the top 3
  top5_changes_min    tracks entering or leaving the top 5 per minute (list stability)

All the metrics above see only tracks in the picture: a real threat that has no track yet (not confirmed,
dropped, missed) is simply absent, so it is never scored as missed. With truth= (every hostile object's
threat priority at every snapshot time, threat/truth_threat.py) these are added, scored against ALL real
threats. An object counts from its first radar report on (before that no tracker could hold it).
  threat_tracked          share of real-threat object-snapshots whose object has its OWN track in the
                          picture (the track matched to it; association by label: always, once detected,
                          unless the track has coasted out)
  threat_covered          ... or is within 500 m of some track in the picture: members of a group the
                          radars cannot resolve (drone swarm, tight formation) share a group track
  threat_captured3_all    truth threat of the objects whose own tracks are in the top 3 / the three largest
                          truth threats among all objects (strict: a group track earns one member's threat)
  threat_captured3_cov    the same, crediting group tracks: the 3 largest truth threats among objects
                          covered by top-3 tracks (own or group track) / the 3 largest overall
  threat_mass3            group view: each track carries the summed truth threat of every object it covers;
                          the top-3 tracks' total / the largest total any 3 tracks could carry
  top1_in_top3_all        share of snapshots where the most threatening object is covered by a top-3 track
  first_top3_delay_all_s  per real-threat episode (object): seconds from becoming a real threat to first
                          being covered by a top-3 track; median over episodes
  never_top3_all          share of real-threat episodes never covered by a top-3 track
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _tau(a, b):
    """Kendall tau-a between two orderings (no scipy needed)."""
    n = len(a)
    if n < 2:
        return np.nan
    da = np.sign(a[:, None] - a[None]); db = np.sign(b[:, None] - b[None])
    return float((da * db).sum() / (n * (n - 1)))


def _dcg(g):
    return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))


def _score_all(ranked, truth, threat_min, k):
    """Metrics against every real threat (tracked or not); see the module docstring."""
    truth = truth[truth.detected.astype(bool)]            # from the object's first radar report on
    tr = truth[truth.threat_priority >= threat_min]
    if tr.empty:
        return dict(threat_tracked=np.nan, threat_covered=np.nan, threat_captured3_all=np.nan, threat_captured3_cov=np.nan, threat_mass3=np.nan,
                    top1_in_top3_all=np.nan, first_top3_delay_all_s=np.nan, never_top3_all=np.nan)
    own = ranked[["run", "t", "truth_eid"]].drop_duplicates().assign(own=True)
    own_top = ranked[ranked["rank"] <= k][["run", "t", "truth_eid"]].drop_duplicates().assign(own_top=True)
    pic = ranked[["run", "t", "eid"]].rename(columns={"eid": "cover_eid"}).drop_duplicates().assign(cov=True)
    pic_top = ranked[ranked["rank"] <= k][["run", "t", "eid"]].rename(columns={"eid": "cover_eid"}) \
        .drop_duplicates().assign(cov_top=True)
    tt = (truth.merge(own, on=["run", "t", "truth_eid"], how="left")
          .merge(own_top, on=["run", "t", "truth_eid"], how="left")
          .merge(pic, on=["run", "t", "cover_eid"], how="left")
          .merge(pic_top, on=["run", "t", "cover_eid"], how="left"))
    for c in ("own", "own_top", "cov", "cov_top"):
        tt[c] = tt[c].fillna(False).astype(bool)
    tt["cov"] |= tt.own; tt["cov_top"] |= tt.own_top
    real = tt[tt.threat_priority >= threat_min]
    # threat mass: each track carries the summed truth threat of the objects it covers (group tracks
    # carry their whole group); top-3 mass / the best possible 3 tracks' mass
    topk = ranked[ranked["rank"] <= k][["run", "t", "eid"]]
    mass = tt[tt.cover_eid >= 0].groupby(["run", "t", "cover_eid"]).threat_priority.sum().rename("m").reset_index()
    mtop = mass.merge(topk.rename(columns={"eid": "cover_eid"}), on=["run", "t", "cover_eid"])
    best = mass.sort_values("m", ascending=False).groupby(["run", "t"]).head(k).groupby(["run", "t"]).m.sum()
    got = mtop.groupby(["run", "t"]).m.sum().reindex(best.index).fillna(0.0)
    real_t = tt[tt.threat_priority >= threat_min][["run", "t"]].drop_duplicates()
    sel = pd.MultiIndex.from_frame(real_t).intersection(best.index)
    mass3 = float((got.loc[sel] / best.loc[sel]).mean()) if len(sel) else np.nan
    cap, capc, t1 = [], [], []
    for _, s in tt.groupby(["run", "t"], sort=False):
        g = s.threat_priority.to_numpy()
        if g.max() < threat_min:
            continue
        cap.append(g[s.own_top.to_numpy()].sum() / np.sort(g)[::-1][:k].sum())
        gc = np.sort(g[s.cov_top.to_numpy()])[::-1][:k]       # best k objects covered by top-3 tracks
        capc.append(gc.sum() / np.sort(g)[::-1][:k].sum())
        t1.append(bool(s.cov_top.to_numpy()[np.argmax(g)]))
    ep = []
    for (run, eid), e in real.groupby(["run", "truth_eid"]):
        t0 = e.t.min()
        h = tt[(tt.run == run) & (tt.truth_eid == eid) & tt.cov_top & (tt.t >= t0 - 1e-9)]
        ep.append(np.nan if h.empty else h.t.min() - t0)
    ep = np.array(ep, float)
    return dict(threat_tracked=float(real.own.mean()), threat_covered=float(real["cov"].mean()),
                threat_captured3_all=float(np.mean(cap)) if cap else np.nan,
                threat_captured3_cov=float(np.mean(capc)) if capc else np.nan, threat_mass3=mass3,
                top1_in_top3_all=float(np.mean(t1)) if t1 else np.nan,
                first_top3_delay_all_s=float(np.nanmedian(ep)) if np.isfinite(ep).any() else np.nan,
                never_top3_all=float(np.isnan(ep).mean()) if len(ep) else np.nan)


def score_lists(ranked: pd.DataFrame, threat_min=0.05, k=3, truth: pd.DataFrame | None = None):
    """ranked: concatenated outputs of prioritize() with truth columns run, t, eid, rank, tier,
    allegiance, cls, threat_priority."""
    rows, prev_top5 = [], {}
    changes, minutes = 0, 0.0
    for (run, t), s in ranked.groupby(["run", "t"], sort=True):
        s = s.sort_values("rank")
        g = s.threat_priority.fillna(0).values
        ideal = np.sort(g)[::-1]
        nd = _dcg(g[:10]) / _dcg(ideal[:10]) if ideal[:10].sum() > 0 else np.nan
        top = s.head(k)
        real = g >= threat_min
        best = s.eid.values[np.argmax(g)] if g.max() > 0 else None
        fa = cap = brank = tau = np.nan
        if g.max() >= threat_min:
            cap = g[:k].sum() / ideal[:k].sum()
            brank = int(np.argmax(g)) + 1
            if real.sum() >= 3:
                tau = _tau(np.arange(len(g))[real], -g[real])
        if best is not None and g.max() >= threat_min:
            rb = int(s["rank"].values[np.argmax(g)])
            fa = bool(((s.allegiance == 0) & (s["rank"] < rb)).any())
        rows.append(dict(run=run, t=t, ndcg10=nd, friend_above=fa, cap=cap, brank=brank, tau=tau,
                         top1_in_top3=(best in set(top.eid)) if best is not None else np.nan,
                         n_real=int(real.sum()), real_in_top=int((real[:k]).sum()),
                         friend_in_top3=bool((top.allegiance == 0).any()),
                         airliner_in_top3=bool(((top.cls == "E") & (top.allegiance == 1)).any())))
        t5 = set(s.eid.head(5))
        if run in prev_top5:
            pt, pset = prev_top5[run]
            changes += len(t5 ^ pset); minutes += (t - pt) / 60.0
        prev_top5[run] = (t, t5)
    snap = pd.DataFrame(rows)
    # per real-threat episode (run, eid): delay from first real-threat snapshot to first top-k snapshot
    r = ranked.assign(real=ranked.threat_priority.fillna(0) >= threat_min, intop=ranked["rank"] <= k)
    ep = []
    for (run, eid), e in r[r.real].groupby(["run", "eid"]):
        t0 = e.t.min()
        hit = r[(r.run == run) & (r.eid == eid) & r.intop & (r.t >= t0 - 1e-9)]
        ep.append(np.nan if hit.empty else hit.t.min() - t0)
    ep = np.array(ep, float)
    fr = ranked[ranked.allegiance == 0]
    return dict(
        snapshots=len(snap), threat_episodes=len(ep),
        threat_captured3=float(np.nanmean(snap.cap)) if snap.cap.notna().any() else np.nan,
        top_threat_rank=float(np.nanmedian(snap.brank)) if snap.brank.notna().any() else np.nan,
        top_threat_rank90=float(np.nanpercentile(snap.brank, 90)) if snap.brank.notna().any() else np.nan,
        order_tau=float(np.nanmean(snap.tau)) if snap.tau.notna().any() else np.nan,
        ndcg10=float(np.nanmean(snap.ndcg10)) if snap.ndcg10.notna().any() else np.nan,
        top1_in_top3=float(snap.top1_in_top3.dropna().astype(float).mean()) if snap.top1_in_top3.notna().any() else np.nan,
        threat_recall3=float(snap.real_in_top.sum() / max(snap.n_real.sum(), 1)),
        first_top3_delay_s=float(np.nanmedian(ep)) if np.isfinite(ep).any() else np.nan,
        never_top3=float(np.isnan(ep).mean()) if len(ep) else np.nan,
        friend_above_threat=float(snap.friend_above.dropna().astype(float).mean()) if snap.friend_above.notna().any() else np.nan,
        friend_in_top3=float(snap.friend_in_top3.mean()),
        friend_engage=float((fr.tier == "engage").mean()) if len(fr) else np.nan,
        airliner_in_top3=float(snap.airliner_in_top3.mean()),
        top5_changes_min=changes / minutes if minutes else np.nan,
        **(_score_all(ranked, truth, threat_min, k) if truth is not None and "truth_eid" in ranked else {}))
