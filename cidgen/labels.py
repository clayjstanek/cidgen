"""Ground-truth labels (spec Section 6.1), keyed to the same (eid, t) rows as features_track."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .entities import PHASE_NAMES
from .library import PLATFORMS


def threat_priority(e, t_idx, world, tau):
    """T_j(t) = c_j * max_i v_i exp(-TTG_ij/tau) 1{CPA_ij < d_i}, hostiles only."""
    if e.allegiance != 2:
        return 0.0
    p, v = e.pos[t_idx, :2], e.vel[t_idx, :2]
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)):
        return 0.0
    best = 0.0
    for a in world.assets:
        rel = p - a.pos[:2]
        r = np.linalg.norm(rel)
        clos = -rel @ v / max(r, 1.0)
        if clos <= 1.0 and r > a.radius_m:
            continue
        ttg = r / clos if clos > 1.0 else 0.0
        tc = max(-rel @ v / max(v @ v, 1e-6), 0.0)
        cpa = np.linalg.norm(rel + tc * v)
        if cpa < a.radius_m:
            best = max(best, a.value * np.exp(-ttg / tau))
    return PLATFORMS[e.cls].capability * best


def build_labels(cfg, world, entities, feats: pd.DataFrame, radar: pd.DataFrame, iff, jem, eoir):
    by_id = {e.eid: e for e in entities}
    tau = cfg["labels"]["tau_s"]
    keys = feats[["eid", "t"]].copy()

    # evidence sufficiency: time each entity first received decisive evidence of each kind
    first = {}
    def note(df, cond, level):
        if df is None or len(df) == 0:
            return
        d = df[cond(df)]
        for eid, tmin in d.groupby("eid_true").time.min().items():
            first.setdefault(int(eid), []).append((tmin, level))
    note(iff, lambda d: d.mode5_valid & d.eid_true.map(lambda x: by_id[x].transponder in ("mode5_l1", "mode5_l2")), 1.0)
    note(jem, lambda d: (d.top_conf > 0.8) & (d.eid == d.eid_true), 0.7)
    note(eoir, lambda d: (d.top_conf > 0.8) & (d.eid == d.eid_true), 0.7)

    # unresolved formations: lead eid of a merged radar report
    m = radar[(radar.n_merged > 1) & ~radar.false_alarm][["eid", "time", "n_merged"]].copy()
    m["tbin"] = np.ceil(m.time / cfg["run"]["feature_dt_s"]) * cfg["run"]["feature_dt_s"]
    merged = m.groupby(["eid", "tbin"]).n_merged.max()

    rows = []
    for eid, t in keys.itertuples(index=False):
        e = by_id[int(eid)]
        ti = min(int(round(t)), len(e.alive) - 1)
        ev = max([lvl for (tm, lvl) in first.get(int(eid), []) if tm <= t], default=0.0)
        nm = int(merged.get((eid, t), 1))
        rows.append((eid, t, e.allegiance, e.cls, e.variant, e.role, e.group_id, e.tier, e.is_ooi,
                     PHASE_NAMES.get(int(e.phase[ti]), "none"), threat_priority(e, ti, world, tau),
                     nm > 1, nm, ev))
    lab = pd.DataFrame(rows, columns=["eid", "t", "allegiance", "cls", "variant", "role", "group_id", "tier",
                                      "is_object_of_interest", "intent_phase", "threat_priority",
                                      "multi_aircraft", "member_count", "evidence_sufficiency"])
    return lab


def entity_table(entities):
    rows = []
    for e in entities:
        rows.append(dict(eid=e.eid, allegiance=e.allegiance, cls=e.cls, variant=e.variant, role=e.role,
                         group_id=e.group_id, tier=e.tier, is_object_of_interest=e.is_ooi,
                         transponder=e.transponder, emitter_obs=e.emitter_obs, engine_obs=e.engine_obs,
                         len_m=e.dims[0], span_m=e.dims[1], tail_m=e.dims[2], target_asset=e.target_asset,
                         **{f"flag_{k}": (str(v) if isinstance(v, list) else v) for k, v in e.flags.items()}))
    return pd.DataFrame(rows)


def truth_states(entities, every_s=1):
    out = []
    for e in entities:
        idx = np.flatnonzero(e.alive)[::every_s]
        out.append(pd.DataFrame(dict(eid=e.eid, t=idx.astype(float), e=e.pos[idx, 0], n=e.pos[idx, 1], u=e.pos[idx, 2],
                                     ve=e.vel[idx, 0], vn=e.vel[idx, 1], vu=e.vel[idx, 2],
                                     emitting=e.emitting[idx], phase=e.phase[idx])))
    return pd.concat(out, ignore_index=True)
