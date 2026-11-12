"""Threat prioritisation: rank every track in one snapshot, most threatening first.

    tp = ThreatPrioritizer.from_yaml("configs/threat/doctrine_default.yaml")
    ranked = tp.prioritize(picture, t_now)          # picture: one row per track (see PICTURE_COLUMNS)

Design
- `prioritize()` never modifies its input. It reads the picture, builds a new table and returns it.
  No copy of the caller's data is needed because nothing is written back.
- The only memory across updates lives in the ThreatPrioritizer object: sticky flags (e.g. a track
  declared hostile stays declared), the last smoothed score per track (rank hysteresis) and the last
  update time. Create one object per scenario run; call reset() between runs.
- Identity (P Friend / Neutral / Hostile) comes from any CID model through the picture, so BN, XGBoost,
  GNN or the stacked ensemble can be swapped without touching this file.

Score per track
    urgency   = max over assets a of  value_a * exp(-TTG_a / tau) * 1{CPA_a < radius_a * margin}
    lethality = first matching row of the doctrine's lethality table (observable class)
    score     = P(H) * lethality * urgency + w_amb * (1 - max P) * lethality * urgency
                (both terms x group_n, the estimated members of a group track, when run_tp --group-count is used)
then the doctrine's rules, in order, adjust the score, set flags and choose the tier.
Ranking: tier (engage, monitor, protected), then smoothed score, then P(Friend) ascending.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Columns the picture must carry; missing optional columns are treated as unknown (NaN).
REQUIRED = ["eid", "p_friend", "p_neutral", "p_hostile"]
OPTIONAL = ["ttg_s", "cpa_m", "asset_value", "asset_radius_m", "urgency_raw", "obs_tier", "speed", "alt",
            "rad_rcs_med30", "iff_mode5_valid", "iff_ever_valid", "iff_mode3c", "corridor_dist_m",
            "airway_xtrack_m", "airway_alt_inband", "descending_closing", "threat_offset_deg",
            "asset_closure_mps", "esm_band", "group_n"]
PICTURE_COLUMNS = REQUIRED + OPTIONAL
TIERS = {"engage": 0, "monitor": 1, "protected": 2}
OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne}


def _cond(df, conds):
    """AND of [field, op, value] conditions over the table; unknown values never satisfy a condition."""
    m = np.ones(len(df), bool)
    for f, op, v in conds:
        if f not in df:
            return np.zeros(len(df), bool)
        col = pd.to_numeric(df[f], errors="coerce") if not isinstance(v, str) else df[f]
        m &= np.asarray(OPS[op](col, v), dtype=bool) & np.asarray(pd.notna(col), dtype=bool)
    return m


@dataclass
class ThreatPrioritizer:
    doctrine: dict
    sticky: dict = field(default_factory=dict)        # eid -> set of sticky flag names
    smoothed: dict = field(default_factory=dict)      # eid -> last smoothed score
    last_t: float | None = None

    @classmethod
    def from_yaml(cls, path):
        return cls(yaml.safe_load(open(path)))

    def reset(self):
        self.sticky, self.smoothed, self.last_t = {}, {}, None

    # ------------------------------------------------------------------ scoring pieces
    def lethality(self, P):
        out = np.full(len(P), np.nan); name = np.full(len(P), "", dtype=object)
        for row in self.doctrine["lethality"]:
            m = np.isnan(out) & _cond(P, row.get("when", []))
            out[m] = row["value"]; name[m] = row["name"]
        return np.nan_to_num(out, nan=0.5), name

    def needs_update(self, picture, t_now):
        """Event trigger for faster-than-scan updates: a new track, or one with a short time-to-go."""
        new = ~picture.eid.isin(list(self.smoothed))
        short = pd.to_numeric(picture.get("ttg_s"), errors="coerce") < self.doctrine["popup"]["ttg_s"]
        return bool(new.any() or np.asarray(short.fillna(False)).any())

    # ------------------------------------------------------------------ main entry point
    def prioritize(self, picture: pd.DataFrame, t_now: float) -> pd.DataFrame:
        """Rank one snapshot. Returns a new table; `picture` is left unaltered."""
        d = self.doctrine
        missing = [c for c in REQUIRED if c not in picture]
        if missing:
            raise ValueError(f"picture lacks required columns {missing}")
        P = pd.DataFrame({c: picture[c].values for c in PICTURE_COLUMNS if c in picture}, index=range(len(picture)))
        for c in OPTIONAL:
            if c not in P:
                P[c] = np.nan
        u = d["urgency"]
        urg = pd.to_numeric(P.urgency_raw, errors="coerce").fillna(0.0).values
        P["urgency"] = np.maximum(urg, u.get("min_urgency", 0.0))
        P["lethality"], P["lethality_class"] = self.lethality(P)
        pmax = P[["p_friend", "p_neutral", "p_hostile"]].max(axis=1).values
        # group tracks: an estimated member count multiplies the threat (sum over similar members)
        base = P.lethality * P.urgency * pd.to_numeric(P.group_n, errors="coerce").fillna(1.0).clip(lower=1.0)
        P["score_raw"] = P.p_hostile * base + d.get("ambiguity_weight", 0.0) * (1 - pmax) * base
        P["score"] = P.score_raw.astype(float)
        P["tier"] = "monitor"
        P["engage_block"] = False
        P["reasons"] = [[] for _ in range(len(P))]
        flags = [set(self.sticky.get(e, set())) for e in P.eid]
        P["flagset"] = flags
        for rule in d.get("rules", []):
            # flags written by earlier rules are visible to later ones as 0/1 columns
            for fl in {f for fs in P["flagset"] for f in fs}:
                P[fl] = [int(fl in fs) for fs in P["flagset"]]
            m = _cond(P, rule.get("when", []))
            if not m.any():
                continue
            act = rule.get("do", {})
            if "multiply" in act:
                P.loc[m, "score"] *= act["multiply"]
            if "floor" in act:
                P.loc[m, "score"] = np.maximum(P.loc[m, "score"], act["floor"])
            if "cap" in act:
                P.loc[m, "score"] = np.minimum(P.loc[m, "score"], act["cap"])
            if "tier" in act:
                P.loc[m, "tier"] = act["tier"]
            if act.get("engage_block"):
                P.loc[m, "engage_block"] = True
            for key in ("set_flag", "sticky_flag"):
                if key in act:
                    for i in np.nonzero(m)[0]:
                        P.at[i, "flagset"] = P.at[i, "flagset"] | {act[key]}
                        if key == "sticky_flag":
                            self.sticky.setdefault(P.eid.iat[i], set()).add(act[key])
            for i in np.nonzero(m)[0]:
                P.at[i, "reasons"] = P.at[i, "reasons"] + [rule.get("reason", rule["name"])]
        # engage eligibility from the weapons-control state
        crit = d["engage_criteria"][d.get("weapons_control", "tight")]
        if not crit.get("never"):
            elig = P.p_hostile >= crit.get("min_p_hostile", 1.0)
            if crit.get("or_flag"):
                elig |= P["flagset"].apply(lambda s: crit["or_flag"] in s)
            elig &= (P.tier != "protected") & ~P.engage_block
            P.loc[elig, "tier"] = "engage"
        # hysteresis: rise immediately, decay no faster than decay_tau
        dt = 0.0 if self.last_t is None else max(t_now - self.last_t, 0.0)
        decay = np.exp(-dt / d["hysteresis"]["decay_tau_s"]) if dt else 1.0
        prev = np.array([self.smoothed.get(e, 0.0) for e in P.eid])
        P["score_smoothed"] = np.where(P.tier == "protected", P.score, np.maximum(P.score, prev * decay))
        P["popup"] = (~P.eid.isin(list(self.smoothed))) | (pd.to_numeric(P.ttg_s, errors="coerce") < d["popup"]["ttg_s"])
        self.smoothed = dict(zip(P.eid, P.score_smoothed))
        self.last_t = t_now
        P["tier_order"] = P.tier.map(TIERS)
        P = P.sort_values(["tier_order", "score_smoothed", "score_raw", "p_friend"], ascending=[True, False, False, True],
                          kind="mergesort")
        P.insert(0, "rank", np.arange(1, len(P) + 1))
        P["flags"] = P["flagset"].apply(lambda s: ",".join(sorted(s)))
        P["reasons"] = P.reasons.apply(lambda r: "; ".join(r))
        P.insert(1, "t", t_now)
        return P.drop(columns=["tier_order", "flagset"]).reset_index(drop=True)


class PHRanker:
    """Baseline: rank by P(Hostile), ties broken by P(Friend) ascending. No doctrine, no memory."""

    def reset(self):
        pass

    def prioritize(self, picture, t_now):
        P = picture[["eid", "p_friend", "p_neutral", "p_hostile"]].copy()
        P = P.sort_values(["p_hostile", "p_friend"], ascending=[False, True], kind="mergesort")
        P["score_smoothed"] = P.p_hostile
        P["tier"] = np.where(P.p_hostile >= 0.5, "engage", "monitor")
        P.insert(0, "rank", np.arange(1, len(P) + 1)); P.insert(1, "t", t_now)
        return P.reset_index(drop=True)


def no_rules(doctrine):
    """The same doctrine with the rules removed: expected-threat score only."""
    d = dict(doctrine); d["rules"] = []; d["name"] = doctrine.get("name", "doctrine") + "-norules"
    return d
