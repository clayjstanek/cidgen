"""Per-tier Bayesian Network baselines (spec Section 9).

Structure (the conventional fielded form): root Allegiance A in {Friend, Neutral, Hostile}, a
hidden Platform Class C beneath it, and discretised observation nodes whose parents are A, C or
(A, C). Inference is exact: P(A | e) = sum_c P(A) P(c | A) prod_i P(e_i | pa_i), skipping missing
evidence. This equals what pgmpy's VariableElimination returns for the same CPTs; `to_pgmpy()`
exports the network for cross-checking.

Variants
  learned   CPTs by maximum likelihood with a Dirichlet prior, from the training split
  oracle    the same structure fitted on all available runs (upper bound for the structure)
  sme       oracle CPTs perturbed: log-odds noise, rounding to 0.05, overconfidence on strong cues
  context   learned + neighbour-context nodes (strongest fair BN; spec "BN-context")

Each snapshot is scored independently: the features already summarise history (rolling RCS,
IFF ever-valid, last NCTR report), so recursive fusion here would count the same evidence twice.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..library import EMITTER_CODES, ENGINE_CODES, EOIR_CLASSES

TIER_CLASSES = {1: ["U1", "U2"], 2: ["A", "B", "C", "D", "E", "F", "R"],
                3: ["H", "G", "X","Y", "Z"}                                  # hypersonic (Tier 3)
TIER3_MIN_SPEED_MPS = 1000.0                               # hypersonic (Tier 3) observable-tier rule


# ----------------------------------------------------------------------------- discretisers
def _bins(x, edges):
    x = pd.to_numeric(x, errors="coerce").astype(float).values
    s = np.digitize(x, edges)
    return np.where(np.isfinite(x), s, -1)


def _cat(x, codes):
    idx = {c: i for i, c in enumerate(codes)}
    return np.array([idx.get(str(v), -1) for v in x])


def _flag(x, thr=0.5):
    x = pd.to_numeric(x, errors="coerce").astype(float).values
    return np.where(np.isfinite(x), (x > thr).astype(int), -1)


def iff_state(df):
    """0 out of coverage, 1 in coverage but silent, 2 Mode 3/C only, 3 Mode 5 valid L1,
    4 Mode 5 valid L2 consistent with track, 5 Mode 5 valid L2 inconsistent, 6 Mode 5 invalid."""
    replied = df.iff_age_s.notna().values
    cov = pd.to_numeric(df.iff_in_coverage, errors="coerce").fillna(0).values > 0.5
    m5 = pd.to_numeric(df.iff_mode5, errors="coerce").fillna(0).values > 0.5
    ok = pd.to_numeric(df.iff_mode5_valid, errors="coerce").fillna(0).values > 0.5
    l2 = pd.to_numeric(df.iff_level2, errors="coerce").fillna(0).values > 0.5
    resid = pd.to_numeric(df.iff_l2_resid_m, errors="coerce").fillna(np.inf).values
    s = np.where(cov, 1, 0)
    s = np.where(replied, 2, s)
    s = np.where(replied & m5 & ok & ~l2, 3, s)
    s = np.where(replied & m5 & ok & l2 & (resid < 2000), 4, s)
    s = np.where(replied & m5 & ok & l2 & (resid >= 2000), 5, s)
    s = np.where(replied & m5 & ~ok, 6, s)
    return s


def closing_state(df, tier):
    clos = pd.to_numeric(df.asset_closure_mps, errors="coerce").fillna(0).values
    ttg = pd.to_numeric(df.asset_min_ttg_s, errors="coerce").fillna(1e5).values
    a, b = (900, 300) if tier == 2 else (300, 60)
    return np.select([clos <= 1.0, ttg > a, ttg > b], [0, 1, 2], 3)


def sector_state(df, tier):
    if "threat_offset_deg" in df:
        d = pd.to_numeric(df.threat_offset_deg, errors="coerce").fillna(180).values
    else:
        axis = df.threat_axis_deg.values if "threat_axis_deg" in df else 0.0
        brg = np.rad2deg(np.arctan2(df.e.values, df.n.values)) % 360
        d = np.abs((brg - axis + 180) % 360 - 180)
    sec = np.select([d < 45, d < 120], [0, 1], 2)
    r = np.hypot(df.e.values, df.n.values)
    near, mid = (30_000, 120_000) if tier >= 2 else (2_000, 6_000)      # tier 3 uses Tier 2 ranges
    rb = np.select([r < near, r < mid], [0, 1], 2)
    return sec * 3 + rb


@dataclass
class Node:
    name: str
    parents: str            # "A", "C" or "AC"
    n_states: int
    fn: object              # df -> int states, -1 = missing
    strong: bool = False    # experts tend to be overconfident about these cues (SME variant)


def node_specs(tier, context=False):
    if tier == 3:                                          # hypersonic (Tier 3): kinematics, RCS, IFF, geometry
        nodes = [
            Node("speed", "AC", 7, lambda d: _bins(d.speed, [600, 1000, 1500, 2000, 3000, 4500])),
            Node("alt", "AC", 6, lambda d: _bins(d.alt, [5000, 15000, 25000, 40000, 60000])),
            Node("climb", "AC", 5, lambda d: _bins(d.climb, [-300, -50, 50, 300])),
            Node("rcs", "C", 7, lambda d: _bins(d.rad_rcs_med30, [-15, -5, 0, 5, 10, 15])),
            Node("iff", "A", 7, iff_state, strong=True),
            Node("closing", "A", 4, lambda d: closing_state(d, 3)),
            Node("sector", "A", 9, lambda d: sector_state(d, 3)),
        ]
    elif tier == 2:
        nodes = [
            Node("speed", "AC", 9, lambda d: _bins(d.speed, [30, 60, 120, 180, 230, 280, 350, 450])),
            Node("alt", "AC", 7, lambda d: _bins(d.alt, [100, 500, 2000, 5000, 8000, 11000])),
            Node("rcs", "C", 7, lambda d: _bins(d.rad_rcs_med30, [-15, -5, 0, 5, 10, 15])),
            Node("iff", "A", 7, iff_state, strong=True),
            Node("jem", "AC", len(ENGINE_CODES), lambda d: _cat(d.jem_top_engine, ENGINE_CODES)),
            Node("esm", "AC", len(EMITTER_CODES), lambda d: _cat(d.esm_top_emitter, EMITTER_CODES), strong=True),
            Node("lidar", "AC", 8, lambda d: _bins(d.lidar_len_fuselage_m, [5, 10, 14, 16, 18, 25, 40])),
            Node("airway", "A", 2, lambda d: ((d.airway_xtrack_m < 3000) & (d.airway_alt_inband > 0.5)).astype(int).values),
            Node("corridor", "A", 2, lambda d: (d.corridor_dist_m < 5000).astype(int).values),
            Node("closing", "A", 4, lambda d: closing_state(d, 2)),
            Node("descent", "A", 2, lambda d: _flag(d.descending_closing, 0.3)),
            Node("sector", "A", 9, lambda d: sector_state(d, 2)),
            Node("jam", "A", 2, lambda d: _flag(d.rad_jam_flag)),
        ]
    else:
        nodes = [
            Node("speed", "AC", 6, lambda d: _bins(d.speed, [2, 8, 15, 25, 35])),
            Node("alt", "AC", 5, lambda d: _bins(d.alt, [40, 80, 150, 300])),
            Node("rcs", "C", 6, lambda d: _bins(d.rad_rcs_med30, [-25, -20, -15, -10, -5])),
            Node("md_rate", "C", 5, lambda d: _bins(d.rad_md_rate, [0.3, 0.5, 0.7, 0.85])),
            Node("md_period", "C", 4, lambda d: _bins(d.rad_md_period, [0.5, 0.7, 0.85])),
            Node("eoir", "C", len(EOIR_CLASSES), lambda d: _cat(d.eoir_top_class, EOIR_CLASSES)),
            Node("eoir_friend", "A", 2, lambda d: np.where(d.eoir_top_class.notna(),
                                                           (d.eoir_friend_variant_recognized.astype(str) == "True").astype(int), -1)),
            Node("iff", "A", 7, iff_state, strong=True),
            Node("esm", "AC", len(EMITTER_CODES), lambda d: _cat(d.esm_top_emitter, EMITTER_CODES), strong=True),
            Node("closing", "A", 4, lambda d: closing_state(d, 1)),
            Node("sector", "A", 9, lambda d: sector_state(d, 1)),
        ]
    if context:
        nodes += [
            Node("nb_count", "A", 3, lambda d: _bins(d.nb_count, [0.5, 2.5])),
            Node("nb_valid_iff", "A", 2, lambda d: _flag(d.nb_any_valid_iff)),
            Node("nb_closing", "A", 3, lambda d: _bins(d.nb_asset_closure_mps, [1.0, 50.0])),
            Node("nb_comove", "A", 3, lambda d: _bins(d.nb_comove, [0.2, 0.6])),
            Node("nb_jam", "A", 2, lambda d: _flag(d.rad_jam_flag) if tier == 2 else -np.ones(len(d), int)),
        ]
    return nodes


def observable_tier(df):
    """Tier from observables only (the BN must not read the label): slow, low, small -> Tier 1."""
    sp = pd.to_numeric(df.speed, errors="coerce").fillna(0).values
    alt = pd.to_numeric(df.alt, errors="coerce").fillna(0).values
    rcs = pd.to_numeric(df.rad_rcs_med30, errors="coerce").fillna(0).values
    tier = np.where((sp < 60) & (alt < 1500) & (rcs < -5), 1, 2)
    return np.where(sp >= TIER3_MIN_SPEED_MPS, 3, tier)    # hypersonic (Tier 3)


# ----------------------------------------------------------------------------- one tier
class TierBN:
    def __init__(self, tier, context=False, alpha=1.0):
        self.tier, self.context, self.alpha = tier, context, alpha
        self.classes = TIER_CLASSES[tier]
        self.nodes = node_specs(tier, context)
        self.prior = None; self.p_c = None; self.cpt = {}

    def _states(self, df):
        return {n.name: np.asarray(n.fn(df), int) for n in self.nodes}

    def fit(self, df):
        cidx = {c: i for i, c in enumerate(self.classes)}
        d = df[df.cls.isin(self.classes)]
        A = d.allegiance.values.astype(int); C = d.cls.map(cidx).values.astype(int)
        nA, nC = 3, len(self.classes)
        self.prior = (np.bincount(A, minlength=nA) + self.alpha) / (len(A) + nA * self.alpha)
        pc = np.full((nA, nC), self.alpha)
        np.add.at(pc, (A, C), 1)
        self.p_c = pc / pc.sum(1, keepdims=True)
        st = self._states(d)
        for n in self.nodes:
            s = st[n.name]; ok = s >= 0
            k = np.full((nA, nC, n.n_states), self.alpha)
            if n.parents == "A":
                kk = np.full((nA, n.n_states), self.alpha); np.add.at(kk, (A[ok], s[ok]), 1)
                k = np.repeat(kk[:, None, :], nC, axis=1)
            elif n.parents == "C":
                kk = np.full((nC, n.n_states), self.alpha); np.add.at(kk, (C[ok], s[ok]), 1)
                k = np.repeat(kk[None], nA, axis=0)
            else:
                np.add.at(k, (A[ok], C[ok], s[ok]), 1)
            self.cpt[n.name] = k / k.sum(-1, keepdims=True)
        return self

    def log_joint(self, df):
        """log P(A, C, e) for every row: (N, 3, nC)."""
        st = self._states(df)
        N = len(df)
        lj = np.log(self.prior)[None, :, None] + np.log(self.p_c)[None]
        lj = np.repeat(lj, N, axis=0)
        for n in self.nodes:
            s = st[n.name]; ok = s >= 0
            if ok.any():
                lj[ok] += np.log(self.cpt[n.name][:, :, s[ok]]).transpose(2, 0, 1)
        return lj

    def predict_proba(self, df):
        lj = self.log_joint(df)
        la = np.logaddexp.reduce(lj, axis=2)
        la -= la.max(1, keepdims=True)
        p = np.exp(la)
        return p / p.sum(1, keepdims=True)

    def sme_perturb(self, rng, sigma=0.5, step=0.05, floor=0.01, overconfidence=1.5):
        """Emulate expert elicitation: noisy log-odds, probabilities rounded to `step` with a floor,
        and sharpened (overconfident) distributions for the strong cues."""
        m = copy.deepcopy(self)
        strong = {n.name for n in self.nodes if n.strong}
        for k, p in m.cpt.items():
            q = np.exp(np.log(p) + rng.normal(0, sigma, p.shape))
            q /= q.sum(-1, keepdims=True)
            if k in strong:
                q = q ** overconfidence; q /= q.sum(-1, keepdims=True)
            q = np.maximum(np.round(q / step) * step, floor)
            m.cpt[k] = q / q.sum(-1, keepdims=True)
        pc = np.maximum(np.round(m.p_c / step) * step, floor)
        m.p_c = pc / pc.sum(1, keepdims=True)
        return m

    def to_pgmpy(self):
        """Export as a pgmpy network (A -> C, observation nodes under A / C / both)."""
        try:
            from pgmpy.models import DiscreteBayesianNetwork as BNModel
        except ImportError:
            from pgmpy.models import BayesianNetwork as BNModel
        from pgmpy.factors.discrete import TabularCPD
        nC = len(self.classes)
        edges = [("A", "C")]
        cpds = [TabularCPD("A", 3, self.prior.reshape(3, 1)),
                TabularCPD("C", nC, self.p_c.T, evidence=["A"], evidence_card=[3])]
        for n in self.nodes:
            p = self.cpt[n.name]                                  # (3, nC, S)
            if n.parents == "A":
                edges.append(("A", n.name)); cpds.append(TabularCPD(n.name, n.n_states, p[:, 0, :].T, ["A"], [3]))
            elif n.parents == "C":
                edges.append(("C", n.name)); cpds.append(TabularCPD(n.name, n.n_states, p[0].T, ["C"], [nC]))
            else:
                edges += [("A", n.name), ("C", n.name)]
                cpds.append(TabularCPD(n.name, n.n_states, p.reshape(3 * nC, -1).T, ["A", "C"], [3, nC]))
        model = BNModel(edges)
        model.add_cpds(*cpds)
        assert model.check_model()
        return model

    def n_parameters(self):
        return int(sum(p.size for p in self.cpt.values()) + self.p_c.size + 3)


# ----------------------------------------------------------------------------- tiered wrapper
class TieredBN:
    """One BN per tier; each row is routed by its observable tier."""

    def __init__(self, context=False, alpha=1.0):
        self.models = {t: TierBN(t, context, alpha) for t in (1, 2, 3)}   # 3 = hypersonic; prior-only if no data

    def fit(self, df):
        tier = observable_tier(df)
        for t, m in self.models.items():
            m.fit(df[tier == t])
        return self

    def predict_proba(self, df):
        tier = observable_tier(df)
        p = np.zeros((len(df), 3))
        for t, m in self.models.items():
            idx = np.flatnonzero(tier == t)
            if len(idx):
                p[idx] = m.predict_proba(df.iloc[idx])
        return p

    def sme(self, seed=0, **kw):
        rng = np.random.default_rng(seed)
        out = copy.deepcopy(self)
        out.models = {t: m.sme_perturb(rng, **kw) for t, m in self.models.items()}
        return out
