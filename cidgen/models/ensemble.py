"""Combine model outputs and score the combinations exactly as compare.py scores single models.

    python -m cidgen.models.ensemble --dir results_transfer --name S0_baseline__on__S9_sensor_shift
    python -m cidgen.models.ensemble --dir results_transfer --name S0_baseline__on__S9_sensor_shift --neural GRU-NOTQ

Needs, in --dir:
  allpred_<name>.parquet or .csv     BN and tree predictions on validation + test rows (compare --save-preds)
  valpred_<tag>_<name>.csv           neural predictions on validation rows (train_gnn v0.5+, or models/predict.py)
  pred_<tag>_<name>.csv              neural predictions on test rows (train_gnn)

Every combination is fitted or chosen on the VALIDATION runs (training scenario) only, then frozen and
scored on the test runs. Nothing about the test condition is used.

Combinations (BN, TREE and NN are the members chosen with --bn, --tree, --neural)
  Mean[...]          average of the members' probabilities (linear opinion pool)
  LogPool[...]       normalised geometric mean (logarithmic opinion pool); one confident "no" vetoes
  Stack[...]         multinomial logistic regression on the members' log probabilities, fitted on validation
  Gate[tier]         per tier (read from speed, altitude, RCS), the member with the lowest validation log loss
  NN|TREE if OOR     the neural model, replaced by the tree model on rows whose track-quality inputs are
                     beyond the training range (the S9 drone failure mode)

Also writes an Unknown table: the five-state CID idea that disagreement between members, or low
combined confidence, should give Unknown / Pending rather than a forced Friend / Neutral / Hostile call.

Outputs ensemble_<name>_<NN>.csv, _ci, _diffs, _by_tier, _by_role, _own_drones_eoir, _unknown,
_stack_weights, _val (member scores on validation, which drive the gate).
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .common import bootstrap, metrics, metrics_table
from .compare import breakdowns

CLS = ["Friend", "Neutral", "Hostile"]
EPS = 1e-6
NEURAL_PREF = ["GNN-NOTQ", "GNN-SD30-NOTQ", "GNN-SD30", "GNN", "GRU-NOTQ", "GRU"]
TREE_PREF = ["XGB-own", "HGB-own", "XGB-own+nb", "HGB-own+nb"]


# ----------------------------------------------------------------------------- loading
def load(dir_, name, log=print):
    """Returns (frames {val, test}, probs {member: {val, test}}) on the rows every member covers."""
    dir_ = Path(dir_)
    f = dir_ / f"allpred_{name}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
    elif f.with_suffix(".csv").exists():
        d = pd.read_csv(f.with_suffix(".csv"), low_memory=False)
    else:
        raise SystemExit(f"{f.with_suffix('.*')} not found: run compare with --save-preds first")
    members = sorted({c.split("|")[0] for c in d.columns if "|" in c})
    fr = {s: d[d.split == s].reset_index(drop=True) for s in ("val", "test")}
    probs = {m: {s: fr[s][[f"{m}|{c}" for c in CLS]].values.astype(float) for s in fr} for m in members}
    for f in sorted(glob.glob(str(dir_ / f"valpred_*_{name}.csv"))):
        tag = Path(f).stem[len("valpred_"):].split("_")[0].upper()
        tf = dir_ / ("pred_" + Path(f).name[len("valpred_"):])
        if not tf.exists():
            log(f"note: {Path(f).name} has no matching test file {tf.name}; {tag} skipped")
            continue
        probs[tag] = {}
        for s, path in (("val", f), ("test", tf)):
            pr = pd.read_csv(path)
            m = fr[s][["run", "eid", "t"]].merge(pr, on=["run", "eid", "t"], how="left")
            probs[tag][s] = m[["p_Friend", "p_Neutral", "p_Hostile"]].values.astype(float)
    for f in sorted(glob.glob(str(dir_ / f"pred_*_{name}.csv"))):
        tag = Path(f).stem[len("pred_"):].split("_")[0].upper()
        if tag not in probs:
            log(f"note: {tag} has test predictions but no validation predictions (valpred_{tag.lower()}_{name}.csv);"
                f" run models/predict.py on its .pt file to include it")
    return fr, probs


def common_rows(fr, probs, members, log=print):
    out_fr, out_p = {}, {m: {} for m in members}
    for s in fr:
        ok = np.ones(len(fr[s]), bool)
        for m in members:
            ok &= np.isfinite(probs[m][s]).all(1)
        if ok.mean() < 0.99:
            short = [m for m in members if np.isfinite(probs[m][s]).all(1).mean() < 0.99]
            log(f"note: {s}: {ok.mean():.1%} of rows covered by every member (short: {', '.join(short)}); "
                f"scoring those rows only")
        out_fr[s] = fr[s][ok].reset_index(drop=True)
        for m in members:
            out_p[m][s] = probs[m][s][ok]
    return out_fr, out_p


# ----------------------------------------------------------------------------- combiners
def norm(p):
    p = np.clip(p, EPS, None)
    return p / p.sum(1, keepdims=True)


def mean_pool(ps):
    return norm(np.mean([norm(p) for p in ps], 0))


def log_pool(ps):
    lp = np.mean([np.log(norm(p)) for p in ps], 0)
    lp -= lp.max(1, keepdims=True)
    return norm(np.exp(lp))


class Stacker:
    """Multinomial logistic regression on log probabilities; balanced class weights like the base models."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, ps, y):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000)
        self.lr.fit(self._X(ps), y)
        return self

    @staticmethod
    def _X(ps):
        return np.hstack([np.log(norm(p)) for p in ps])

    def predict(self, ps):
        q = np.zeros((len(ps[0]), 3)); q[:, self.lr.classes_] = self.lr.predict_proba(self._X(ps))
        return norm(q)

    def weights(self, names):
        """Share of the absolute coefficient mass on each member: roughly how much each one counts."""
        a = np.abs(self.lr.coef_).reshape(len(self.lr.classes_), len(names), 3).sum((0, 2))
        return dict(zip(names, a / a.sum()))


def log_loss(p, y):
    return float(-np.log(norm(p)[np.arange(len(y)), y]).mean())


def tier_gate(P, members, fr):
    """Per observable tier, the member with the lowest validation log loss."""
    yv, tv, tt = fr["val"].allegiance.values, fr["val"].obs_tier.values, fr["test"].obs_tier.values
    out = np.zeros((len(fr["test"]), 3)); choice = {}
    for t in np.unique(np.concatenate([tv, tt])):
        mv = tv == t
        best = (min(members, key=lambda m: log_loss(P[m]["val"][mv], yv[mv])) if mv.any()
                else min(members, key=lambda m: log_loss(P[m]["val"], yv)))
        choice[int(t)] = best
        out[tt == t] = norm(P[best]["test"][tt == t])
    return out, choice


# ----------------------------------------------------------------------------- Unknown / Pending
def unknown_table(fr, P, combos):
    """combos: {label: (member list, combined test probabilities)}. For each, several rules for
    declaring Unknown, and what the operator would see: decisions, fratricide and missed threats."""
    d = fr["test"]; y = d.allegiance.values
    fr_m, ho_m = y == 0, y == 2
    ou = fr_m & (d.role.values == "own_uas")
    air = (d.cls.values == "E") & (y == 1)
    rows = []
    for label, (mem, p) in combos.items():
        pred = p.argmax(1); conf = p.max(1)
        votes = np.stack([P[m]["test"].argmax(1) for m in mem], 1)
        disagree = (votes != votes[:, :1]).any(1)
        rules = {"never (forced call)": np.zeros(len(y), bool)}
        if len(mem) > 1:
            rules["members disagree"] = disagree
            rules["members disagree on Hostile"] = (votes == 2).any(1) & ~(votes == 2).all(1)
        for tau in (0.6, 0.8, 0.9):
            rules[f"confidence < {tau}"] = conf < tau
        for rule, unk in rules.items():
            dec = ~unk
            hostile_call = dec & (pred == 2)
            rec = [((pred == k) & dec & (y == k)).sum() / max((y == k).sum(), 1) for k in range(3)]
            rows.append(dict(
                combination=label, members="+".join(mem), unknown_rule=rule,
                unknown_share=float(unk.mean()),
                decided_accuracy=float((pred[dec] == y[dec]).mean()) if dec.any() else np.nan,
                friend_called_hostile=float(hostile_call[fr_m].mean()) if fr_m.any() else np.nan,
                friend_unknown=float(unk[fr_m].mean()) if fr_m.any() else np.nan,
                own_drone_called_hostile=float(hostile_call[ou].mean()) if ou.any() else np.nan,
                own_drone_unknown=float(unk[ou].mean()) if ou.any() else np.nan,
                airliner_called_hostile=float(hostile_call[air].mean()) if air.any() else np.nan,
                hostile_called_hostile=float(rec[2]),
                hostile_unknown=float(unk[ho_m].mean()) if ho_m.any() else np.nan,
                hostile_called_friend_or_neutral=float((dec & (pred != 2))[ho_m].mean()) if ho_m.any() else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def pick(pref, avail, given, what):
    if given:
        if given not in avail:
            raise SystemExit(f"--{what} {given}: not available; have {', '.join(sorted(avail))}")
        return given
    for m in pref:
        if m in avail:
            return m
    return None


def run(dir_, name, out_dir=None, bn=None, tree=None, neural=None, B=2000, log=print):
    out_dir = Path(out_dir or dir_); out_dir.mkdir(parents=True, exist_ok=True)
    fr, probs = load(dir_, name, log)
    avail = set(probs)
    BN = pick(["BN-learned"], avail, bn, "bn")
    TREE = pick(TREE_PREF, avail, tree, "tree")
    NN = pick(NEURAL_PREF, avail, neural, "neural")
    if NN is None:
        log("warning: no neural model has both validation and test predictions; ensembles use BN and trees only")
    alias = {k: v for k, v in (("BN", BN), ("TREE", TREE), ("NN", NN)) if v}
    log("members: " + ", ".join(f"{k} = {v}" for k, v in alias.items()) +
        f" | all with validation + test predictions: {', '.join(sorted(avail))}")
    members = sorted(avail)
    fr, P = common_rows(fr, probs, members, log)
    yv = fr["val"].allegiance.values
    log(f"validation: {len(fr['val']):,} track-snapshots in {fr['val'].run.nunique()} runs; "
        f"test: {len(fr['test']):,} in {fr['test'].run.nunique()} runs")

    short = lambda ms: "+".join(next(k for k, v in alias.items() if v == m) if m in alias.values() else m for m in ms)
    sets = []
    for ks in (("BN", "NN"), ("TREE", "NN"), ("BN", "TREE"), ("BN", "TREE", "NN")):
        ms = [alias[k] for k in ks if k in alias]
        if len(ms) == len(ks):
            sets.append(ms)
    if len(members) > 3:
        sets.append(members)

    te = fr["test"]
    res = {m: (norm(P[m]["test"]), te) for m in members}
    combos, wrows = {}, []
    for ms in sets:
        lab = short(ms) if ms is not members else "all"
        tp = [P[m]["test"] for m in ms]
        st = Stacker().fit([P[m]["val"] for m in ms], yv)
        for kind, p in (("Mean", mean_pool(tp)), ("LogPool", log_pool(tp)), ("Stack", st.predict(tp))):
            key = f"{kind}[{lab}]"
            res[key] = (p, te); combos[key] = (ms, p)
        for m, w in st.weights(ms).items():
            wrows.append(dict(combination=f"Stack[{lab}]", member=m, weight_share=round(float(w), 3)))
    gp, choice = tier_gate(P, members, fr)
    res["Gate[tier]"] = (gp, te); combos["Gate[tier]"] = (members, gp)
    log("Gate[tier] choices (lowest validation log loss): " + ", ".join(f"tier {t}: {m}" for t, m in choice.items()))
    if NN and TREE:
        oor = te.oor_tq.astype(bool).values
        fb = np.where(oor[:, None], norm(P[TREE]["test"]), norm(P[NN]["test"]))
        res["NN|TREE if OOR"] = (fb, te); combos["NN|TREE if OOR"] = ([NN, TREE], fb)
        if BN:
            sk = f"Stack[{short([BN, NN])}]"
            if sk in res:
                fb2 = np.where(oor[:, None], norm(P[TREE]["test"]), res[sk][0])
                res[f"{sk}|TREE if OOR"] = (fb2, te); combos[f"{sk}|TREE if OOR"] = ([BN, NN, TREE], fb2)
        log(f"out of training range (track quality): validation {fr['val'].oor_tq.astype(bool).mean():.1%}, "
            f"test {oor.mean():.1%} of track-snapshots")

    t = metrics_table(res)
    refs = tuple(m for m in (BN, TREE, NN) if m)
    ci, diff = bootstrap(res, B=B, refs=refs)
    bt, br, be = breakdowns(res)
    unk = unknown_table(fr, P, {**{m: ([m], norm(P[m]["test"])) for m in (BN, TREE, NN) if m}, **combos})
    val = metrics_table({m: (norm(P[m]["val"]), fr["val"]) for m in members})

    tag = f"{name}_{(NN or 'none').lower()}"
    t.to_csv(out_dir / f"ensemble_{tag}.csv")
    ci.round(4).to_csv(out_dir / f"ensemble_{tag}_ci.csv")
    diff.round(4).to_csv(out_dir / f"ensemble_{tag}_diffs.csv", index=False)
    bt.round(4).to_csv(out_dir / f"ensemble_{tag}_by_tier.csv", index=False)
    br.round(4).to_csv(out_dir / f"ensemble_{tag}_by_role.csv")
    if len(be):
        be.round(4).to_csv(out_dir / f"ensemble_{tag}_own_drones_eoir.csv")
    unk.round(4).to_csv(out_dir / f"ensemble_{tag}_unknown.csv", index=False)
    pd.DataFrame(wrows).to_csv(out_dir / f"ensemble_{tag}_stack_weights.csv", index=False)
    val.to_csv(out_dir / f"ensemble_{tag}_val.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40):
        log(f"\n=== ensembles, {name} (fitted on validation runs, scored on test runs) ===")
        log(t[["n", "bal_acc", "log_loss", "brier", "ece", "recall_friend", "recall_hostile",
               "airliner_called_hostile", "friend_called_hostile"]].to_string())
        if len(be):
            log("\nOwn drones called Hostile:")
            log(be[["called_hostile_all"]].round(3).T.to_string())
        if wrows:
            log("\nStacking: share of weight on each member")
            log(pd.DataFrame(wrows).pivot(index="combination", columns="member", values="weight_share").to_string())
        for ref in refs:
            k = diff[(diff.vs == ref) & ~diff.model.isin(members)]
            if len(k):
                log(f"\nCombinations vs {ref} (paired, resampling whole aircraft):")
                log(k.pivot(index="model", columns="metric", values="verdict").to_string())
        log("\nUnknown / Pending: forced call vs disagreement rules (confidence rules are in the _unknown file):")
        show = unk[unk.unknown_rule.isin(["never (forced call)", "members disagree", "members disagree on Hostile"])]
        log(show[["combination", "unknown_rule", "unknown_share", "friend_called_hostile", "own_drone_called_hostile",
                  "hostile_called_hostile", "hostile_unknown"]].round(3).to_string(index=False))
    return t, ci, diff, unk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder with allpred_, valpred_ and pred_ files")
    ap.add_argument("--name", required=True, help="e.g. S0_baseline__on__S9_sensor_shift")
    ap.add_argument("--out", default=None, help="output folder (default: --dir)")
    ap.add_argument("--bn", default=None, help="BN member (default BN-learned)")
    ap.add_argument("--tree", default=None, help="tree member (default XGB-own)")
    ap.add_argument("--neural", default=None, help=f"neural member (default: first available of {', '.join(NEURAL_PREF)})")
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    run(a.dir, a.name, a.out, a.bn, a.tree, a.neural, a.bootstrap)


if __name__ == "__main__":
    main()
