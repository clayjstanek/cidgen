"""Ensemble plumbing test, CPU only, no torch needed (about 3 minutes).

    python tests/test_ensemble.py

Generates four short S0 runs and one short S9 run, runs compare --save-preds in shift mode, writes
stand-in neural predictions (noisy copies of the labels) as valpred_/pred_ files, and runs the
ensemble evaluation. Checks that every output file is written and the combinations are valid
probabilities. The numbers themselves mean nothing here.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fake_neural(frame, path, seed, skill=2.5):
    rng = np.random.default_rng(seed)
    logit = rng.normal(0, 1, (len(frame), 3))
    logit[np.arange(len(frame)), frame.allegiance.values] += skill
    p = np.exp(logit); p /= p.sum(1, keepdims=True)
    pd.DataFrame(dict(run=frame.run.values, eid=frame.eid.values, t=frame.t.values,
                      p_Friend=p[:, 0], p_Neutral=p[:, 1], p_Hostile=p[:, 2])).to_csv(path, index=False)


def main():
    from cidgen.generate import run_scenario
    from cidgen.models.compare import compare
    from cidgen.models.ensemble import run
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); res = td / "results"; res.mkdir()
        short = {"run": {"horizon_s": 300}}
        tr = [str(run_scenario("S0", s, td / "runs", short, verbose=False)) for s in (1, 2, 3, 4)]
        te = [str(run_scenario("S9", 1, td / "runs", short, verbose=False))]
        out = compare(tr, res, res, test_runs=te, B=200, save_preds=True)
        name = next(iter(out))
        allp = next(res.glob(f"allpred_{name}.*"))
        d = pd.read_parquet(allp) if allp.suffix == ".parquet" else pd.read_csv(allp)
        assert set(d.split) == {"val", "test"}, set(d.split)
        assert {"oor_tq", "oor_count", "obs_tier", "BN-learned|Friend"} <= set(d.columns)
        for s, prefix in (("val", "valpred"), ("test", "pred")):
            fake_neural(d[d.split == s], res / f"{prefix}_gnn-notq_{name}.csv", seed=1)
        t, ci, diff, unk = run(res, name, B=200)
        for kind in ("Mean", "LogPool", "Stack"):
            assert any(k.startswith(kind) for k in t.index), kind
        assert "Gate[tier]" in t.index and "NN|TREE if OOR" in t.index
        assert t.bal_acc.between(0, 1).all() and len(unk)
        for suffix in ("", "_ci", "_diffs", "_by_tier", "_by_role", "_unknown", "_stack_weights", "_val"):
            assert (res / f"ensemble_{name}_gnn-notq{suffix}.csv").exists(), suffix
    print("ENSEMBLE TEST PASSED")


if __name__ == "__main__":
    main()
