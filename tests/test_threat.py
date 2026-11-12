"""Threat prioritisation test, CPU only (about 2 minutes).

    python tests/test_threat.py

Generates four short S0 runs and one short S9 run, fits the BN and tree models with compare --save-preds,
then ranks every snapshot of the S9 run with truth identity, BN-learned and a BN + tree stack, under the
three rankers (P(Hostile) only, expected threat, doctrine). Checks the outputs and that prioritize()
leaves its input unaltered.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
DOCTRINE = Path(__file__).resolve().parent.parent / "configs" / "threat" / "doctrine_default.yaml"


def main():
    from cidgen.generate import run_scenario
    from cidgen.models.compare import compare
    from cidgen.threat.picture import build_frames, identity, pictures
    from cidgen.threat.prioritize import ThreatPrioritizer
    from cidgen.threat.run_tp import run
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); res = td / "results"; res.mkdir()
        short = {"run": {"horizon_s": 300}}
        tr = [str(run_scenario("S0", s, td / "runs", short, verbose=False)) for s in (1, 2, 3, 4)]
        te = [str(run_scenario("S9", 1, td / "runs", short, verbose=False))]
        name = next(iter(compare(tr, res, res, test_runs=te, B=100, save_preds=True, log=lambda m: None)))
        ap = next(res.glob(f"allpred_{name}.*"))
        cols = pd.read_parquet(ap).columns if ap.suffix == ".parquet" else pd.read_csv(ap, nrows=1).columns
        tree = next(c.split("|")[0] for c in cols if c.endswith("-own|Friend"))
        # purity: prioritize must not modify the caller's picture
        doc = yaml.safe_load(open(DOCTRINE))
        frame = build_frames(te, doc)
        _, t, pic = next(pictures(frame, identity(frame, "BN-learned", res, name)))
        before = pic.copy(deep=True)
        out = ThreatPrioritizer(doc).prioritize(pic, t)
        pd.testing.assert_frame_equal(pic, before)
        assert list(out["rank"]) == list(range(1, len(out) + 1)) and set(out.tier) <= {"engage", "monitor", "protected"}
        tab = run(te, res, name, ["truth", "BN-learned", f"STACK:BN-learned+{tree}"], [DOCTRINE], res, dump_run="seed1")
        tab = next(iter(tab.values()))
        assert len(tab) == 9 and tab.ndcg10.between(0, 1).all(), tab
        assert next(res.glob("tp_*__lists_seed1.csv")).stat().st_size > 0
        print(tab[["identity", "ranker", "ndcg10", "top1_in_top3", "friend_above_threat"]].round(3).to_string(index=False))
    print("THREAT TEST PASSED")


if __name__ == "__main__":
    main()
