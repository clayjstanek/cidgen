"""Cross-check the numpy BN inference against pgmpy's exact VariableElimination.

    pip install pgmpy
    python tests/test_bn_pgmpy.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from pgmpy.inference import VariableElimination

    from cidgen.generate import run_scenario
    from cidgen.models.bn import TieredBN, observable_tier
    from cidgen.models.common import eval_frame

    with tempfile.TemporaryDirectory() as td:
        run = run_scenario("S0", 3, td, {"run": {"horizon_s": 600}}, verbose=False)
        df = eval_frame(run)
    bn = TieredBN().fit(df)
    tier = observable_tier(df)
    rng = np.random.default_rng(0)
    worst = 0.0
    for t, m in bn.models.items():
        model = m.to_pgmpy()
        ve = VariableElimination(model)
        rows = df[tier == t]
        if not len(rows):
            continue
        rows = rows.iloc[rng.choice(len(rows), size=min(25, len(rows)), replace=False)]
        p_np = m.predict_proba(rows)
        st = m._states(rows)
        for k in range(len(rows)):
            ev = {n.name: int(st[n.name][k]) for n in m.nodes if st[n.name][k] >= 0}
            q = ve.query(["A"], evidence=ev, show_progress=False).values
            worst = max(worst, float(np.abs(q - p_np[k]).max()))
        print(f"tier {t}: {m.n_parameters()} parameters, {len(rows)} rows checked")
    print(f"max |pgmpy - numpy| = {worst:.2e}")
    assert worst < 1e-6, "numpy BN inference disagrees with pgmpy"
    print("BN PGMPY CROSS-CHECK PASSED")


if __name__ == "__main__":
    main()
