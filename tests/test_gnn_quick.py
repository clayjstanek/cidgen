"""Quick end-to-end GNN test (about a minute on a GPU). Run before a full training job.

    python tests/test_gnn_quick.py

Generates three 5-minute S0 runs, builds caches, trains the GNN, the GRU-only ablation and the GNN
with sensor dropout for two epochs each, calibrates, writes predictions, and scores them with compare.py.
Then reloads each saved model with models/predict.py (checking it reproduces the test predictions),
saves BN and tree predictions with compare --save-preds, and runs the ensemble evaluation.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from cidgen.generate import run_scenario
    from cidgen.models.compare import compare
    from cidgen.models.ensemble import run as ensemble
    from cidgen.models.predict import recover
    from cidgen.models.train_gnn import run_one

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        runs = [str(run_scenario("S0", s, td / "runs", {"run": {"horizon_s": 300}}, verbose=False)) for s in (1, 2, 3)]
        print("generated", [Path(r).name for r in runs])
        for no_graph, sd in ((False, 0.0), (True, 0.0), (False, 0.3)):     # GNN, GRU, GNN with sensor dropout
            a = SimpleNamespace(cache=str(td / "cache"), val_frac=0.34, test_frac=0.34, window=15, device="cuda",
                                seed=0, batch=8, workers=0, hidden=64, layers=2, no_graph=no_graph, epochs=2, sensor_dropout=sd, tag=None, drop_features=None,
                                lr=1e-3, patience=5)
            (td / "results").mkdir(exist_ok=True)
            print(run_one("S0_baseline", runs, a, td / "results", print))
        for pt in sorted((td / "results").glob("*.pt")):
            r = recover(pt, td / "cache", td / "results", device="cuda", check_test=True)
            assert r.get("check_ok"), f"{pt.name} did not reproduce its test predictions"
        compare(runs, td / "results", td / "results", B=200, save_preds=True)
        ensemble(td / "results", "S0_baseline", B=200)
    print("GNN QUICK TEST PASSED")


if __name__ == "__main__":
    main()
