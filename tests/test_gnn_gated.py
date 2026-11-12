"""Quick end-to-end test of the graph options (about a minute on a GPU). Run after test_gnn_quick.py.

    python tests/test_gnn_gated.py

Generates three 5-minute S0 runs and trains, for two epochs each:
  1. the GNN with co-movement edge features, the 0.5 km Tier 1 gate and gated edges (--comove-features
     --tier1-gate-km 0.5 --edge-gate), and
  2. the plain GNN, to confirm the default path still works next to the new cache files.
Checks the option caches are separate files with 16 edge features, the gates are valid and not all
identical, every saved model reloads and reproduces its test predictions, and compare.py scores them.
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader

    from cidgen.generate import run_scenario
    from cidgen.models.compare import compare
    from cidgen.models.gnn import CIDGNN, GatedEdgeConv, SnapshotDataset
    from cidgen.models.gnn_data import build_run_cache, fit_normalizer, load_cache
    from cidgen.models.predict import recover
    from cidgen.models.train_gnn import run_one

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        runs = [str(run_scenario("S0", s, td / "runs", {"run": {"horizon_s": 300}}, verbose=False)) for s in (1, 2, 3)]
        print("generated", [Path(r).name for r in runs])
        (td / "results").mkdir()
        base = dict(cache=str(td / "cache"), val_frac=0.34, test_frac=0.34, window=15, device="cuda", seed=0, batch=8,
                    workers=0, hidden=64, layers=2, no_graph=False, epochs=2, sensor_dropout=0.1, tag=None,
                    drop_features="track-quality", lr=1e-3, patience=5)
        r1 = run_one("S0_baseline", runs, SimpleNamespace(**base, comove_features=True, tier1_gate_km=0.5, edge_gate=True),
                     td / "results", print)
        r2 = run_one("S0_baseline", runs, SimpleNamespace(**base), td / "results", print)
        assert r1["model"] == "gnn-sd10-notq-cm-t1g500m-eg" and r2["model"] == "gnn-sd10-notq", (r1["model"], r2["model"])

        names = sorted(p.name for p in (td / "cache").iterdir())
        assert any("__cm-t1g500m" in n for n in names) and any(n.endswith("seed1.pkl") for n in names), names
        blob = load_cache(build_run_cache(runs[0], td / "cache", comove=True, tier1_gate_km=0.5))
        widths = {np.asarray(s["edges"][("track", "near", "track")][1]).shape[1] for s in blob["snaps"]
                  if len(s["edges"][("track", "near", "track")][1])}
        assert widths == {16}, widths
        print("option caches: separate files, 16 track-track edge features")

        # gates: run the trained gated model on one batch and look at them
        ck = torch.load(td / "results" / f"{r1['model']}_S0_baseline.pt", map_location="cpu", weights_only=False)
        model = CIDGNN(ck["n_feat"], n_fix=ck["n_fix"], hidden=64, layers=2, tt_edge_dim=ck["tt_edge_dim"], tt_conv="gated")
        model.load_state_dict(ck["state"]); model.eval()
        ds = SnapshotDataset([blob], ck["norm"], 15, drop_features=["pos_sigma_m", "vel_sigma_mps", "age_s"])
        with torch.no_grad():
            model(next(iter(DataLoader(ds, batch_size=8))))
        gated = [m for m in model.modules() if isinstance(m, GatedEdgeConv)]
        assert len(gated) == 2 and all(m.last_gate is not None for m in gated), "gated layers not used"
        g = torch.cat([m.last_gate.flatten() for m in gated])
        assert torch.isfinite(g).all() and (g >= 0).all() and (g <= 1).all() and g.std() > 0, g
        print(f"gates: {len(g)} edges, mean {g.mean():.3f}, min {g.min():.3f}, max {g.max():.3f}")

        for pt in sorted((td / "results").glob("*.pt")):
            r = recover(pt, td / "cache", td / "results", device="cuda", check_test=True)
            assert r.get("check_ok"), f"{pt.name} did not reproduce its test predictions"
        compare(runs, td / "results", td / "results", B=200)
    print("GATED GNN QUICK TEST PASSED")


if __name__ == "__main__":
    main()
