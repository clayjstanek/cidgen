"""GPU smoke test: run this first on the target machine.

    python tests/smoke_gpu.py

Checks: torch + CUDA, torch_geometric import, a short S0 scenario generated end to end,
a HeteroData snapshot moved to the GPU, and one forward/backward pass of a heterogeneous
GATv2 (track-track) + GraphSAGE (asset/ESM -> track) model.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import torch
    import torch_geometric
    from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv

    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"pyg {torch_geometric.__version__}")
    assert torch.cuda.is_available(), "CUDA not available to torch"
    dev = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    from cidgen.generate import run_scenario
    from cidgen.graph_export import RunGraphs, to_heterodata
    from cidgen.io import parquet_available
    print("parquet available:", parquet_available())

    with tempfile.TemporaryDirectory() as td:
        t0 = time.time()
        run = run_scenario("S0", seed=11, out_root=td, overrides={"run": {"horizon_s": 300}}, verbose=False)
        print(f"generated 5-min S0 in {time.time() - t0:.1f}s -> {run.name}")
        rg = RunGraphs(run)
        snap = rg.snapshot(float(rg.times[len(rg.times) // 2]))
        data = to_heterodata(snap, dev)
        print(data)

        hid = 64
        conv = HeteroConv({
            ("track", "near", "track"): GATv2Conv((-1, -1), hid, heads=2, concat=False, edge_dim=10,
                                                  add_self_loops=True, fill_value="mean"),
            ("asset", "threatened_by", "track"): SAGEConv((-1, -1), hid),
            ("esm_fix", "candidate", "track"): SAGEConv((-1, -1), hid),
            ("track", "threatens", "asset"): SAGEConv((-1, -1), hid),
            ("track", "hears", "esm_fix"): SAGEConv((-1, -1), hid),
        }, aggr="sum").to(dev)
        head = torch.nn.Linear(hid, 3).to(dev)
        edge_attr = {k: data[k].edge_attr for k in data.edge_types if k == ("track", "near", "track")}
        out = conv(data.x_dict, data.edge_index_dict, edge_attr_dict=edge_attr)
        logits = head(torch.relu(out["track"]))
        mask = data["track"].y >= 0                       # birds are not objects of interest
        loss = torch.nn.functional.cross_entropy(logits[mask], data["track"].y[mask])
        loss.backward()
        print(f"forward/backward OK on {dev}: logits {tuple(logits.shape)}, loss {loss.item():.3f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
