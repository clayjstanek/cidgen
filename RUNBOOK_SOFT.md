# Runbook: JPDA and VB trackers (S12)

Unzip `cidgen_soft_trackers.zip` into `cidgen_v0.6.1_hypersonic_GNN` (the folder that holds `cidgen`),
replacing files when asked. Background, and the bench results so far: `TRACKERS.md`, section
"JPDA and the variational (VB) tracker".

Run times on a 2-core cloud box, per S12 run: GNN about 1 min, JPDA about 1.5 min, VB about 20 min.
VB is pure Python and re-solves every contested window at every grid time; your machine should be faster.

## V0. Check (once)

| Step | Command | Expect |
| --- | --- | --- |
| V0.1 | `python -m pytest tests\test_soft_trackers.py tests\test_gnn_tracker.py -q` | `6 passed` |

## V1. Tracker bench: the trackers on their own, where the answer is known (about 10 min)

| Step | Command | Needs |
| --- | --- | --- |
| V1.1 | `python tracker_bench.py --runs 20 --out results_bench\bench.csv` | V0 |

## V2. Tracker comparison on S12 test seeds 18-20, perfect identity (no training needed)

The threat-list comparison with perfect identity isolates the tracker. Only the three test seeds are needed.
Run each line in its own window; they are independent.

| Step | Command | Needs |
| --- | --- | --- |
| V2.1 | `python -m cidgen.generate --scenario S12_jpda --seed 18 --seeds 3 --out runs_jpda` | V0 |
| V2.2 | `python -m cidgen.generate --scenario S12_vb --seed 18 --out runs_vb` | V0 |
| V2.3 | `python -m cidgen.generate --scenario S12_vb --seed 19 --out runs_vb` | V0 |
| V2.4 | `python -m cidgen.generate --scenario S12_vb --seed 20 --out runs_vb` | V0 |
| V2.5 | `python track_report.py --runs "runs_jpda/S12_jpda_seed*" --out results_jpda` | V2.1 |
| V2.6 | `python track_report.py --runs "runs_vb/S12_vb_seed*" --out results_vb` | V2.2-V2.4 |
| V2.7 | `python -m cidgen.threat.run_tp --runs "runs_jpda/S12_jpda_seed*" --name S12_jpda --identity truth --out results_tp_jpda` | V2.1 |
| V2.8 | `python -m cidgen.threat.run_tp --runs "runs_vb/S12_vb_seed*" --name S12_vb --identity truth --out results_tp_vb` | V2.2-V2.4 |

The label-mode and GNN rows with perfect identity are already in `results_tp_label_all` and `results_tp_gnn`.

## What to send back

- `results_bench\bench_summary.csv`
- `results_jpda\tracker_runs.csv`, `results_jpda\tracker_tiers.csv`, `results_vb\tracker_runs.csv`, `results_vb\tracker_tiers.csv`
- `results_tp_jpda\tp_*.csv`, `results_tp_vb\tp_*.csv`

## Later, only if V2 shows a tracker worth the full chain

Generate all 20 seeds for that tracker and run G3.1-G3.6 of `RUNBOOK_GNN.md` with `S12_gnn` replaced by
`S12_jpda` or `S12_vb` and the folders renamed (`runs_vb`, `results_vb`, `cache_vb`, ...).
