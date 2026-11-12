# Runbook: merged reports (vbm) and protecting resolved tracks (S12)

Unzip `cidgen_merge_trackers.zip` into `cidgen_v0.6.1_hypersonic_GNN` (the folder that holds `cidgen`),
replacing files when asked. Results and explanation: `TRACKERS.md`, sections "Merged reports in the
tracker: vbm" and "Protecting resolved tracks, on S12".

Nothing you have run before changes: protection is off unless a scenario turns it on, and the GNN, JPDA
and VB scenarios you already have give the same output as before.

## M0. Check

| Step | Command | Expect |
| --- | --- | --- |
| M0.1 | `python -m pytest tests\test_soft_trackers.py tests\test_gnn_tracker.py -q` | `8 passed` |

## M1. GNN + protection on the S12 test seeds (about 1 min per run)

| Step | Command | Needs |
| --- | --- | --- |
| M1.1 | `python -m cidgen.generate --scenario S12_gnnp --seed 18 --seeds 3 --out runs_gnnp` | M0 |
| M1.2 | `python track_report.py --runs "runs_gnnp/S12_gnnp_seed*" --out results_gnnp` | M1.1 |
| M1.3 | `python -m cidgen.threat.run_tp --runs "runs_gnnp/S12_gnnp_seed*" --name S12_gnnp --identity truth --out results_tp_gnnp` | M1.1 |

## M2. Optional, slow: VB + protection (about 45 min per run here)

Run each seed in its own window.

| Step | Command | Needs |
| --- | --- | --- |
| M2.1 | `python -m cidgen.generate --scenario S12_vbp --seed 18 --out runs_vbp` (and again with `--seed 19`, `--seed 20`) | M0 |
| M2.2 | `python track_report.py --runs "runs_vbp/S12_vbp_seed*" --out results_vbp` | M2.1 |
| M2.3 | `python -m cidgen.threat.run_tp --runs "runs_vbp/S12_vbp_seed*" --name S12_vbp --identity truth --out results_tp_vbp` | M2.1 |

## Send back

`results_gnnp\tracker_runs.csv`, `results_tp_gnnp\tp_S12_gnnp__doctrine_default.csv`, and the same two
from `results_vbp` and `results_tp_vbp` if you ran M2.
