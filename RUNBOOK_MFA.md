# Runbook: Stone Soup MFA tracker (S12)

Prerequisites in your conda environment: `stonesoup` (1.9.1 tested by the probe) and `ortools`.
Unzip `cidgen_mfa_tracker.zip` into `cidgen_v0.6.1_hypersonic_GNN` (the folder that holds `cidgen`),
replacing files when asked. Apply it after `cidgen_merge_trackers.zip` and `tracker_timing_update.zip`.

How it works: `cidgen/trackers/mfa_ss.py` (docstring). cidgen keeps its own track management (as for
JPDA and VB); Stone Soup's MFADataAssociator does association and update for confirmed tracks, one scan =
one radar look, slide window 3 looks, per-report noise R, per-track q by tier.

This adapter was written against Stone Soup's documented API but could not be run in my workspace
(Stone Soup cannot be installed there), so M0 is the first real test. If anything fails, send the full
output.

## M0. Tests (about 1 min)

| Step | Command | Expect |
| --- | --- | --- |
| M0.1 | `python -m pytest tests\test_mfa_stonesoup.py -q` | `3 passed` |
| M0.2 | `python -m pytest tests\test_soft_trackers.py tests\test_gnn_tracker.py -q` | `8 passed` |

## M1. Bench: MFA against the others where the answer is known

| Step | Command | Time |
| --- | --- | --- |
| M1.1 | `python tracker_bench.py --runs 10 --methods gnn jpda vb mfa --out results_bench\bench_mfa.csv` | crossings and formations, ~10-30 min |
| M1.2 | `python tracker_bench.py --swarm --protect --runs 4 --methods gnn vb mfa --out results_bench\swarm_mfa.csv` | unresolved swarm, ~10-30 min |

## M2. Timing check on S12 before the full runs (first 300 s, seed 20)

| Step | Command |
| --- | --- |
| M2.1 | `python -m cidgen.trackers.run_tracker --scenario S12_mfa --seed 20 --horizon 60 --debug 1 --out timing` (then 300 if quick) |

The summary shows ms per look (`look` row, p50/p95) and the share in `look/mfa/associate` (Stone Soup).
Multiply the tracker wall time by 30 (60 s) or 6 (300 s) for a full 1,800 s run. The log is written to disk every 5 s; its last line shows the simulated time reached.

## M3. S12 test seeds 18-20, perfect identity (run each seed in its own window)

| Step | Command | Needs |
| --- | --- | --- |
| M3.1 | `python -m cidgen.generate --scenario S12_mfa --seed 18 --out runs_mfa` (and `--seed 19`, `--seed 20`) | M0 |
| M3.2 | `python track_report.py --runs "runs_mfa/S12_mfa_seed*" --out results_mfa` | M3.1 |
| M3.3 | `python -m cidgen.threat.run_tp --runs "runs_mfa/S12_mfa_seed*" --name S12_mfa --identity truth --out results_tp_mfa` | M3.1 |

## Send back

M0 output, `results_bench\bench_mfa_summary.csv`, `results_bench\swarm_mfa_summary.csv`, the M2 summary
(`timing\..._summary.txt`), `results_mfa\tracker_runs.csv`, `results_tp_mfa\tp_S12_mfa__doctrine_default.csv`.

## M2.2 Covariance guard check (after the 2026-11-11 fix)

Full runs printed `RuntimeWarning: invalid value encountered in sqrt` (gnn.py, soft.py, mfa_ss.py): a track's
covariance had stopped being a valid covariance (negative position variance), so its sigma became NaN, its gate
never matched again, and its rows could not be matched to truth. Every covariance that enters or leaves Stone
Soup is now checked and repaired, and the tracker stats count where each repair happened:

| Stat | Meaning |
| --- | --- |
| `covar_fix_predict` / `_worst` | Stone Soup prediction was not a valid covariance (worst min eigenvalue / largest) |
| `covar_fix_update` / `_worst` | Stone Soup Kalman update (P - K S K^T) lost positive-definiteness |
| `covar_fix_mixture` | the moment-matched mixture handed back to cidgen needed repair |
| `covar_fix_input` | a state handed from cidgen to Stone Soup needed repair |
| `covar_nonfinite` | NaN or inf covariance (the last valid state is kept) |
| `ss_negative_dt` / `_worst_s` | Stone Soup asked to predict backwards in time |

| Step | Command | Expect |
| --- | --- | --- |
| M2.2a | `python -m pytest tests\test_psd_repair.py tests\test_mfa_stonesoup.py -q` | `7 passed` |
| M2.2b | `python -m cidgen.trackers.run_tracker --scenario S12_mfa --seed 20 --horizon 300` | no sqrt warnings; send the `tracker stats:` line |
