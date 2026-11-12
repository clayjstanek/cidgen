# Runbook: GNN tracker in cidgen (S12)

The new folder `cidgen_v0.6.1_hypersonic_GNN` is a copy of your untouched `cidgen_v0.6.1_hypersonic`
with this package unzipped over it. Your `cidgen_v0.*` folders are not touched.

Background: `TRACKERS.md` (what the tracker does, how labels are made, first results).

Runs, results, analysis and caches for tracker runs all get their own folders (`runs_gnn`,
`results_gnn`, `analysis_gnn`, `results_tp_gnn`, `cache_gnn`), so they never mix with label-mode files.

## G0. Setup (once)

| Step | Do this | Needs |
| --- | --- | --- |
| G0.1 | Copy `cidgen_v0.6.1_hypersonic` to `cidgen_v0.6.1_hypersonic_GNN` | - |
| G0.2 | Unzip `cidgen_gnn_tracker.zip` into `cidgen_v0.6.1_hypersonic_GNN` (the folder that holds `cidgen`), replacing files when asked. It **replaces** 3 files: `cidgen\generate.py`, `analyze_preds.py` and `README.md`. It **adds** `cidgen\trackers\` (3 files), `configs\scenarios\tracker\` (3 scenarios), `track_report.py`, `tests\test_gnn_tracker.py` and `TRACKERS.md`. | G0.1 |
| G0.3 | `cd` into `cidgen_v0.6.1_hypersonic_GNN`, then run `python -m pytest tests\test_gnn_tracker.py -q`. Expect `3 passed`. | G0.2 |
| G0.4 (optional) | Check that label mode is unchanged. Run `python -m cidgen.generate --scenario S12 --seed 15 --out runs_check`, then `fc /b runs_check\S12_hypersonic_seed15\labels.csv <your old runs>\S12_hypersonic_seed15\labels.csv`. Expect `no differences encountered`. Do the same for `features_track.csv` if you like. | G0.2 |

Run everything below from `cidgen_v0.6.1_hypersonic_GNN`.

## G1. Generate S12 with the GNN tracker (4 processes, CPU only, about 2 min per run)

| Step | Command | Needs |
| --- | --- | --- |
| G1.1 | `python -m cidgen.generate --scenario S12_gnn --seed 1 --seeds 5 --out runs_gnn` | G0 |
| G1.2 | `python -m cidgen.generate --scenario S12_gnn --seed 6 --seeds 5 --out runs_gnn` | G0 |
| G1.3 | `python -m cidgen.generate --scenario S12_gnn --seed 11 --seeds 5 --out runs_gnn` | G0 |
| G1.4 | `python -m cidgen.generate --scenario S12_gnn --seed 16 --seeds 5 --out runs_gnn` | G0 |

Each run prints a `tracker:` line (coverage, tracks per object, false-track rows, swaps).

## G2. Tracker quality

| Step | Command | Needs |
| --- | --- | --- |
| G2.1 | `python track_report.py --runs "runs_gnn/S12_gnn_seed*" --out results_gnn` | G1 all done |

## G3. Same CID and TP chain as S12, on tracks

Run G3.2 only after G3.1 has printed its first `epoch 1` line. G3.1 builds the new graph caches, and
two jobs must not build the same cache at once.

| Step | Command | Needs |
| --- | --- | --- |
| G3.1 | `python -m cidgen.models.train_gnn --runs "runs_gnn/S12_gnn_*" --out results_gnn --cache cache_gnn --no-graph --sensor-dropout 0.3 --drop-features track-quality --seed 1 --tag gru-sd30-notq-s1` | G1 all done |
| G3.2 | `python -m cidgen.models.train_gnn --runs "runs_gnn/S12_gnn_*" --out results_gnn --cache cache_gnn --edge-gate --sensor-dropout 0.1 --drop-features track-quality --seed 1 --tag gnn-sd10-notq-eg-s1` | G3.1 has printed `epoch 1` |
| G3.3 | `python -m cidgen.models.compare --runs "runs_gnn/S12_gnn_*" --preds results_gnn --out results_gnn --save-preds` | G3.1 and G3.2 finished |
| G3.4 | `python -m cidgen.models.route_preds --dir results_gnn --name S12_gnn --runs runs_gnn --route 1=gnn-sd10-notq-eg-s1 2=gru-sd30-notq-s1 3=gru-sd30-notq-s1 --tag route-eg-gru-s1` | G3.1 and G3.2 finished |
| G3.5 | `python analyze_preds.py --preds "results_gnn/pred_*_S12_gnn.csv" --runs runs_gnn --out analysis_gnn` | G3.4 |
| G3.6 | `python -m cidgen.threat.run_tp --runs "runs_gnn/S12_gnn_seed18" "runs_gnn/S12_gnn_seed19" "runs_gnn/S12_gnn_seed20" --preds results_gnn --name S12_gnn --identity truth BN-learned XGB-own ROUTE-EG-GRU-S1 STACK:BN-learned+XGB-own+ROUTE-EG-GRU-S1 --out results_tp_gnn` | G3.3 and G3.4 |

## What to send back

- From `results_gnn`: `tracker_runs.csv`, `tracker_tiers.csv`, and the compare tables.
- From `analysis_gnn`: the files `analyze_preds.py` writes.
- From `results_tp_gnn`: everything.
- The `tracker:` lines from G1, if they are handy.

With those I can set the tracker results against your S12 label-mode results: CID by tier,
friendly aircraft placed in ENGAGE, and threat captured in the top 3.

## Later (not in this package)

- `S12F_gnn` and `S12D_gnn` are ready to use the same way. Change the scenario and the names.
- JPDA and MFA through Stone Soup. Install Stone Soup on your side with `pip install stonesoup`.
- The VB tracker adapter.
