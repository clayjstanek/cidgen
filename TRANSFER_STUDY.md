# Transfer study: was EO/IR reliance the cause, and which models transfer best?

The S0 → S9 shift test showed the GNN losing its fratricide advantage under changed conditions,
with 42% of our own drones called hostile and the cause traced to reliance on the EO/IR camera.
This study tests that diagnosis directly, and characterises how well each model family (Bayesian
Networks, XGBoost, GRU, GRU + GATv2) transfers to conditions it was not trained on.

## Design

Every experiment trains on one condition and tests on another. All model families are trained on
the same runs and scored on the same aircraft, exactly as in the earlier comparisons.

| ID | Train on | Test on | Camera visibility, train → test | Question |
| --- | --- | --- | --- | --- |
| T0 | S0 | S9 | 0.8 → 0.5 | Reference: the shift test already run |
| T1 | S0_EOIR | S9 | 0.5 → 0.5 | EO/IR matched at S9's level: does the own-drone failure disappear? |
| T2 | S0 | S9_EOIR | 0.8 → 0.8 | EO/IR matched at S0's level: the mirror-image check |
| T3 | S0_NOEOIR | S9 | none → 0.5 | With no camera to rely on, does the model learn the other cues? |
| T4 | S0_VARIED | S9 | 0.6–1.0 → 0.5 | Does training on varied conditions make models degrade more gracefully? |
| T5 | S0, sensor dropout | S9 | 0.8 → 0.5 | Can training alone (hiding sensors at random) fix it, with no new data? |
| T6 (optional) | S0_EOIR | S9_EOIR | 0.5 → 0.8 | More camera at test than in training |

**Controlled by construction.** For the same seed, S0_EOIR and S0_NOEOIR are identical to S0 in
every aircraft, flight path and radar, IFF, ESM, JEM and LIDAR report; only the camera differs.
S9_EOIR is likewise identical to S9 except the camera. So T1 and T2 change the camera and nothing else.

**S0_VARIED** draws its conditions per seed from ranges that stop short of S9's values (radar noise
0.8–1.8 against S9's 2.5, ESM 72–88% correct against 65%, and so on; see the YAML). S9 remains an
extrapolation, never a condition seen in training. Each seed's draws are printed by the generator
and saved in its `config_resolved.yaml`.

**Sensor dropout (T5)** is a training option, not data: for each track and each sensor family
(IFF, ESM, JEM, LIDAR, EO/IR, micro-Doppler) independently, with probability 0.3 the family's
evidence is removed for the whole history window during training, exactly as if that sensor had
never reported. Validation and test data are never altered.

## Files

New scenario files are in `configs/scenarios/transfer/`: `S0_EOIR`, `S0_NOEOIR`, `S0_VARIED`, `S9_EOIR`.
Each declares a `base:` scenario and changes only what differs from it.

## Commands

Run from the folder that contains `cidgen` and `configs`. New runs go to a separate folder,
`runs_transfer`, so that patterns such as `runs/S0_*` never pick up a variant by accident.
Use the full scenario names below. `<cache>` stands for your cache folder.

### 1. Generate (about 2 hours of CPU; run 3 Anaconda prompts in parallel)

```
python -m cidgen.generate --scenario S0_EOIR   --seed 1 --seeds 20 --out runs_transfer
python -m cidgen.generate --scenario S0_NOEOIR --seed 1 --seeds 20 --out runs_transfer
python -m cidgen.generate --scenario S0_VARIED --seed 1 --seeds 20 --out runs_transfer
python -m cidgen.generate --scenario S9_EOIR   --seed 1 --seeds 10 --out runs_transfer
```

Check what you have:

```
python -m cidgen.runinfo runs runs_transfer
```

Expected: S0_EOIR, S0_NOEOIR and S0_VARIED with 20 runs each, S9_EOIR with 10, and your existing
S0_baseline (20) and S9_sensor_shift (10) in `runs`.

### 2. Train GNN and GRU for each experiment (each line trains one model, 5–10 minutes on the GPU)

Put all results in one folder, `results_transfer`. Copy in the T0 predictions you already have:

```
mkdir results_transfer
copy results_v3\pred_gnn_S0_baseline__on__S9_sensor_shift.csv results_transfer\
copy results_v3\pred_gru_S0_baseline__on__S9_sensor_shift.csv results_transfer\
```

```
:: T1  S0_EOIR -> S9
python -m cidgen.models.train_gnn --runs "runs_transfer/S0_EOIR_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache <cache>
python -m cidgen.models.train_gnn --runs "runs_transfer/S0_EOIR_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache <cache> --no-graph

:: T2  S0 -> S9_EOIR
python -m cidgen.models.train_gnn --runs "runs/S0_baseline_*" --test-runs "runs_transfer/S9_EOIR_*" --out results_transfer --cache <cache>
python -m cidgen.models.train_gnn --runs "runs/S0_baseline_*" --test-runs "runs_transfer/S9_EOIR_*" --out results_transfer --cache <cache> --no-graph

:: T3  S0_NOEOIR -> S9
python -m cidgen.models.train_gnn --runs "runs_transfer/S0_NOEOIR_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache <cache>
python -m cidgen.models.train_gnn --runs "runs_transfer/S0_NOEOIR_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache <cache> --no-graph

:: T4  S0_VARIED -> S9
python -m cidgen.models.train_gnn --runs "runs_transfer/S0_VARIED_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache <cache>
python -m cidgen.models.train_gnn --runs "runs_transfer/S0_VARIED_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache <cache> --no-graph

:: T5  S0 with sensor dropout -> S9 (outputs are tagged gnn-sd30 and gru-sd30, so they sit beside T0's)
python -m cidgen.models.train_gnn --runs "runs/S0_baseline_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache <cache> --sensor-dropout 0.3
python -m cidgen.models.train_gnn --runs "runs/S0_baseline_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache <cache> --sensor-dropout 0.3 --no-graph
```

### 3. Compare (each line fits the Bayesian Networks and XGBoost on the same training runs; about 10 minutes)

```
python -m cidgen.models.compare --runs "runs/S0_baseline_*"         --test-runs "runs/S9_sensor_shift_*"  --preds results_transfer --out results_transfer
python -m cidgen.models.compare --runs "runs_transfer/S0_EOIR_*"    --test-runs "runs/S9_sensor_shift_*"  --preds results_transfer --out results_transfer
python -m cidgen.models.compare --runs "runs/S0_baseline_*"         --test-runs "runs_transfer/S9_EOIR_*" --preds results_transfer --out results_transfer
python -m cidgen.models.compare --runs "runs_transfer/S0_NOEOIR_*"  --test-runs "runs/S9_sensor_shift_*"  --preds results_transfer --out results_transfer
python -m cidgen.models.compare --runs "runs_transfer/S0_VARIED_*"  --test-runs "runs/S9_sensor_shift_*"  --preds results_transfer --out results_transfer
```

The first line covers T0 and T5 together: its table has rows for GNN, GRU, GNN-SD30 and GRU-SD30.

Each compare writes six files named `compare_<train>__on__<test>`:

| File | Contents |
| --- | --- |
| `.csv` | Every metric for every model |
| `_ci.csv` | 95% ranges, resampling whole aircraft |
| `_diffs.csv` | Paired verdicts against BN-learned, BN-oracle and GRU |
| `_by_tier.csv` | Balanced accuracy, friends called hostile and missed hostiles for Tier 1 (drones) and Tier 2 |
| `_by_role.csv` | Friends called hostile for each friendly role (own drones, CAP, corridor returns, ...) |
| `_own_drones_eoir.csv` | Own drones called hostile, split by whether the camera had reported on them |

## How to read the results

The decisive number is **own drones called hostile** (`_own_drones_eoir.csv`, column
`called_hostile_all`), with overall **friends called hostile** and the GNN-versus-BN verdicts
beside it. In T0 the GNN calls 42% of own drones hostile and the GRU 48%.

| If you see | It means |
| --- | --- |
| T1 and T2 bring the GNN's own-drone error down to a few percent, and its fratricide back below the BN's with a clear verdict | The EO/IR mismatch was the critical issue |
| T1 and T2 leave own-drone error high | Something else in the shift drives it (ESM accuracy 80% → 65%, NCTR misreads 10% → 20%, more swarms); the next step is an ESM-matched pair |
| T3 gives low own-drone error with no camera in training | The model can learn the non-camera cues when it is not handed a shortcut |
| T4 degrades less than T0 for the GRU and GNN | Training on varied conditions improves extrapolation |
| T5 fixes own drones in T0's own conditions | The remedy is a training change; no special data is needed |
| GNN better than GRU in each experiment (the `vs GRU` rows of `_diffs.csv`) | The graph's value under shift is consistent, not specific to one condition |

For drift, compare each model's balanced accuracy and friends called hostile across T0–T5 with its
in-distribution S0 result (GNN 0.983 and 1.5%; see the main report). The model whose numbers move
least has the most margin before operational limits are crossed.

Send back the `results_transfer` compare files (all six per experiment) and I will write up the study.
