# cidgen — synthetic multi-sensor air picture for Combat ID research

Generator v0.1 implementing the *Combat ID Synthetic Scenario Generator — Specification v0.1*
(Tiers 1–2, scenarios S0 and S1). It produces labeled, multi-sensor scenarios to compare a
Bayesian Network baseline against GNN, temporal and tree models under controlled conditions.

**All platform, emitter, engine and sensor parameters are notional.** The three radar classes
follow the *roles* of Echodyne-, Sentinel- and Patriot-class systems with invented numbers.

## Install

```
pip install -r requirements.txt        # numpy scipy pandas pyarrow pyyaml matplotlib scikit-learn
python tests/smoke_gpu.py              # run first: CUDA + PyG + one hetero GATv2/SAGE pass
```

PyG core (torch-geometric 2.8.1) is all the GNN side needs; the compiled extensions
(pyg-lib, torch-scatter, torch-sparse) are not used. Python 3.12, torch 2.13 + cu130 targeted.

## Quick start

```
python -m cidgen.generate --scenario S0 --seed 1 --out runs            # 30 min scenario, ~80 s
python -m cidgen.generate --scenario S1 --seed 1 --seeds 10 --out runs # 10 seeds of the ablation
python -m cidgen.replay runs/S0_baseline_seed1                         # map PNG for SME review
python -m cidgen.validate runs/S0_baseline_seed1 runs/S1_zero_relational_seed1 --determinism S0
```

```python
from cidgen.graph_export import RunGraphs, to_heterodata
rg = RunGraphs("runs/S0_baseline_seed1")
data = to_heterodata(rg.snapshot(float(rg.times[90])), device="cuda")   # torch_geometric HeteroData
```

## Real tracker (GNN) -- additions to v0.6.1-hypersonic

Optional: scenarios without `tracking.method: gnn` are unchanged (S12 seed 15 regenerates byte-for-byte). See `TRACKERS.md`.

- `cidgen/trackers/gnn.py`: global nearest neighbour tracker on all radar reports (clutter included, true ids hidden): chi-square gate, Hungarian assignment, 3-hit confirmation, coast deletion, duplicate-track dropping, filter tier judged from observables.
- `cidgen/trackers/integrate.py`: IFF and cued-sensor reports re-attributed to tracks; track-to-truth matching (labels and scoring only); false-track rows; tracker quality tables.
- `configs/scenarios/tracker/S12_gnn.yaml`, `S12F_gnn.yaml`, `S12D_gnn.yaml`; `track_report.py`; `tests/test_gnn_tracker.py`.

## Hypersonic (Tier 3) profile tracks -- additions to v0.6.1

Optional; scenarios without a `hypersonic.raid` block are unchanged (S0 regenerates byte-for-byte). See `HYPERSONIC_PROFILES.md`.

- `cidgen/hypersonic_profiles.py`: plays back track shapes described leg by leg in the scenario YAML (altitude, speed, optional weave per segment; any value may be a distribution). No vehicle model and no vehicle values in code.
- Tier 3 plumbing: generic placeholder class `H`, per-entity time-varying RCS multiplier, Tier 3 tracker settings, Tier 3 BN and `observable_tier` = 3 at speed >= 1000 m/s, `tier3_high_speed` doctrine lethality row, `hypersonic_segments` label table.
- `configs/scenarios/S12_hypersonic.yaml`: S0 plus an airliner worked example and an artificial plumbing-test target. Replace the test target with your own profiles.
- `tests/test_hypersonic_profiles.py`: CPU-only check (about 15 s).

## What changed in v0.6.1

- `run_tp` also reports distance from the optimal list (truth threat order): threat captured in the top 3, rank of the most dangerous track, and order among real threats.
- Default doctrine: the `imminent` rule now requires the track to be closing, so own drones loitering inside an asset's radius are no longer pushed to the top.

## What changed in v0.6

Threat prioritisation; no data or model changes, existing runs and predictions remain valid. See `THREAT_PRIORITIZATION.md`.

- `cidgen/threat/prioritize.py`: `ThreatPrioritizer.prioritize(picture, t)` ranks all tracks in a snapshot from P(F,N,H), lethality and urgency against the defended assets, then applies doctrine rules and engagement tiers. Pure function of its input; memory (sticky flags, hysteresis) lives in the object.
- `configs/threat/doctrine_default.yaml`: notional weapons-control state, lethality table, rules with reasons.
- `cidgen/threat/picture.py`: the CID-to-TP interface; identity from truth, any BN or tree model, any neural prediction file, or a stack.
- `python -m cidgen.threat.run_tp`: replays runs, compares the P(Hostile), expected-threat and doctrine rankers for each identity source, and scores them against the truth threat label (`cidgen/threat/evaluate.py`).
- `tests/test_threat.py`: CPU-only check (about a minute).

## What changed in v0.5

Ensemble evaluation; no data or model changes, existing runs and trained models remain valid. See `ENSEMBLE_STUDY.md`.

- `train_gnn` also writes `valpred_<model>_<name>.csv` (validation predictions) and stores its exact split in the `.pt` file.
- `python -m cidgen.models.predict <model.pt>` recreates validation predictions from models trained before v0.5, with no retraining; `--check-test` confirms the reload reproduces the saved test predictions.
- `compare --save-preds` also writes `allpred_<name>`: validation and test predictions of the deployable BN and tree models, with out-of-training-range flags.
- `python -m cidgen.models.ensemble` fits mean and log pools, stacking, tier gating and an out-of-range fallback on validation runs, scores them on test runs like `compare`, and tabulates Unknown / Pending rules.
- `tests/test_ensemble.py`: CPU-only plumbing test (about a minute).

## What changed in v0.4

Transfer study support; existing v0.3 runs remain valid. See `TRANSFER_STUDY.md` for the design and commands.

- Scenario files in `configs/scenarios/transfer/`: S0_EOIR, S0_NOEOIR, S0_VARIED, S9_EOIR. A scenario can name a `base:` scenario, and a `randomize:` block draws conditions per seed.
- `compare` also writes breakdowns by tier, by friendly role, and own drones by camera coverage.
- `train_gnn --sensor-dropout 0.3` hides each sensor family at random during training; `--tag` names outputs.
- `python -m cidgen.runinfo runs runs_transfer` lists run folders with their key conditions.

## What changed in v0.3 (regenerate all runs)

v0.2 results showed every model at 0.98+ in four of five scenarios, and a BN close to optimal because
the generator drew each sensor's evidence independently given allegiance and class, exactly the
BN's own assumption. v0.3 adds the mechanisms that make fielded BNs struggle:

| Change | Parameter | Default | Why |
| --- | --- | --- | --- |
| Threat axis random per run | `world.threat_axis_deg` | `random` | Models can no longer learn the map; all models get `threat_offset_deg` instead |
| IFF outage episodes | `iff.outage_fraction`, `outage_mean_s` | 0.1, 180 s | Friends go silent for minutes; history (did it reply earlier?) matters |
| ESM misclassified inside jam strobes | `ambiguity.esm_jam_factor` | 0.35 | Evidence reliability depends on context the BN structure cannot represent |
| Persistent NCTR misreads | `ambiguity.p_nctr_persistent_error` | 0.1 | Same wrong emitter/engine match on every report for that airframe |
| Off-axis hostile ingress | `deception.p_offaxis_ingress` | 0.15 | Threat sector is no longer decisive |
| Friendly tactical returns | `composition.n_friendly_tactical` | 1 | Friends flying low, fast and inbound from the threat sector |
| S9 sensor shift | `S9_sensor_shift.yaml` | — | Train on S0, test under changed sensors and force mix |

Evaluation changes: tree models no longer see raw position; all learned models train on the same
`train` runs; `compare` reports 95% intervals by resampling whole aircraft, plus paired differences
against BN-oracle and BN-learned.

```
python -m cidgen.models.train_gnn --runs "runs/S0_*" --test-runs "runs/S9_*" --out results            # shift
python -m cidgen.models.compare   --runs "runs/S0_*" --test-runs "runs/S9_*" --preds results --out results
```

## Models (v0.2)

```
pip install pgmpy xgboost                    # optional extras
python tests/test_bn_pgmpy.py                # numpy BN inference == pgmpy VariableElimination
python tests/test_gnn_quick.py               # 2-epoch GNN + GRU run on tiny data (~1 min)

python -m cidgen.models.train_gnn --runs "runs/S0_*" "runs/S1_*" --out results              # GNN
python -m cidgen.models.train_gnn --runs "runs/S0_*" "runs/S1_*" --out results --no-graph   # GRU only
python -m cidgen.models.compare   --runs "runs/S0_*" "runs/S1_*" --preds results --out results
```

| Model | What it is | Where |
| --- | --- | --- |
| BN-learned | Per-tier BN (root Allegiance, hidden Platform Class, 11–13 observation nodes), CPTs from train runs | `models/bn.py` |
| BN-oracle | Same structure fitted on all runs incl. test: upper bound for the structure | `models/bn.py` |
| BN-SME | Oracle CPTs with log-odds noise, 0.05 rounding and overconfident strong cues | `models/bn.py` |
| BN-context | BN-learned plus neighbour-context nodes | `models/bn.py` |
| XGB-own, XGB-own+nb | Gradient-boosted trees on flat features, without / with neighbour aggregates | `models/compare.py` |
| GRU | Per-track GRU over 30 s of history, no message passing | `models/gnn.py --no-graph` |
| GNN | Same GRU encoder + 2 layers of heterogeneous GATv2 (track–track) and SAGE (asset, ESM) | `models/gnn.py` |

Every model is scored on the same held-out runs (`models/common.split_runs`: last 15% of seeds per
scenario for test, the 15% before that for validation) and the same rows: objects of interest at
snapshot times. Metrics: balanced accuracy, log loss, Brier, ECE, per-class recall, and the two
guardrails, airliners called hostile (S3) and friends called hostile (S8). GNN and GRU
probabilities are temperature-scaled on the validation runs. The BN reads tier from observables
(speed, altitude, RCS), not from the label. Weights saved by `train_gnn` contain numpy arrays, so
load them with `torch.load(path, weights_only=False)`.

## Layout

| Path | What it does |
| --- | --- |
| `configs/defaults.yaml` | Every scenario parameter with its default (spec Section 7) |
| `configs/scenarios/*.yaml` | Scenario overlays: `S0_baseline`, `S1_zero_relational` |
| `cidgen/world.py` | Assets, airways, friendly corridor, sensor sites |
| `cidgen/library.py` | Notional platform classes, variants, emitter and engine catalogues |
| `cidgen/hypersonic_profiles.py` | Hypersonic (Tier 3) segment-profile player; see `HYPERSONIC_PROFILES.md` |
| `cidgen/kinematics.py` | Waypoint guidance with turn/climb/accel limits, formation and swarm followers |
| `cidgen/entities.py` | Behaviors: strike packages, sweeps, CAP, RTB, airways, GA, helos, AEW, swarms, own UAS, hobbyists, birds |
| `cidgen/sensors/` | `radar.py` (R-C, R-S, R-P), `iff.py`, `esm.py`, `nctr.py` (JEM, LIDAR, EO/IR) |
| `cidgen/tracking.py` | Constant-velocity Kalman filter per track from radar reports |
| `cidgen/features.py` | Per-track features, track–track / track–asset / ESM–track edges |
| `cidgen/labels.py` | Truth labels, threat priority, unresolved-formation flag |
| `cidgen/graph_export.py` | Snapshots as numpy arrays or PyG `HeteroData` |
| `cidgen/validate.py`, `cidgen/replay.py` | Generator checks and replay plot |

## Output tables (one folder per run; Parquet if pyarrow is present, else CSV)

| Table | Grain | Notes |
| --- | --- | --- |
| `entities` | entity | truth: allegiance, class, variant, role, group, transponder, deception flags |
| `truth_states` | entity × 1 s | ENU position/velocity, emitter on/off, intent phase |
| `assets`, `sites`, `airways`, `corridor` | static | world geometry |
| `reports_radar` | report | polar measurement + sigmas, RCS, SNR, jam flag, micro-Doppler (R-C) |
| `reports_iff` | reply | Mode 3/C, Mode 5 validity, Level 2 self-reported position |
| `reports_esm`, `esm_fixes` | report / fix | per-site AOA + emitter posterior + RF; triangulated fixes incl. ghosts |
| `reports_jem`, `reports_lidar`, `reports_eoir` | report | NCTR evidence (cued) |
| `features_track` | track × 2 s | everything a model may use, computed from sensors only |
| `labels` | track × 2 s | same keys as `features_track` |
| `edges_track_track`, `edges_track_asset`, `edges_esm_track` | edge × 10 s | graph structure |

### Label columns — never use as model inputs

`entities.*` (except `eid`), `truth_states.*`, `labels.*`, and in report tables: `eid_true`,
`merged_eids`, `n_merged`, `false_alarm`, `is_ghost`. Radar `eid` stands in for the tracker's
association (merged reports go to the strongest contributor; the others coast).

## Scenarios in v0.1

| ID | Purpose | Expected GNN vs. oracle BN |
| --- | --- | --- |
| S0 | Baseline mixed air picture, Tiers 1–2 | Better |
| S1 | Zero-relational ablation: all starred dials at null | Tie |
| S4 | Weak package members: rho_variant 0.9, emcon to 30 km, larger packages | Clearly better |
| S7 | Deception: airliner mimics, IFF spoofs incl. Level 2 replays, bird-like drones | Better |
| S11 | Two drone swarms among 12 bird flocks, hobbyists and own UAS | Better |
| S9 | Sensor and force-mix shift; test-only, models trained on S0 | Degrades least |

The remaining spec scenarios are YAML overlays on the same defaults and will be added as needed.

## Validation results (this build)

See `VALIDATION.md`: radar residuals match configured sigmas, IDs carry no label information, the S1 ablation removes all neighbour information, and runs are deterministic.

## Known simplifications (v0.1)

- No tracker: radar reports carry the truth-associated `eid`; non-radar reports have a configurable mis-association rate.
- Flat earth; 4/3-earth radar horizon; no terrain masking.
- Swerling Pd uses the aspect-mean RCS; the reported RCS is drawn independently per dwell.
- R-C update set to 0.5 s (spec 0.25 s) to halve data volume; one config value restores it.
- Clutter is written as false-alarm reports but not formed into tracks.
- Tier 3 (ballistic), OPIR, anti-radiation missiles and hypersonic glide vehicles are not in v0.1. Tier 3 tracks can be added as user-described kinematic profiles (`HYPERSONIC_PROFILES.md`); there is no vehicle model.
