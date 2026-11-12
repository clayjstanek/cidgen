# Threat prioritisation (v0.6)

Ranks every visible track at each update, most threatening first and confidently friendly last, from
the Combat ID likelihood vector plus context, under a doctrine file you edit.

## Architecture

```
run folders ──► picture.py ──────────────► prioritize.py ─────────► evaluate.py
 (features,      one row per track:          ThreatPrioritizer         scores ranked lists
  asset edges,   P(F,N,H) from any CID        .prioritize(picture, t)   against truth threat
  labels)        model + context columns      → new ranked table
                       ▲                           ▲
       identity source: truth | BN | XGB |   configs/threat/doctrine_*.yaml
       GNN | STACK:a+b+c (prediction files)
```

- **Interface.** The track picture (`picture.py`) is the only thing TP sees: identity columns
  (`p_friend`, `p_neutral`, `p_hostile`) and context columns (time-to-go, closest approach, asset value,
  IFF, airway and corridor geometry, observable tier). Any CID model can feed it.
- **No shared global state, no copies.** `prioritize()` is a pure function of its input: it reads the
  picture and returns a new table, and a test checks that the caller's picture is unchanged. The only
  memory across updates is inside the `ThreatPrioritizer` object: sticky flags (a track declared
  hostile stays declared), the last smoothed score per track, and the last update time.
- **Update timing.** Call `prioritize()` on every full scan. Between scans, `needs_update(picture, t)`
  returns True when a new track appears or a track's time-to-go drops below `popup.ttg_s`, so a pop-up
  threat can trigger an immediate re-rank. The stored runs have snapshots every 10 s, so replay
  ranks every snapshot.

## Score and rules

```
urgency   = max over assets a: value_a · exp(−TTG_a / τ) · 1{CPA_a < radius_a · margin}
lethality = first matching row of the doctrine's lethality table (observable class)
score     = P(H) · lethality · urgency + w_amb · (1 − max P) · lethality · urgency
```

Then the doctrine's rules run in order. Each has conditions (ANDed), actions (`multiply`, `floor`,
`cap`, `tier`, `set_flag`, `sticky_flag`, `engage_block`) and a reason that is attached to the track.
Engage eligibility comes from the weapons-control state (`free`, `tight`, `hold`). The list is ordered
by tier (engage, monitor, protected), then smoothed score, then P(Friend) ascending.

Scores rise immediately but decay no faster than `hysteresis.decay_tau_s`, so ranks do not flicker.

## Rankers compared

| Ranker | What it does |
| --- | --- |
| `ph` | P(Hostile), ties broken by P(Friend): the starting point |
| `threat` | Expected-threat score, no rules |
| `doctrine` | Expected-threat score plus the doctrine's rules and engagement tiers |

## Metrics (`evaluate.py`)

The optimal list at each snapshot is every track sorted by truth threat priority; the first four
metrics measure distance from it and are the headline measures.

Truth is the generator's threat priority label: platform capability × asset value × exp(−TTG/τ), counted
only when the track will pass within an asset's radius, and zero for non-hostiles.

| Metric | Meaning | Better |
| --- | --- | --- |
| `threat_captured3` | Truth threat in the top 3 ÷ truth threat in the optimal top 3 | Higher; 1 is optimal |
| `top_threat_rank` / `top_threat_rank90` | Median and 90th-percentile rank of the single most dangerous track | Lower; 1 is optimal |
| `order_tau` | Kendall τ between list order and truth among real threats (snapshots with 3 or more) | Higher; 1 is optimal |
| `ndcg10` | Order of the top 10 against truth threat; 1 is perfect | Higher |
| `top1_in_top3` | The truly most threatening track is in the top 3 | Higher |
| `threat_recall3` | Share of real-threat track-snapshots in the top 3 | Higher |
| `first_top3_delay_s` | Median seconds from becoming a real threat to first reaching the top 3 | Lower |
| `never_top3` | Real threats that never reached the top 3 | Lower |
| `friend_above_threat` | A friend ranked above the most threatening real threat | Lower |
| `friend_engage` | Friendly track-snapshots placed in the ENGAGE tier | Lower |
| `airliner_in_top3` | An airliner in the top 3 | Lower |
| `top5_changes_min` | Tracks entering or leaving the top 5 per minute | Lower |

A real threat is a track with truth threat priority ≥ `--threat-min` (default 0.05).

Caution: the truth label uses the same geometry as the urgency term, so `ndcg10` partly rewards the
score for sharing the label's formula. The identity-driven measures (friends ranked above threats,
friends in the engage tier, delay for real threats) are the informative comparisons between CID models.

## Commands (Windows, from the folder that holds `cidgen`)

```
:: quick check, CPU only
python tests\test_threat.py

:: S9 test runs, identities from the existing S0 -> S9 predictions
python -m cidgen.threat.run_tp --runs "runs/S9_sensor_shift_*" --preds results_transfer --name S0_baseline__on__S9_sensor_shift --identity truth BN-learned XGB-own GNN-SD30-NOTQ STACK:BN-learned+XGB-own+GNN-SD30-NOTQ --out results_tp --dump-run seed3

:: compare two doctrine files
copy configs\threat\doctrine_default.yaml configs\threat\doctrine_free.yaml
:: (edit doctrine_free.yaml: weapons_control: free)
python -m cidgen.threat.run_tp --runs "runs/S9_sensor_shift_*" --preds results_transfer --name S0_baseline__on__S9_sensor_shift --identity BN-learned STACK:BN-learned+XGB-own+GNN-SD30-NOTQ --doctrine configs/threat/doctrine_default.yaml configs/threat/doctrine_free.yaml --out results_tp
```

`run_tp` needs, in `--preds`, the `allpred_<name>` file from `compare --save-preds` (BN and tree
identities) and the `pred_` files (neural identities); a `STACK:` source also needs the `valpred_` files.
It writes `tp_<name>__<doctrine>.csv` (metrics per identity × ranker) and, with `--dump-run`, every ranked
list for that run with tiers, flags and rule reasons.

## Using it live

```python
from cidgen.threat.prioritize import ThreatPrioritizer
tp = ThreatPrioritizer.from_yaml("configs/threat/doctrine_default.yaml")
for t, picture in stream:                       # picture: DataFrame with the PICTURE_COLUMNS
    if scan_complete(t) or tp.needs_update(picture, t):
        ranked = tp.prioritize(picture, t)      # ranked list with rank, tier, score, flags, reasons
```
