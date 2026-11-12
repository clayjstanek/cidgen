# Trackers in cidgen (v0.6.1-gnn)

Until now cidgen did not solve data association. Every radar report carried the id of the object
that produced it, clutter was thrown away, and one Kalman filter ran per id ("association by
label"). That is still the default and its output is unchanged, byte for byte.

A scenario can now ask for a real multi-target tracker instead:

```yaml
tracking:
  method: label   # default: the v0.6.1 stand-in
  method: gnn     # global nearest neighbour tracker
```

Three ready-made scenarios are in `configs/scenarios/tracker/`:

| Scenario   | Same as | Output folder name    |
|------------|---------|-----------------------|
| `S12_gnn`  | S12     | `S12_gnn_seed<N>`     |
| `S12F_gnn` | S12F    | `S12F_gnn_seed<N>`    |
| `S12D_gnn` | S12D    | `S12D_gnn_seed<N>`    |

The same seed gives the same world, the same objects and the same sensor reports as the base
scenario. Only the tracking differs, so a label run and a GNN run of one seed can be compared
directly.

```
python -m cidgen.generate --scenario S12_gnn --seed 1 --seeds 20 --out runs_gnn
python track_report.py --runs "runs_gnn/S12_gnn_seed*" --out results_gnn
```

After that, `gnn_train`, `compare`, `predict`, `route_preds` and `run_tp` run on `runs_gnn` exactly
as they do on `runs`. Use separate results and cache folders so label-mode files are never mixed in.

## What the GNN tracker does

The tracker is in `cidgen/trackers/gnn.py`. It receives **every** radar report, clutter included,
with the true object id hidden. It processes one radar look at a time (all reports from one radar at
one time), in time order:

1. **Predict** every live track to the look time with a constant-velocity Kalman filter.
2. **Gate.** A report can go to a track only if d² = rᵀS⁻¹r ≤ `gate_chi2`, where r is the
   report-minus-prediction position difference and S = HPHᵀ + R. The default is 16, which keeps
   about 99.9% of true reports for 3-D positions.
3. **Assign** reports one-to-one at the lowest total cost d² + ln|S| (Hungarian algorithm,
   `scipy.optimize.linear_sum_assignment`). Confirmed tracks choose first. Tentative tracks then
   compete for the reports that are left.
4. **Update** the assigned tracks. Every report still left over starts a new tentative track.
5. **Confirm** a tentative track once it has `confirm_hits` updates (default 3) within
   `confirm_window_s` (default 10 s) of its first report. A tentative track that misses that window is
   deleted. A confirmed track is deleted after its tier's `coast_s` without an update.
6. **Drop duplicates.** At each feature-grid time, two confirmed tracks are checked on their 6-D
   state difference against P₁ + P₂. If they agree within `merge_chi2` (default 12.6) at
   `merge_checks` consecutive grid times (default 3), the younger track is dropped.

The tracker emits only confirmed tracks. Their estimates go on the feature grid in the same format
as before, with the track number in the `eid` column. Track numbers start at 1,000,000, so they can
never be confused with true object ids.

**Filter settings per track** (q, initial velocity sigma, coast time) come from
`tracking.tiers`, and the tier is chosen from what is observed, never from the truth:

- **First guess, from the first report:**
  - Tier 3 if the altitude is at least `tier3_min_alt_m` (default 15 km), or if the measured radial
    speed is at least the acquisition rule's Tier 3 speed.
  - Tier 1 if the report comes from the counter-UAS radar, or the object is low and small.
  - Tier 2 otherwise.
- **Final tier:** fixed after `acquisition.window_s` by the same speed / altitude / RCS rule as
  `tier_source: observed`.

**Registration bias.** Each radar has a fixed registration bias. The tracker knows only the
specified bias sigma (`sensors.radar_bias_sigma` × `bias_scale`), never the realised bias, and adds
that bias covariance to every report's R. Without it, two radars see one aircraft about one sigma
apart, reports fall outside the gate, and duplicate tracks result. On S12 seed 15 this one change
cut false-track rows from 13.9% to 3.4%.

## How the other sensors reach tracks

| Sensor | Label mode | GNN mode |
|---|---|---|
| Radar | report's true id | the track the GNN assigned the report to |
| IFF | reply's (possibly garbled) id | gated to the nearest track by measured range and azimuth (2-D chi-square, `iff_gate_chi2` = 9.21) |
| JEM, LIDAR, EO/IR (cued) | the id the cue went to | the track following the object the cue went to (the generator's cue mis-association is kept) |
| ESM | gated geometrically | unchanged |

## Labels and scoring (truth used here, and only here)

At every grid time, confirmed tracks are matched one-to-one to true objects:

- Matching uses the Hungarian algorithm on distance.
- The gate is max(`match_gate_m` = 2 km, `match_gate_sigmas` = 4 × the track's position sigma).
- A pairing that still gates is preferred from one time to the next.

A matched (track, t) row takes the matched object's labels. An unmatched row is a **false-track
row**:

- allegiance −1, class `FALSE_TRACK`, not an object of interest;
- CID metrics and the threat list skip it, the same way they skip birds;
- it still appears as a neighbour in the track graph.

New tables in a GNN run:

| Table | Contents |
|---|---|
| `track_truth` | (track, t) → matched object and distance. Scoring only, never a model input. |
| `tracks` | One row per confirmed track: first report, confirmation time, last estimate, updates, filter tier. |
| `tracker_objects` | One row per object the radars detected: coverage, tracks used, id switches, confirmation delay, mean error. |
| `tracker_summary` | One row per run. |
| `reports_radar.track_id` | The track each radar report updated (−1 = none). |

`labels` gains two columns, `truth_eid` and `is_false_track`. `analyze_preds.py` uses `truth_eid`
when present, so an aircraft whose track broke is counted once.

## First results (S12 seed 15, one run)

| | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Objects | 12 | 46 | 17 |
| Coverage, median (label mode: 1.00 / 1.00 / 1.00) | 0.98 | 0.99 | 0.88 |
| Tracks per object, median | 1 | 2.5 | 2 |
| Confirmation delay, median | 2 s | 3 s | 6 s |
| Position error, median | 36 m | 262 m | 242 m |

Whole run: false-track rows 3.4%, 265 duplicate tracks dropped.

- **Tier 3 coverage.** The loss is mostly at first detection. At long range the radar sees a Tier 3
  object only every 5–7 s, so 3 hits take 10–20 s to collect, and a 6 s coast can drop a track in a
  gap.
- **Tracks per object.** The mean is pulled up by one tight 4-ship hostile formation. Its members
  are 200 m–1 km apart, about the radar's cross-range error at 50 km. Reports merge and split, and
  tracks keep breaking and swapping between members (about 100 tracks per member). A friendly pair
  shows the same effect more mildly (about 20).

These are real tracking effects that association by label hid. They are what the JPDA, MFA and VB
trackers should improve on.

## Scoring the threat list against all threats (run_tp)

The original threat-list metrics (`threat_captured3`, `first_top3_delay_s`, ...) are computed only over
tracks in the picture. A threat the tracker has not confirmed yet, has dropped, or holds only inside a
group track is simply absent, never scored as a miss. That flatters any real tracker. `run_tp` now also
scores against every hostile object's true threat (recomputed from `truth_states`, exactly the labels'
formula), counting each object from its first radar report on:

| Column | Meaning |
|---|---|
| `threat_tracked` | Share of real-threat object-snapshots whose object has its own track in the picture. |
| `threat_covered` | ... or lies within 500 m of some track: unresolved group members sharing a group track. |
| `threat_captured3_all` | Strict top-3 capture against all objects: a group track earns one member's threat. |
| `top1_in_top3_all`, `first_top3_delay_all_s`, `never_top3_all` | As before, but an object counts as in the top 3 if a top-3 track covers it. |

The original columns are unchanged, so earlier results stay comparable. Label-mode runs get the new
columns too, and need them as the baseline.

First check (perfect identity, doctrine ranker):

| Run | Label: tracked / covered / captured3_all | GNN: tracked / covered / captured3_all |
|---|---|---|
| seed 18 | 0.961 / 0.991 / 0.869 | 0.869 / 0.983 / 0.860 |
| seed 20 | 0.968 / 0.998 / 0.887 | 0.785 / 0.976 / 0.802 |

Almost all of the gap is **unresolved groups**:
- **Hostile drone swarm.** Eight drones flying about 55 m apart, 13 m at the closest. The radars merge
  them into one or two reports per look. Every member is within 200 m of a track, but only 34–42% have
  their own track. Association by label held 82% of the swarm only because each merged report updated
  whichever member returned the strongest echo, and that member changes from look to look.
- **Tier 2 escort and helicopter formations.** The same effect, less severe.

This is a sensor-resolution limit, not a GNN tuning fault. The low Tier 1 coverage in seeds 18 and 20
is the swarm alone: own drones and hobbyist drones are at 1.00.

## JPDA and the variational (VB) tracker

```yaml
tracking:
  method: jpda    # or vb
  soft: {p_detect: 0.9, min_assoc_mass: 0.2, window_s: 20.0, iterations: 5, tol_m: 1.0}
```

Both run on the GNN's track management: tentative tracks, 3-hit confirmation, coast deletion,
duplicate dropping and the observable tier are all unchanged. They differ only in how a **confirmed**
track uses one radar look. The code is in `cidgen/trackers/soft.py`, and the assignment marginals are
in `cidgen/trackers/assoc.py`.

- **Assignment marginals.** For each gated cluster of tracks and reports in one look, these are the
  exact probabilities that track i produced report j, over all one-to-one assignments. They are
  computed by a forward-backward pass over subsets of tracks, and are checked against brute-force
  enumeration and against the subset DP in your MATLAB-port code. Clusters larger than 12 tracks fall
  back to loopy belief propagation, which is within about 0.04 of exact.
- **Scores.** ln(P_D/(1−P_D)) − ln ρ + ln N(z; Hx, S). ρ is the radar's specified clutter density at
  the report, taken from `clutter_rate` spread uniformly over range, the azimuth sector and 0–3°
  elevation, as the generator spreads its clutter. It is never estimated from truth.
- **JPDA update.** The textbook moment-matched mixture over "missed" and each gated report.
- **VB filter step.** The same marginals, but the variational update: one soft pseudo-measurement,
  weighted by information (R̃⁻¹ = Σ β_j R_j⁻¹).
- **VB window step.** At every grid time, this runs for every cluster of tracks that shared reports in
  the last `window_s` seconds. It is coordinate ascent on q(X)q(A):
  1. Per look, assignment marginals from the smoothed states, with scores
     w + E_q[ln N(z; Hx, R)] = w + ln N(z; Hm, R) − ½ tr(R⁻¹P).
  2. A soft Kalman/RTS smoother per track, started from the track's state at the window start.
  3. Steps 1–2 repeat up to `iterations` times, stopping early when nothing moves more than `tol_m`.

  Each track then restarts from its smoothed state at the newest look. Only data up to the grid time
  is used, so the output stays causal: a fixed-lag re-estimate, not hindsight.

### The tracker bench (`tracker_bench.py`)

Because the VB tracker is new as a tracker in its own right, it is first tested where the answer is
known. The bench has one radar (σ = 100 m per axis, a look every 2 s, Pd 0.9, clutter at the density
the trackers assume) and two kinds of case:
- two aircraft crossing at 10°, 30° or 90°;
- four aircraft line abreast, k σ apart, flying straight for 90 s and then fanning out.

Each tracker is compared with an ORACLE: the same Kalman filter fed the true associations. Duplicate
dropping is off on the bench, since one radar has no registration bias.

20 Monte Carlo runs each:

| Case | Tracks per object (oracle / GNN / JPDA / VB) | RMSE, m (oracle / GNN / JPDA / VB) | NEES (3 = consistent) |
|---|---|---|---|
| Crossing at 10° | 1.45 / 1.75 / 1.75 / 1.85 | 114 / 112 / 110 / 112 | all about 2.5 |
| Crossing at 30° and 90° | all about 1.0–1.25 | all about 110 | all about 2.5 |
| Formation, 1σ apart | 2.9 / 3.7 / 3.9 / 3.8 | 113 / 125 / **211** / 118 | 2.4 / 3.1 / 1.7 / 2.8 |
| Formation, 2σ apart | 1.7 / 3.1 / 3.7 / 3.1 | 115 / 120 / 127 / **119** | 2.5 / 2.9 / 2.1 / 2.9 |
| Formation, 3σ apart | 1.1 / 2.1 / 3.2 / **2.0** | 115 / 120 / 123 / **119** | 2.5 / 2.8 / 2.4 / 2.8 |
| Formation, 5σ and 8σ apart | all about 1.0–1.2 | all about 112 | all about 2.5 |

Reading the bench:
- **Well-separated targets.** All three trackers match the oracle. The VB machinery does no harm when
  association is easy.
- **Close formations.** JPDA shows its known weakness, track coalescence. Its tracks are pulled
  together (RMSE 211 m at 1σ), and its covariance is too large (NEES 1.7). VB does not coalesce: its
  window re-solves the assignments with the smoothed states. VB has the lowest error of the three real
  trackers and slightly fewer breaks than the GNN, and its covariance stays consistent.
- **Fine separations.** No tracker approaches the oracle below 3σ. That gap is what association by
  truth used to hide.
- **Duplicate dropping.** With duplicate dropping on, all three lose real tracks in formations 3σ
  apart: GNN 3.1 tracks per object against 2.1 with it off. In cidgen it is still needed, because
  radar biases create duplicates. Tuning it is a separate question.

## Merged reports in the tracker: `vbm` (option A)

```yaml
tracking:
  method: vbm
  merge: {max_set: 6, p_unresolved_min: 0.02, merge_spread: 0.5, max_cluster_tracks: 10, protect_resolved_s: 30.0}
```

The GNN, JPDA and VB trackers all assume a report comes from at most one aircraft. When a radar cannot
resolve a group, it returns one report for several aircraft. The other members then go unexplained,
coast and die, and the tracker re-forms them later as new tracks.

`vbm` (`cidgen/trackers/merged.py`) is the VB tracker with one change to the assignment model: a report
may be explained by a **set** of tracks. Details are in the module docstring.

- **Allowed sets.** A set is allowed only if every pair in it could lie inside one resolution cell of
  that radar: range, azimuth and 1.5× azimuth in elevation, from the radar's specification. The
  probability is computed from the predicted positions and their covariances.
- **Scoring.** A merged report is scored at the members' mean position, with its noise widened by the
  members' spread.
- **Marginals stay exact.** The subset forward-backward takes "report → set" steps. It is checked
  against brute-force enumeration in `tests/test_soft_trackers.py`.
- **Mean-field update.** A member i of set S sees the pseudo-measurement |S|·z − Σ(other members'
  estimates). Each member is told where the others are, which holds the members apart. JPDA's mixture
  update pulls them together.
- **Protection from duplicate dropping.** Two tracks that took *different* reports in the same look in
  the last 30 s are two objects, and are never dropped as duplicates.

**Swarm bench** (`tracker_bench.run_swarm`). A counter-UAS radar (2° azimuth cell, 10 m range cell)
watches drones 4–6 km away, each wandering on its own slow loop. The radar merges echoes in one cell
exactly as the generator does. Six runs of 180 s each:

| Case (share of reports merged) | Own-track share: GNN / JPDA / VB / **VBM** | Tracks per drone over its life | Live tracks, mean |
|---|---|---|---|
| 4 drones, 60 m loops (10%) | 0.91 / 0.87 / 0.90 / **0.96** | 4.2 / 4.8 / 4.3 / **2.9** | 3.7 / 3.5 / 3.7 / **3.9** of 4 |
| 4 drones, 25 m loops (20%) | 0.74 / 0.70 / 0.73 / **0.85** | 5.3 / 5.7 / 5.4 / **3.8** | 3.0 / 2.9 / 3.0 / **3.5** of 4 |
| 8 drones, 25 m loops (40%) | 0.55 / 0.50 / 0.53 / **0.63** | 6.6 / 8.1 / 7.0 / **5.3** | 4.4 / 4.1 / 4.3 / **5.1** of 8 |

**Ablation: where VBM's gain comes from.** Each of VBM's two changes was switched off in turn:

| Own-track share | VBM | Without merged sets | Without duplicate protection | Neither |
|---|---|---|---|---|
| 4 drones, 25 m | 0.877 | 0.876 | 0.750 | 0.743 |
| 8 drones, 25 m | 0.647 | 0.652 | 0.538 | 0.531 |

Nearly all of the gain is the **protection of resolved tracks from duplicate dropping**, not the
merged-report hypotheses. Protection is not specific to VBM, so it is now a setting for every tracker:
`tracking.gnn.protect_resolved_s`, off by default. With protection on for every tracker (4 runs each):

| Own-track share (tracks per drone) | GNN | GNN + protection | JPDA + protection | **VB + protection** | VBM |
|---|---|---|---|---|---|
| 4 drones, 60 m | 0.92 (4.1) | 0.93 (3.9) | 0.95 (2.9) | 0.95 (3.4) | 0.95 (2.5) |
| 4 drones, 25 m | 0.76 (5.6) | 0.80 (4.4) | 0.83 (3.6) | **0.87** (3.7) | 0.84 (3.7) |
| 8 drones, 25 m | 0.55 (6.8) | 0.57 (6.1) | 0.61 (5.8) | **0.66** (5.6) | 0.65 (5.6) |

Once protection is on, soft association helps in swarms: VB gains about 10 points over the GNN, and
JPDA about 6. Earlier, that gain was being thrown away by duplicate dropping, which deleted the members
VB was holding. The merged-report sets add nothing beyond VB + protection.

### Protecting resolved tracks, on S12 (seed 20, perfect identity, doctrine ranker)

| Tracker | Real threats with own track | Covered by any track | Top-3 capture | Tracks per object | Duplicates dropped | Run time here |
|---|---|---|---|---|---|---|
| Label mode | 0.968 | 0.998 | 0.887 | 1 | 0 | 1 min |
| GNN | 0.785 | 0.976 | 0.802 | 5.8 | 243 | 1 min |
| GNN + protection | 0.826 | 0.980 | 0.805 | 5.3 | — | 1 min |
| VB | 0.782 | 0.970 | 0.804 | 6.1 | 982 | 20 min |
| **VB + protection** | **0.831** | 0.979 | 0.803 | **4.9** | 44 | 45 min |

On seeds 18–20, GNN + protection reaches 0.872 own-track and 0.859 top-3 capture, against 0.854 and
0.852 for the GNN.

What it shows:
1. **Protection is the one change worth keeping.** It costs nothing, works with every tracker and adds
   about 4–5 points of own-track share on seed 20.
2. **VB + protection holds tracks together best.** It has the fewest breaks of any tracker (4.9 tracks
   per object) and drops very few duplicates. But it beats GNN + protection by only 0.5 points of
   own-track share, at 45 times the run time.
3. **Top-3 capture doesn't move.** It stays at about 0.80 for every tracker. The threats that drive the
   top 3 are mostly already tracked, and the members gained are mostly swarm drones, which rank below
   the top 3.

VBM keeps more members as their own tracks, with fewer breaks. Its position error is about the same
(23–25 m against 20–27 m). What it cannot do is start a track for a member the radar has never
resolved, because no report ever stands for that member alone. The 8-drone swarm therefore still holds
only about 5 tracks for 8 drones.

## Group tracks: estimating members from RCS and nearby echoes (option B)

`cidgen/threat/group_count.py` estimates how many aircraft a track stands for. It looks at each
track's last 20 s of its own radar reports and uses two cues:

- **RCS fluctuation, k = mean² / variance of linear RCS.** The summed echo of n independently
  fluctuating members is steadier than one echo, and k needs no aircraft type or aspect.
- **Nearby echoes.** The mean number of echoes within 500 m in the same radar look. A partly resolved
  group throws extra echoes around its track.

A calibration table, fitted on training runs, maps (k, nearby echoes) to the expected member count.
The "true group size" used for calibration and scoring is the number of true objects whose nearest
track, within 500 m, is this one.

```
python -m cidgen.threat.group_count --fit <training runs> --test <test runs> --out results_gnn/group_count.json
python -m cidgen.threat.run_tp ... --group-count est --group-calib results_gnn/group_count.json --tag grp_est
python -m cidgen.threat.run_tp ... --group-count true --tag grp_true     # upper bound: true group sizes
```

With `--group-count`, each track's threat is multiplied by its member count, as the sum of similar
members. `run_tp` also reports two group-aware columns:
- `threat_captured3_cov`: the same capture as before, but crediting objects covered by a top-3 group
  track;
- `threat_mass3`: each track carries the summed threat of every object it covers, so top-3 tracks'
  total is divided by the best any 3 tracks could carry.

**S12, GNN tracker, calibrated on seeds 15–17 and tested on 18–20.** How well each cue separates
groups from single aircraft (AUC; 0.5 = chance, 1 = perfect):

| Cue | Group of 2 or more | 3 or more | 5 or more |
|---|---|---|---|
| RCS fluctuation alone | 0.57 | 0.58 | 0.60 |
| Nearby echoes alone | 0.83 | 0.80 | 0.73 |
| Both (the estimator) | **0.86** | **0.85** | **0.81** |

Mean estimated members: 1.47 for true singles, 2.30 for pairs, 2.72 for groups of 3–4 and 2.73 for
groups of 5 or more. Large groups are undercounted.

Threat list, perfect identity, doctrine ranker:

| Group weighting | Per-object top-3 capture | Group-credited capture | Threat mass in the top 3 |
|---|---|---|---|
| None (as before) | **0.851** | **0.894** | 0.809 |
| Estimated members | 0.842 | 0.884 | 0.786 |
| True members (upper bound) | 0.825 | 0.880 | **0.851** |

What it shows:
1. **RCS fluctuation is a real but weak cue on its own.** It adds most for big groups (AUC 0.73 to
   0.81 for 5 or more). Nearby echoes carry most of the signal.
2. **Weighting by members is a doctrine choice, not a free gain.** Even with perfect counts, it moves
   group tracks above dangerous single aircraft. Per-object capture drops 2.6 points while the group's
   threat mass in the top 3 rises 4 points. Whether 8 small drones outrank one striker is a doctrine
   decision.
3. **With today's estimate, weighting doesn't pay on either measure.** Singles in dense airspace get
   boosted and big swarms are undercounted. The estimate is better used as information than as a
   weight: a CID feature, or a "group, ~n" tag on the operator's track.

## Known limits of this first version

- False tracks never reach the threat list. CID and TP run on objects of interest only, so a clutter
  track that a real system might engage is not scored. A later option could keep false-track rows
  in the threat picture and score them.
- The tracker uses position only. Doppler is used only to guess the first tier, not in the gate or
  the update.
- There is no IMM. Each track uses one constant-velocity filter per tier.
