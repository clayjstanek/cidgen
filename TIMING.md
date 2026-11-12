# Timing the trackers (`--debug 1`)

`--debug 1` turns on section timers in the trackers (gnn, jpda, vb, vbm). Any other value, or leaving
it out, means no timing: the timers become do-nothing calls and the results are identical.

## Two ways to run it

Quick loop, tracker only (world, radar, tracker; no features or labels):

```
python -m cidgen.trackers.run_tracker --scenario S12_vb --seed 20 --horizon 300 --debug 1 --out timing
```

Inside a full generate run:

```
python -m cidgen.generate --scenario S12_vb --seed 20 --out runs_vb --debug 1
```

## What it writes

| File | Contents |
|---|---|
| `<scenario>_seed<N>_timingLog_<YYYYMMDD_HHMMSS>.csv` | One line per timed block as it finishes: `wall_time` (ISO date and time, µs), `sim_time_s`, `section`, `dt_ms` |
| `..._summary.txt` | The summary table, also printed to the screen |
| `..._summary.csv` | The same table as CSV |

**Log size.** The log gets one line per timed block, about 1,400 lines per simulated second for VB on
S12: about 170,000 lines for 120 s, and about 2.5 million (about 150 MB) for a full 1,800 s run. Use
`--horizon` while you iterate.

**Summary columns.** For each section: calls, total s, share of the tracker's wall time, and mean and
max ms per call. The two update cycles, `look` (one radar look) and `snapshot` (one feature-grid time),
also get p50 and p95 ms. Indentation shows nesting.

## Sections timed now

```
look                         one radar look
  expire / predict           coast deletion; predict all live tracks
  soft                       confirmed tracks (jpda / vb / vbm)
    gate / cluster / build   chi-square gate; connected clusters; score matrix
    marginals                assignment marginals (exact subset DP or BP)
    update                   state updates
    labels+buffer            report labels; store the look for the VB window
  tentative / births / confirm
snapshot                     one feature-grid time
  window                     VB window step
    clusters                 which tracks shared reports in the window
    solve                    one cluster's coordinate ascent
      setup / assoc_scores / assoc_marginals / smoother
  output                     predict to the grid time, drop duplicates
```

## Going deeper on a bottleneck

Wrap the code inside it in another named block. The name's "/" puts it under its parent in the report:

```python
with self.T("snapshot/window/solve/smoother/rts_backward"):
    ...
```

Nothing else needs to change.

## First survey (S12 seed 20, VB, first 120 s, 2-core cloud box)

| Section | Share of tracker time |
|---|---|
| snapshot/window/solve/assoc_marginals | 57% (74,300 calls, 0.36 ms mean) |
| snapshot/window/solve/smoother | 24% |
| snapshot/window/solve/assoc_scores | 11% |
| look (all of the per-look filter step) | 5% |

The VB window step is 95% of VB's run time. Inside it, the cost is many small exact-marginal calls,
roughly one per look per cluster per iteration, rather than any single expensive call.

## Iteration sweep: VB + protection (`soft.iterations`)

**S12 seed 20, first 900 s:**

| Iterations (most allowed) | Tracker time | Iterations actually used per solve | Coverage | Tracks per object | Swarm coverage | Median position error |
|---|---|---|---|---|---|---|
| 1 | **319 s** | 1.00 | 0.841 | 1.82 | 0.376 | 82.8 m |
| 2 | 480 s | 2.00 | 0.842 | 1.78 | 0.376 | 83.1 m |
| 3 | 773 s | 2.78 | 0.841 | 1.78 | 0.368 | 84.8 m |
| 5 | 1,173 s | 4.10 | 0.841 | 1.78 | 0.368 | 84.3 m |

**Swarm bench (4 runs each, 180 s):** the own-track share is flat across iteration counts: 0.85–0.87 for
4 drones and 0.654–0.657 for 8 drones. Position error *rises* slightly with more iterations, and time
rises 3–4×.

**Conclusion.** Extra coordinate-ascent iterations buy nothing measurable here. One pass (assignment
probabilities from the filtered states, then one smoother pass) gives the same tracking at about 3.7×
less run time. `S12_vbp` now uses `soft: {iterations: 1}`. The code default stays at 5, so earlier VB
scenarios reproduce unchanged.
