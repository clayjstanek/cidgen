# Runbook: VB tracker tests and version control

Run everything from the folder that holds `cidgen` (`cidgen_v0.6.1_hypersonic_GNN`), in `pytorch_new`.

## What is new

| File | What it does |
|---|---|
| `tests/test_vb_tracker.py` | 16 fast VB tests (about 2 s), no scenario needed |
| `tests/vb_golden.py` | Makes or checks the pinned VB result on a real S12 seed |
| `tests/test_vb_regression.py` | The pinned S12 check as a test (slow, runs only when asked) |
| `cidgen/trackers/soft.py` | The VB window smoother is now a separate function, `soft_rts`, so it can be tested directly. Results are bit-identical to before (checked on S12_vbp and S12_vbm, seed 20, 120 s) |
| `.gitignore` | Keeps generated runs, results, models and zips out of git |

`tests/test_vb_tracker.py` checks:

| Group | Tests | Against |
|---|---|---|
| Pseudo-measurement | `_pseudo` identities: certain report = the report; weight b = noise R / b; equal split = midpoint | Closed form |
| Window smoother | `soft_rts` with certain associations, missed looks, fractional and spread weights (9 cases) | An independent batch solution: the joint Gaussian posterior of all states, solved at once in information form |
| Window smoother | The newest look's smoothed state (where the tracker restarts) is the filtered state | Batch solution of the data up to that look |
| Causality | Output at every grid time up to 60 s is bit-identical when the reports after 60 s are replaced, or cut off | Itself (the test fails if the tracker ever sees future reports; checked by injecting such a leak) |
| Causality | Two runs are bit-identical | Itself |
| Window | Where tracks share reports, the window solves and moves estimates; isolated tracks are never re-solved and equal the filter step | `iterations: 0` (filter step only) |
| Track management | VB starts, confirms, coasts and deletes tracks exactly as the GNN (same first, confirm and last times, same report-to-track labels) | GNN tracker |

## V0. Unzip, then confirm nothing moved (about 1 min)

| Step | Command | Expect |
|---|---|---|
| V0.1 | `python -m pytest tests\test_vb_tracker.py -q` | `16 passed` |
| V0.2 | `python -m pytest tests\test_gnn_tracker.py tests\test_soft_trackers.py tests\test_mfa_stonesoup.py -q` | `11 passed` (8 + 3 Stone Soup) |

## V1. Put the folder under git (once)

Do this before making the golden file, so the pinned result is tied to a commit.

| Step | Command |
|---|---|
| V1.1 | `git --version` (if missing: `conda install -c conda-forge git`) |
| V1.2 | `git init` |
| V1.3 | `git add .` then `git status` and check no runs, results, `.pt` or zips are listed |
| V1.4 | `git commit -m "cidgen v0.6.1 hypersonic + trackers (GNN, JPDA, VB, VBM, MFA), ABM/MARV classes, VB tests"` |
| V1.5 | `git tag vb-baseline` |

From then on: `git status` shows what changed, `git diff` shows how, and `git checkout vb-baseline -- <file>` restores a
file. Commit before and after each change you want to be able to return to.

## V2. Pin the VB result on S12 (once, about 1 min)

| Step | Command | Result |
|---|---|---|
| V2.1 | `python tests\vb_golden.py` | Writes `tests\golden\vb_S12_vbp_seed20_h120.json` from S12_vbp, seed 20, first 120 s, on your scenario files |
| V2.2 | `git add tests\golden` and `git commit -m "pin VB result"` | |
| V2.3 | `set CIDGEN_SLOW=1` then `python -m pytest tests\test_vb_regression.py -q` | `1 passed` |

In PowerShell, V2.3 is `$env:CIDGEN_SLOW=1; python -m pytest tests\test_vb_regression.py -q`. Without
`CIDGEN_SLOW=1`, the test is skipped, so a normal `pytest` run stays fast.

The golden file is made on your machine with your scenario files (including the ABM and MARV classes) on purpose:
results can differ in the last digits between machines and change whenever the scenario does. It stores a fingerprint
of the resolved scenario: if the scenario files change later, the test skips with "scenario changed" instead of
failing. Re-make it (V2.1) only when a change is meant to move the VB results, and say so in the commit.

`python tests\vb_golden.py --check` compares without pytest and lists every difference. Other seeds or lengths:
`python tests\vb_golden.py --scenario S12_vb --seed 18 --horizon 300`.

## Send back

The V0 output, and the V2.3 output.
