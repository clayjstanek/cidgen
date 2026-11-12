# Generator validation — v0.1 build, 2026-11-02

Runs: S0 and S1, seeds 1–4 each, 30-minute horizon. Reproduce with
`python -m cidgen.validate runs/S0_baseline_seed{1..4} runs/S1_zero_relational_seed{1..4} --determinism S0`.

## 1. Radar residuals vs. configured sigma (S0 seed 1; unmerged, unjammed reports)

| Radar | Reports | Range residual std (m) | Stated σr (m) | Az residual std (°) | Stated σaz (°) |
| --- | --- | --- | --- | --- | --- |
| R_C-1 | 42,631 | 3.0 | 3.0 | 0.502 | 0.50 |
| R_C-2 | 15,410 | 3.0 | 3.0 | 0.497 | 0.50 |
| R_P-1 | 30,302 | 15.0 | 15.0 | 0.150 | 0.15 |
| R_S-1 | 13,996 | 30.1 | 30.0 | 0.399 | 0.40 |
| R_S-2 | 12,960 | 29.8 | 30.0 | 0.403 | 0.40 |

Per-radar registration biases appear as medians (e.g. R_S-1 az +0.24°, R_S-2 −0.28°), as configured.
R_S-1 Pd for class A stays 0.91–0.96 out to 60 km and falls to about 0.8 at 90–105 km, against a
design point of 0.86 at ~99 km for mean-aspect RCS; jamming, aspect and horizon account for the dips.

## 2. Leakage audit

A classifier given only entity IDs scores at chance (balanced accuracy 0.333 on three classes) in all
eight runs. IDs are drawn from a shuffled pool, so creation order carries no allegiance information.

## 3. Relational check (pooled over 4 seeds; split by run and group)

Gradient-boosted trees on per-track features alone, versus per-track plus mean neighbour features.

| Scenario | Own features: bal. acc / log loss | + neighbours: bal. acc / log loss | Log-loss gain |
| --- | --- | --- | --- |
| S0 baseline | 0.959 / 0.243 | 0.962 / 0.199 | **+0.044** (18% lower) |
| S1 zero-relational | 0.958 / 0.269 | 0.957 / 0.278 | −0.009 (none) |

Neighbours carry information in S0 and none in S1, as designed: the ablation is clean.

**Finding:** per-track evidence already reaches ~96% balanced accuracy in S0, so the default scene is
too easy to separate model families on accuracy. The relational gain shows up in calibration
(log loss) rather than accuracy. Separating the GNN from a BN will need the harder scenarios — S4
(weak package members, rho_variant 0.9, emcon), S7 (deception) and S11 (swarm among birds) — or a
harder default (e.g. rho_variant 0.7, p_ESM_correct 0.6).

## 4. Determinism

Two runs of S0 with the same seed produce byte-identical tables (hash 5212aa400f288248).

---

# v0.3 local check — 2026-11-03

Small local sample (S0 8 seeds, S1 and S7 6 seeds, S9 4 seeds; one test run per scenario, so wide
intervals). Purpose: confirm the scenarios are off the ceiling and the ablation is still clean.

| Check | v0.2 | v0.3 |
| --- | --- | --- |
| BN-learned balanced accuracy, S0 / S1 / S7 | 0.986 / 0.991 / 0.932 | 0.967 / 0.966 / 0.932 |
| Relational log-loss gain from neighbours, S0 | +0.044 | +0.022 |
| Relational log-loss gain from neighbours, S1 (must be ~0) | −0.009 | +0.001 |

Shift test, trained on S0 and tested on S9 (4 test runs, 339 aircraft):

| Model | Balanced accuracy [95%] | Friends called hostile [95%] |
| --- | --- | --- |
| BN-learned | 0.920 [0.894, 0.944] | 0.125 [0.065, 0.200] |
| BN-oracle (fitted with the S9 data) | 0.940 [0.916, 0.960] | 0.113 [0.057, 0.183] |
| Gradient-boosted trees | 0.980 [0.967, 0.990] | 0.033 [0.008, 0.063] |

The fixed-structure BN loses most where evidence reliability depends on context it cannot represent
(IFF outages, where the track's earlier valid reply matters) and under sensor shift.
