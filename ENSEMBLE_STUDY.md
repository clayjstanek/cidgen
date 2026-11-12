# Ensemble study: can combining models keep the neural gains without the shift failures?

Under the S0 → S9 shift, XGBoost transferred almost perfectly while the GNN and GRU called 40–48% of
our own drones hostile. This study asks whether a combination does better than any single model,
and in particular whether the fielded Bayesian Network plus a GNN beats the Bayesian Network alone.

## Rule that keeps it honest

Every combination is fitted or chosen on the **validation runs of the training scenario (S0)** and then
frozen. The test scenario (S9) is used only for scoring, exactly as for single models. That is why the
validation predictions of every model are needed, which earlier versions did not save.

One caveat: the neural models used the same validation runs to decide when to stop training and to
calibrate, so their validation predictions are slightly optimistic. Stacking may therefore lean a
little too much on them. The test scores are unaffected by this and are the numbers to judge by.

## Combinations

BN, TREE and NN are the members: BN-learned, XGB-own and, by default, GNN-NOTQ (choose others with
`--bn`, `--tree`, `--neural`).

| Name | What it does | Question it answers |
| --- | --- | --- |
| `Mean[...]` | Average of the members' probabilities | Simplest probabilistic combination |
| `LogPool[...]` | Normalised geometric mean; a confident "no" from one member outweighs a "yes" | Combination that resists one member's confident error |
| `Stack[...]` | Logistic regression on the members' log probabilities, fitted on validation | Learned weighting; `_stack_weights` shows how much each member counts |
| `Gate[tier]` | Per tier (from speed, altitude, RCS), the member with the best validation log loss | Selecting a model before the data reaches it |
| `NN\|TREE if OOR` | The neural model, replaced by the tree model where track-quality inputs exceed the training range | A fallback rule aimed at the known failure mode |
| `Stack[BN+NN]\|TREE if OOR` | The same fallback applied to the BN + GNN stack | |

Member sets: BN+NN (keep the fielded BN, add a GNN), TREE+NN, BN+TREE, BN+TREE+NN, and all models.

The `_unknown` file restores the five-state idea: a track is declared Unknown / Pending when the
members disagree (or disagree about Hostile), or when the combined confidence is below 0.6, 0.8 or 0.9.
For each rule it reports how many tracks become Unknown, friends and own drones still called hostile,
and hostiles called hostile or left Unknown.

## Commands (Windows, from the folder that holds `cidgen`; `%CACHE%` is your cache folder)

```
:: 0. checks (the first needs no GPU; about a minute each)
python tests\test_ensemble.py
python tests\test_gnn_quick.py

:: 1. track-quality-free neural models for T0, if not yet trained (v0.5 also saves validation predictions)
python -m cidgen.models.train_gnn --runs "runs/S0_baseline_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache %CACHE% --drop-features track-quality
python -m cidgen.models.train_gnn --runs "runs/S0_baseline_*" --test-runs "runs/S9_sensor_shift_*" --out results_transfer --cache %CACHE% --drop-features track-quality --no-graph

:: 2. validation predictions for models trained before v0.5 (no retraining; seconds per model)
python -m cidgen.models.predict "results_v3/*_S0_baseline__on__S9_sensor_shift.pt" "results_transfer/*_S0_baseline__on__S9_sensor_shift.pt" --cache %CACHE% --out results_transfer --check-test

:: 3. BN and tree predictions on validation and test runs (about 10 minutes)
python -m cidgen.models.compare --runs "runs/S0_baseline_*" --test-runs "runs/S9_sensor_shift_*" --preds results_transfer --out results_transfer --save-preds

:: 4. ensembles (a minute or two each)
python -m cidgen.models.ensemble --dir results_transfer --name S0_baseline__on__S9_sensor_shift
python -m cidgen.models.ensemble --dir results_transfer --name S0_baseline__on__S9_sensor_shift --neural GNN
python -m cidgen.models.ensemble --dir results_transfer --name S0_baseline__on__S9_sensor_shift --neural GNN-SD30
```

Step 2 must print `OK` for every model. `MISMATCH` means the model could not be reloaded exactly;
do not use its validation file, and tell me which model it was.

## Send back

All `ensemble_S0_baseline__on__S9_sensor_shift_*` files, the new `compare_S0_baseline__on__S9_sensor_shift*`
files, and one zip of `allpred_S0_baseline__on__S9_sensor_shift.*`, `ranges_*`, and every
`valpred_*` and `pred_*` file for that name. With the prediction files I can test further
combinations myself without another round trip.
