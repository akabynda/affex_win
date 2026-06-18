# Inference

`src/predict.py` runs the EXP-043 ensemble over a directory of `fold*_seed*/`
checkpoints and reports ensemble metrics.

```bash
uv run python src/predict.py \
  --checkpoints-dir logs/multiruns/EXP-043 \
  --test-csv data/test/testAB-clean.csv \
  --output predictions_testAB.csv
```

## What it does

1. Discovers every `fold*_seed*/` subdirectory under `--checkpoints-dir` (25 for EXP-043).
2. For each: loads `.hydra/config.yaml`, instantiates the datamodule + model, loads the
   best `epoch_*.ckpt`, and forwards the test set.
3. Averages predictions per UID across all folds (the ensemble).
4. Writes a `uid,target,pred` CSV and prints ensemble **MAE / Pearson / Spearman**.

## Flags

| Flag | Meaning |
|------|---------|
| `--checkpoints-dir` | Directory of `fold*_seed*/` runs (required) |
| `--test-csv` | Test set CSV — already leak-filtered (required) |
| `--output` | Where to write the `uid,target,pred` ensemble CSV (optional) |
| `--folds 0,1` | Restrict to a subset of folds (fast smoke test) |

The shipped `testAB-clean.csv` is pre-filtered, so there is **no** leak filtering at
inference time. Each run directory must contain `checkpoints/epoch_*.ckpt` **and**
`.hydra/config.yaml` — the latter is what the model is rebuilt from.

## Expected headline (CPU)

| Test set | N | Ensemble MAE |
|----------|---|--------------|
| testAB-clean | 103 | ≈ 1.40 |
| test-fabs | 70 | ≈ 1.40 |
