# Reproducing EXP-043

EXP-043 is a 25-fold cluster-aware cross-validation ensemble (single seed, 42),
trained on CPU.

```bash
uv run python src/train.py --multirun \
  +experiment=pcann_reimpl-mc10 \
  datamodule.val_fold=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 \
  seed=42 group=EXP-043 \
  datamodule.train_csv=data/train/pcann-plus-trainval.csv \
  datamodule.folds_csv=data/train/folds_pcann_plus_ppisplit_mc25.csv
```

`make train` runs exactly this. It writes 25 models to
`logs/multiruns/EXP-043/fold{N}_seed42/`.

## Configuration (`pcann_reimpl-mc10`)

- ESM residue-interface graphs
- 2 GNN layers, `node_feature_dim = 1280`
- Adam, `lr = 1e-3`, `weight_decay = 3e-3`
- `max_epochs = 20`, early stopping (patience 20)
- `trainer.accelerator: cpu` (GPU only speeds it up; exact numeric reproduction is a
  CPU statement)

## GPU

After `make install-cu128` on a Linux+NVIDIA host, add `trainer.accelerator=gpu` to
the command above.

## Evaluating the result

```bash
uv run python src/predict.py \
  --checkpoints-dir logs/multiruns/EXP-043 \
  --test-csv data/test/testAB-clean.csv
```

Expected ensemble headline: testAB-clean (N=103) MAE ≈ 1.40, test-fabs (N=70) MAE ≈ 1.40.
