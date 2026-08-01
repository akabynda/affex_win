# Current 25-fold GPU protocol

All comparable experiments use 25-fold cluster-aware cross-validation, seed 42,
one GPU and a physical batch size of 32.

```bash
uv run python src/train.py --multirun \
  +experiment=pcann_reimpl-mc10 \
  datamodule.val_fold=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 \
  seed=42 group=GPU-pcann-42 \
  datamodule.train_csv=data/train/pcann-plus-trainval.csv \
  datamodule.folds_csv=data/train/folds_pcann_plus_ppisplit_mc25.csv
```

The run writes 25 models to
`logs/multiruns/GPU-pcann-42/fold{N}_seed42/`.

## Configuration (`pcann_reimpl-mc10`)

- ESM residue-interface graphs
- 2 GNN layers, `node_feature_dim = 1280`
- Adam, `lr = 1e-3`, `weight_decay = 3e-3`
- `max_epochs = 20`, early stopping (patience 20)
- `trainer.accelerator: gpu`, physical `batch_size = 32`

## Shared protocol for comparable experiments

All standard PCANN ablations load
`configs/training_protocol/pcann_shared_gpu.yaml` after their experiment config.
That file is the single source of truth for the dataset paths, seed, real batch
size, worker count, trainer settings, early stopping and optimizer/scheduler.
Change those settings there, not in an individual experiment YAML. Experiment
tracking is disabled (`logger: null`) so it cannot change or block training.

`src/train.py` validates the resolved configuration before allocating a model.
A one-off command-line override such as `datamodule.batch_size=64` therefore
fails instead of silently producing an incomparable run. Run
`python scripts/validate_training_protocols.py` to compose and validate every
experiment config.

End-to-end experiments that cannot use this protocol must set
`training_protocol.enforce: false` and state why they are non-comparable. The
ESM-2 LoRA-tail experiment is currently the only such exception.

## Evaluating the result

```bash
uv run python src/predict.py \
  --checkpoints-dir logs/multiruns/GPU-pcann-42 \
  --test-csv data/test/testAB-clean.csv
```

Expected ensemble headline: testAB-clean (N=103) MAE ≈ 1.40, test-fabs (N=70) MAE ≈ 1.40.
