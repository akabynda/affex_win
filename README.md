# affex

Structure-based prediction of protein–protein binding free energy (ΔG), using the
PCANN model: an ESM2-conditioned residue-interface graph network. This repository
ships the trained EXP-043 ensemble (25 cluster-aware cross-validation folds), the
data needed to run and reproduce it, and the training/inference entry points.

## Requirements

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) (dependency manager)
- ~7 GB free disk for the artifacts (≈1.5 GB structures, ≈4.7 GB ESM embeddings, ≈0.1 GB checkpoints)

## Install

Two mutually-exclusive install profiles:

```bash
make install            # uv sync --extra cpu --no-dev    — macOS, or Linux without a GPU (default)
make install-cu128      # uv sync --extra cu128 --no-dev  — Linux + NVIDIA (CUDA 12.8 wheels)
```

For CUDA, the `cu128` wheels bundle the CUDA 12.8 runtime — the host needs only a
recent NVIDIA driver, not a system CUDA toolkit. To target a different CUDA version,
swap `cu128` → `cu126`/`cu129` in `pyproject.toml`. (For `torch 2.8.0`, PyG ships
extension wheels only for `cpu`, `cu126`, `cu128`, `cu129`.)

## Get data & checkpoints

The artifacts ship as three separate archives:

| Component | Contents | Hosted |
|-----------|----------|--------|
| `checkpoints` | the 25 EXP-043 models | GitHub Release asset |
| `pdb` | structures (large) | out-of-band — download link distributed separately |
| `esm` | precomputed ESM2 embeddings (large) | out-of-band |

Download the archives (see [`docs/data-prep.md`](docs/data-prep.md) for the exact links),
drop them in the repo root, then:

```bash
make data        # unpacks any downloaded affex-*.tar.gz in place
```

This populates `logs/multiruns/EXP-043/` and `data/raw/ppb-affinity/{pdb,esm}/`. You only
need the components you'll use: `checkpoints` for inference, plus `pdb`/`esm` for the
test/train UIDs.

## Run inference

```bash
make infer       # 25-fold ensemble on testAB-clean -> predictions_testAB.csv
```

or directly, for either test set:

```bash
uv run python src/predict.py \
  --checkpoints-dir logs/multiruns/EXP-043 \
  --test-csv data/test/testAB-clean.csv \
  --output predictions_testAB.csv          # add --folds 0,1 for a fast subset
```

Expected ensemble headline (CPU): **testAB-clean (N=103) MAE ≈ 1.40**,
**test-fabs (N=70) MAE ≈ 1.40**.

## Reproduce EXP-043 training

```bash
make train       # 25-fold multirun, seed 42, CPU
```

EXP-043 ran on CPU (`trainer.accelerator: cpu`); a GPU only speeds it up. Exact
numeric reproduction of the headline is a CPU statement.

## ESM embeddings from scratch (optional)

The `esm` archive already ships `data/raw/ppb-affinity/esm/`; this is only needed to embed
*new* PDBs. See [`docs/data-prep.md`](docs/data-prep.md).

```bash
git clone https://github.com/facebookresearch/esm
export ESM_MODEL_DIR=$PWD/esm     # weights esm2_t33_650M_UR50D auto-download on first run
uv run python scripts/data/run_esm_extraction.py \
  data/raw/ppb-affinity/pdb --savedir data/raw/ppb-affinity/esm --workers 2
```

## Test

```bash
make test        # install + unpack + 2-fold inference + 1-epoch training smoke test
```

## Docs

- [`docs/data-prep.md`](docs/data-prep.md) — data layout, ESM extraction
- [`docs/inference.md`](docs/inference.md) — `predict.py` contract
- [`docs/training.md`](docs/training.md) — reproducing EXP-043

## Citation / License

Released under the [MIT License](LICENSE).

<!-- TODO: add citation (paper / DOI) once published. -->
