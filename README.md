# affex_win

Windows/GPU-friendly fork of [`norsage/affex`](https://github.com/norsage/affex).

affex predicts protein-protein binding free energy (dG) with the PCANN model:
an ESM2-conditioned residue-interface graph network. This fork keeps the
upstream EXP-043 inference path, but adds practical support for running it on
Windows with an NVIDIA GPU.

## What This Fork Adds

- `src/predict.py --device auto|cpu|cuda`, so inference can run on CUDA.
- Windows-compatible ESM extraction through the active Python interpreter.
- Optional GPU ESM extraction instead of forcing `--nogpu`.
- `scripts/data/run_esm_extraction_inprocess.py`, which loads ESM2 once and
  generates embeddings much faster for selected CSVs.
- `.gitignore` entries for local ESM clones, logs, downloaded archives, raw
  embeddings, checkpoints, and prediction CSVs.

## Tested Setup

- Windows
- Python 3.12
- NVIDIA GeForce RTX 3070 Laptop GPU
- PyTorch `2.8.0+cu128`
- CUDA runtime `12.8` bundled by the PyTorch wheel
- PyG CUDA wheels from `https://data.pyg.org/whl/torch-2.8.0+cu128.html`

## Requirements

- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/)
- Recent NVIDIA driver for CUDA inference
- Enough disk for model/data artifacts

The upstream project declares `aim>=3.29.1`, but `aimrocks` wheels are not
available for Windows/Python 3.12. For inference, `aim` is not required, so this
README installs only the runtime dependencies needed to predict.

## Install on Windows + CUDA

From the repo root:

```powershell
py -3.12 -m pip install uv
py -3.12 -m uv venv --python 3.12 .venv

py -3.12 -m uv pip install --python .venv\Scripts\python.exe `
  torch==2.8.0 `
  --index-url https://download.pytorch.org/whl/cu128

py -3.12 -m uv pip install --python .venv\Scripts\python.exe `
  "torch-geometric>=2.6.1,<2.8" `
  torch-cluster==1.6.3 `
  torch-scatter==2.1.2 `
  torch-sparse==0.6.18 `
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

py -3.12 -m uv pip install --python .venv\Scripts\python.exe `
  pandas scipy gemmi hydra-core loguru jaxtyping lightning torchmetrics omegaconf fair-esm

py -3.12 -m uv pip install --python .venv\Scripts\python.exe -e . --no-deps
```

Verify CUDA:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Data and Checkpoints

For release `v0.1.0rc`, the GitHub release assets are:

- `pcann_v2.tar.gz`: EXP-043 checkpoints and Hydra configs
- `pcann_v2-pdb.tar.gz`: PDB structures

Download them from:

https://github.com/norsage/affex/releases/tag/v0.1.0rc

Then extract in the repo root:

```powershell
tar -xzf pcann_v2.tar.gz
tar -xzf pcann_v2-pdb.tar.gz
```

This creates:

```text
logs/multiruns/EXP-043/
data/raw/ppb-affinity/pdb/
```

The ESM embeddings are not included in those two release assets, so generate
them locally.

## Generate ESM Embeddings

For the shipped test sets only:

```powershell
.venv\Scripts\python.exe scripts\data\run_esm_extraction_inprocess.py `
  data\raw\ppb-affinity\pdb `
  --savedir data\raw\ppb-affinity\esm `
  --csv data\test\testAB-clean.csv `
  --csv data\test\test-fabs.csv `
  --device cuda
```

This loads `esm2_t33_650M_UR50D` once, downloads the weights on first run, and
saves embeddings under:

```text
data/raw/ppb-affinity/esm/
```

You can also use the upstream-style extractor:

```powershell
git clone https://github.com/facebookresearch/esm
.venv\Scripts\python.exe scripts\data\run_esm_extraction.py `
  data\raw\ppb-affinity\pdb `
  --savedir data\raw\ppb-affinity\esm `
  --esm-model-dir esm
```

Add `--cpu` to force CPU extraction.

## Run Inference

Full 25-fold ensemble on `testAB-clean`:

```powershell
.venv\Scripts\python.exe src\predict.py `
  --checkpoints-dir logs\multiruns\EXP-043 `
  --test-csv data\test\testAB-clean.csv `
  --output predictions_testAB.csv `
  --device cuda
```

Full 25-fold ensemble on `test-fabs`:

```powershell
.venv\Scripts\python.exe src\predict.py `
  --checkpoints-dir logs\multiruns\EXP-043 `
  --test-csv data\test\test-fabs.csv `
  --output predictions_test_fabs.csv `
  --device cuda
```

Fast smoke test with two folds:

```powershell
.venv\Scripts\python.exe src\predict.py `
  --checkpoints-dir logs\multiruns\EXP-043 `
  --test-csv data\test\testAB-clean.csv `
  --output predictions_testAB_smoke.csv `
  --folds 0,1 `
  --device cuda
```

## Verified Results

On the tested Windows CUDA setup:

| Test set | Models | N | MAE | Pearson | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| `testAB-clean.csv` | 25 | 103 | 1.4036 | 0.4867 | 0.4747 |
| `test-fabs.csv` | 25 | 70 | 1.3994 | 0.3651 | 0.2921 |
| `testAB-clean.csv` smoke | 2 | 103 | 1.4271 | 0.4428 | 0.4540 |

## Experimental PLM-interact Embeddings

This fork also includes an experimental path inspired by
[`liudan111/PLM-interact`](https://github.com/liudan111/PLM-interact): encode
both sides of a complex jointly with a PLM-interact-tuned ESM2 encoder, then
split the residue embeddings back by chain and train the same PCANN graph model.

Install the extra dependencies:

```powershell
py -3.12 -m uv pip install --python .venv\Scripts\python.exe transformers huggingface-hub
```

Generate pair-aware embeddings for train and test CSVs:

```powershell
.venv\Scripts\python.exe scripts\data\run_plm_interact_extraction.py `
  data\raw\ppb-affinity\pdb `
  --savedir data\raw\ppb-affinity\plm_interact `
  --csv data\train\pcann-plus-trainval.csv `
  --csv data\test\testAB-clean.csv `
  --csv data\test\test-fabs.csv `
  --checkpoint-repo danliu1226/PLM-interact-650M-Leakage-Free-Dataset `
  --skip-too-long `
  --device cuda
```

For complexes with multiple chains on one side, such as antibody Fab entries,
the extractor concatenates all receptor chains as protein 1 and all ligand
chains as protein 2, inserting `X` between chains by default. The saved files
are keyed by the exact chain assignment, for example
`uid_receptorchains_ligandchains.pt`, so different chain groupings from the same
PDB can coexist.

Optional flags:

- `--chain-separator ""` disables the linker between chains.
- `--chain-separator GGGGSGGGGS` uses a flexible-linker-like join.
- `--bidirectional-average` averages embeddings from receptor-ligand and
  ligand-receptor orderings.
- `--checkpoint-repo danliu1226/PLM-interact-650M-humanV12` switches encoder
  checkpoint.
- `--skip-too-long` skips complexes above `--max-length` instead of aborting.

With the default `--max-length 1603`, the current release data leaves out 20
chain assignments: 16 train rows, 1 `testAB-clean` row, and 3 `test-fabs` rows.
So this first PLM-interact baseline evaluates on `testAB-clean` N=102 and
`test-fabs` N=67 unless a longer-context or truncation strategy is added.

Train PCANN on those embeddings:

```powershell
.venv\Scripts\python.exe src\train.py +experiment=pcann_reimpl-plm-interact-mc10 `
  datamodule.val_fold=0 seed=42 trainer.accelerator=gpu
```

The provided config assumes a 650M encoder with `node_feature_dim: 1280`. If you
try a 35M PLM-interact/base ESM2 model, set `lightning.model.node_feature_dim=480`.

## Linux / Upstream Workflow

The original upstream workflow is still available for Linux/macOS:

```bash
make install
make install-cu128
make infer
```

See the upstream docs for training and full reproduction details:

- [`docs/data-prep.md`](docs/data-prep.md)
- [`docs/inference.md`](docs/inference.md)
- [`docs/training.md`](docs/training.md)

## License

Released under the [MIT License](LICENSE), following the upstream project.
