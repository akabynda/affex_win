# Data preparation

## What ships vs. what you fetch

- **In git:** the annotation CSVs (train / testAB-clean / test-fabs) and the fold
  assignments. See [`../data/README.md`](../data/README.md).
- **Downloaded separately, then unpacked by `make data`:** three archives.

| Archive | Contents | Where to download |
|---------|----------|-------------------|
| `affex-checkpoints.tar.gz` | the 25 EXP-043 models | this repo's GitHub **Releases** page |
| `affex-pdb.tar.gz` | structures (~1.5 GB) | out-of-band link (see the release notes) |
| `affex-esm.tar.gz` | ESM2 embeddings (~4.7 GB) | out-of-band link |

Download the archive(s) you need into the repo root, then unpack:

```bash
make data        # extracts every affex-*.tar.gz present
```

`make data` is just a convenience wrapper around `tar xzf`; the archives extract to their
repo-relative paths (`logs/multiruns/EXP-043/…`, `data/raw/ppb-affinity/{pdb,esm}/…`).

## ESM2 embeddings from scratch

The tarball already ships `data/raw/ppb-affinity/esm/<uid>.pt`. You only need this to
embed **new** PDB structures. The model uses per-residue `esm2_t33_650M_UR50D`
embeddings (layer 33), read directly as `.pt` files.

```bash
# 1. Clone the ESM repo (weights auto-download on first run).
git clone https://github.com/facebookresearch/esm
export ESM_MODEL_DIR=$PWD/esm

# 2. Extract embeddings for every .pdb in a directory.
uv run python scripts/data/run_esm_extraction.py \
  <pdb_dir> \
  --savedir data/raw/ppb-affinity/esm \
  --workers 2
```

`run_esm_extraction.py` resolves the model directory from `--esm-model-dir`, falling
back to the `ESM_MODEL_DIR` environment variable; it errors if neither is set. It
writes one `<uid>.pt` per structure and skips any that already exist.

## Annotation CSV columns

Each annotation row describes one complex: `uid`, `receptor_chains`, `ligand_chains`,
and the measured affinity (`KD` / `dG`). The composite key
`uid_receptorchains_ligandchains` is what fold assignments and predictions are keyed on.
