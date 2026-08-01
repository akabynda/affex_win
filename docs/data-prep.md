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

All experiments use the same local Hugging Face checkpoint at
`models/esm2_t33_650M_UR50D`. The model uses its per-residue final-layer
embeddings, read directly as `.pt` files.

```bash
# Extract embeddings for every selected structure.
uv run python scripts/data/run_esm_extraction_inprocess.py \
  <pdb_dir> \
  --savedir data/raw/ppb-affinity/esm2_hf_per_chain \
  --model-name models/esm2_t33_650M_UR50D
```

The extractor writes one `<uid>.pt` per structure and skips any that already exist.

## Annotation CSV columns

Each annotation row describes one complex: `uid`, `receptor_chains`, `ligand_chains`,
and the measured affinity (`KD` / `dG`). The composite key
`uid_receptorchains_ligandchains` is what fold assignments and predictions are keyed on.
