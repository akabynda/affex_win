# Data layout

## Shipped in git (small, this repo)

```
data/
├── train/pcann-plus-trainval.csv               # 1268 rows / 1256 UIDs — training source
├── train/folds_pcann_plus_ppisplit_mc25.csv    # 25-fold cluster-aware CV assignments for EXP-043
├── test/testAB-clean.csv                        # 103 rows — primary test set (pre-filtered, leak-free)
└── test/test-fabs.csv                           # 70 rows — antibody-Fab test set (leak-free)
```

`testAB-clean.csv` is shipped pre-filtered — no leak removal happens at inference time.

## Added by `make data` (large, out-of-band — gitignored)

Download three independent archives (checkpoints from the GitHub Releases page; pdb + esm
from the out-of-band link in the release notes), drop them in the repo root, and
`make data` unpacks them in place:

```
# checkpoints — GitHub Release asset
logs/multiruns/EXP-043/fold{0..24}_seed42/
├── checkpoints/epoch_*.ckpt
└── .hydra/config.yaml   # required — predict.py rebuilds the model from this

# pdb + esm — large; hosted out-of-band (download link distributed separately)
data/raw/ppb-affinity/
├── pdb/<uid>.pdb        # structures for the train + test UIDs
└── esm/<uid>.pt         # precomputed ESM2 (esm2_t33_650M_UR50D) per-residue embeddings
```

Download only the components you need: `checkpoints` for inference, plus `pdb`/`esm`.

To embed *new* structures yourself, see [`../docs/data-prep.md`](../docs/data-prep.md).
