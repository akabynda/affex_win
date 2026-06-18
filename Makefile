# affex release — install / data / inference / training / smoke test
.PHONY: install install-cu128 data infer train test

# Default install profile (macOS, or Linux without a GPU).
# --no-dev: skip the dev tooling group (pytest/ruff/pyright/kaleido), which uv would
# otherwise install by default. The shipped uv.lock still validates (it's an install flag).
install:
	uv sync --extra cpu --no-dev

# Linux + NVIDIA (CUDA 12.8 wheels). Then run with trainer.accelerator=gpu.
install-cu128:
	uv sync --extra cu128 --no-dev

# Unpack any artifact archives you've downloaded into the repo root (see docs/data-prep.md).
# Download sources: checkpoints = GitHub Release; pdb + esm = out-of-band link.
data:
	@found=0; for f in affex-checkpoints.tar.gz affex-pdb.tar.gz affex-esm.tar.gz; do \
	  if [ -f "$$f" ]; then echo "Extracting $$f"; tar xzf "$$f"; found=1; fi; \
	done; \
	if [ "$$found" = 0 ]; then echo "No affex-*.tar.gz found — download them first (see docs/data-prep.md)."; fi

# Ensemble inference over the 25 EXP-043 checkpoints on testAB-clean.
infer:
	uv run python src/predict.py \
	  --checkpoints-dir logs/multiruns/EXP-043 \
	  --test-csv data/test/testAB-clean.csv \
	  --output predictions_testAB.csv

# Reproduce EXP-043 training (25-fold multirun, seed 42, CPU).
train:
	uv run python src/train.py --multirun \
	  +experiment=pcann_reimpl-mc10 \
	  datamodule.val_fold=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 \
	  seed=42 group=EXP-043 \
	  datamodule.train_csv=data/train/pcann-plus-trainval.csv \
	  datamodule.folds_csv=data/train/folds_pcann_plus_ppisplit_mc25.csv

# Release smoke test (install + fetch + 2-fold inference + 1-epoch train).
test:
	bash scripts/test_release.sh
