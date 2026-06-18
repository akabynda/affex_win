#!/usr/bin/env bash
# Release smoke test: from a clean checkout + tarball, prove install / inference /
# training run. Fast and CI-runnable.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/4] install"
uv sync --extra cpu --no-dev
uv run python -c "import affex, affex.model.pcann, affex.data.datamodule"

echo "[2/4] unpack artifacts (download affex-*.tar.gz into the repo root first)"
make data

echo "[3/4] inference smoke — 2 folds"
uv run python src/predict.py \
  --checkpoints-dir logs/multiruns/EXP-043 \
  --test-csv data/test/test-fabs.csv \
  --folds 0,1 --output /tmp/affex_smoke_preds.csv
uv run python - <<'PY'
import pandas as pd
d = pd.read_csv("/tmp/affex_smoke_preds.csv")
assert {"uid", "pred"} <= set(d.columns) and len(d) and d["pred"].notna().all()
print(f"inference OK: {len(d)} rows")
PY

echo "[4/4] training smoke — 1 fold, 1 epoch"
uv run python src/train.py +experiment=pcann_reimpl-mc10 \
  datamodule.val_fold=0 seed=42 trainer.max_epochs=1 group=SMOKE \
  datamodule.train_csv=data/train/pcann-plus-trainval.csv \
  datamodule.folds_csv=data/train/folds_pcann_plus_ppisplit_mc25.csv
ls logs/multiruns/SMOKE/fold0_seed42/checkpoints/*.ckpt >/dev/null

echo "ALL SMOKE CHECKS PASSED"
