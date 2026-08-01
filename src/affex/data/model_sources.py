"""Canonical pretrained model sources used by every experiment pipeline."""

from pathlib import Path


ESM2_MODEL = Path("models/esm2_t33_650M_UR50D")
ESMC_WEIGHTS = Path("models/esmc_600m_2024_12/esmc_600m_2024_12_v0.pth")
ESMC_LIBRARY = Path("models/biohub_esm_lib")
PLM_INTERACT_REPO = "danliu1226/PLM-interact-650M-Leakage-Free-Dataset"
PLM_INTERACT_CHECKPOINT = Path("models/plm_interact/pytorch_model.bin")
