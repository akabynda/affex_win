"""Analyze how resolved protein-chain length relates to absolute prediction error."""

from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PDB_DIR = ROOT / "data/raw/ppb-affinity/pdb"
OUT_DIR = ROOT / "analysis/length_mae"
DATASETS = {
    "testAB": ROOT / "predictions_testAB.csv",
    "test_fabs": ROOT / "predictions_test_fabs.csv",
}
TRAINVAL_CSV = ROOT / "data/train/pcann-plus-trainval.csv"
LENGTH_COLUMNS = {
    "receptor_length": "Receptor length",
    "ligand_length": "Ligand length",
    "total_length": "Total complex length",
}


def chain_lengths(pdb_path: Path) -> dict[str, int]:
    """Count unique residues having a CA atom in the first PDB model."""
    residues: set[tuple[str, str, str]] = set()
    started_model = False
    with pdb_path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                if started_model:
                    break
                started_model = True
                continue
            if line.startswith("ENDMDL"):
                break
            if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA":
                residues.add((line[21].strip(), line[22:26].strip(), line[26].strip()))
    lengths: dict[str, int] = {}
    for chain, _, _ in residues:
        lengths[chain] = lengths.get(chain, 0) + 1
    return lengths


def load_dataset(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    parsed = df["uid"].str.extract(r"^(?P<pdb_id>[^_]+)_(?P<receptor_chains>[^_]+)_(?P<ligand_chains>[^_]+)$")
    if parsed.isna().any().any():
        raise ValueError(f"Unexpected uid format in {path}")
    df = pd.concat([df, parsed], axis=1)

    cache: dict[str, dict[str, int]] = {}
    receptor_lengths, ligand_lengths = [], []
    for row in df.itertuples(index=False):
        lengths = cache.setdefault(row.pdb_id, chain_lengths(PDB_DIR / f"{row.pdb_id}.pdb"))
        receptor_lengths.append(sum(lengths.get(chain, 0) for chain in row.receptor_chains))
        ligand_lengths.append(sum(lengths.get(chain, 0) for chain in row.ligand_chains))
    df["dataset"] = name
    df["receptor_length"] = receptor_lengths
    df["ligand_length"] = ligand_lengths
    df["total_length"] = df["receptor_length"] + df["ligand_length"]
    df["absolute_error"] = (df["pred"] - df["target"]).abs()
    return df


def load_trainval_lengths() -> pd.DataFrame:
    """Load chain assignments and resolved lengths for the training pool."""
    df = pd.read_csv(TRAINVAL_CSV)
    cache: dict[str, dict[str, int]] = {}
    receptor_lengths, ligand_lengths = [], []
    for row in df.itertuples(index=False):
        pdb_id = str(row.uid).lower()
        lengths = cache.setdefault(pdb_id, chain_lengths(PDB_DIR / f"{pdb_id}.pdb"))
        receptor_lengths.append(sum(lengths.get(chain, 0) for chain in str(row.receptor_chains)))
        ligand_lengths.append(sum(lengths.get(chain, 0) for chain in str(row.ligand_chains)))
    df["receptor_length"] = receptor_lengths
    df["ligand_length"] = ligand_lengths
    df["total_length"] = df["receptor_length"] + df["ligand_length"]
    return df


def statistics_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, df in frames.items():
        for column, label in LENGTH_COLUMNS.items():
            x, y = df[column].to_numpy(float), df["absolute_error"].to_numpy(float)
            pearson = stats.pearsonr(x, y)
            spearman = stats.spearmanr(x, y)
            regression = stats.linregress(x, y)
            rows.append({
                "dataset": dataset,
                "length_measure": column,
                "n": len(df),
                "mean_length": x.mean(),
                "mean_absolute_error": y.mean(),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "slope_mae_per_residue": regression.slope,
                "slope_p": regression.pvalue,
                "r_squared": regression.rvalue**2,
            })
    return pd.DataFrame(rows)


def make_svg(
    df: pd.DataFrame,
    trainval: pd.DataFrame,
    dataset: str,
    stats_df: pd.DataFrame,
    output: Path,
) -> None:
    width, height = 1320, 510
    # Reserve a marginal strip above each scatter plot for trainval lengths.
    panel_w, margin_l, margin_t, plot_w, plot_h = 430, 62, 148, 330, 270
    colors = {"testAB": "#2563EB", "test_fabs": "#D97706"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:20px;font-weight:700}.label{font-size:12px}.small{font-size:11px;fill:#4B5563}.grid{stroke:#E5E7EB;stroke-width:1}.axis{stroke:#6B7280;stroke-width:1.2}</style>',
        f'<text x="20" y="28" class="title">Protein length vs absolute error — {html.escape(dataset)}</text>',
        '<circle cx="28" cy="48" r="3" fill="#059669" fill-opacity="0.32"/><text x="38" y="52" class="small">trainval length distribution</text>',
    ]
    for panel, (column, label) in enumerate(LENGTH_COLUMNS.items()):
        ox = panel * panel_w
        x = df[column].to_numpy(float)
        train_x = trainval[column].to_numpy(float)
        y = df["absolute_error"].to_numpy(float)
        xmin, xmax = 0.0, max(max(x), max(train_x)) * 1.05
        ymin, ymax = 0.0, max(y) * 1.10
        sx = lambda value: ox + margin_l + (value - xmin) / (xmax - xmin) * plot_w
        sy = lambda value: margin_t + plot_h - (value - ymin) / (ymax - ymin) * plot_h
        for i in range(6):
            gy = margin_t + plot_h * i / 5
            val = ymax * (5 - i) / 5
            parts.append(f'<line x1="{ox+margin_l}" y1="{gy:.1f}" x2="{ox+margin_l+plot_w}" y2="{gy:.1f}" class="grid"/>')
            parts.append(f'<text x="{ox+margin_l-8}" y="{gy+4:.1f}" text-anchor="end" class="small">{val:.1f}</text>')
        parts.append(f'<line x1="{ox+margin_l}" y1="{margin_t}" x2="{ox+margin_l}" y2="{margin_t+plot_h}" class="axis"/>')
        parts.append(f'<line x1="{ox+margin_l}" y1="{margin_t+plot_h}" x2="{ox+margin_l+plot_w}" y2="{margin_t+plot_h}" class="axis"/>')
        for i in range(6):
            gx = ox + margin_l + plot_w * i / 5
            val = xmax * i / 5
            parts.append(f'<text x="{gx:.1f}" y="{margin_t+plot_h+19}" text-anchor="middle" class="small">{val:.0f}</text>')
        for xv, yv in zip(x, y):
            parts.append(f'<circle cx="{sx(xv):.2f}" cy="{sy(yv):.2f}" r="3.2" fill="{colors[dataset]}" fill-opacity="0.58"/>')
        # Deterministic vertical jitter makes the one-dimensional training
        # distribution visible without assigning it fictitious MAE values.
        strip_top, strip_height = margin_t - 42, 18
        for idx, xv in enumerate(train_x):
            jitter = ((idx * 37) % 101) / 100
            parts.append(f'<circle cx="{sx(xv):.2f}" cy="{strip_top + jitter * strip_height:.2f}" r="2.0" fill="#059669" fill-opacity="0.20"/>')
        slope, intercept = np.polyfit(x, y, 1)
        parts.append(f'<line x1="{sx(xmin):.2f}" y1="{sy(intercept):.2f}" x2="{sx(xmax):.2f}" y2="{sy(intercept+slope*xmax):.2f}" stroke="#DC2626" stroke-width="2.2"/>')
        row = stats_df[(stats_df.dataset == dataset) & (stats_df.length_measure == column)].iloc[0]
        parts.append(f'<text x="{ox+margin_l}" y="{margin_t-78}" class="label" font-weight="700">{html.escape(label)}</text>')
        parts.append(f'<text x="{ox+margin_l}" y="{margin_t-59}" class="small">Pearson r={row.pearson_r:.3f} (p={row.pearson_p:.3g}); Spearman ρ={row.spearman_rho:.3f}</text>')
        parts.append(f'<text x="{ox+margin_l}" y="{margin_t-47}" class="small" fill="#047857">trainval</text>')
        parts.append(f'<text x="{ox+margin_l+plot_w/2}" y="{margin_t+plot_h+39}" text-anchor="middle" class="label">Length (resolved residues)</text>')
        if panel == 0:
            parts.append(f'<text x="16" y="{margin_t+plot_h/2}" text-anchor="middle" class="label" transform="rotate(-90 16 {margin_t+plot_h/2})">Absolute error (kcal/mol)</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {name: load_dataset(name, path) for name, path in DATASETS.items()}
    trainval = load_trainval_lengths()
    combined = pd.concat(frames.values(), ignore_index=True)
    combined.to_csv(OUT_DIR / "length_mae_observations.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    summary = statistics_table(frames)
    summary.to_csv(OUT_DIR / "length_mae_statistics.csv", index=False)
    for name, frame in frames.items():
        make_svg(frame, trainval, name, summary, OUT_DIR / f"{name}_length_vs_mae.svg")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
