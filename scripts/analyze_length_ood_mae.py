"""Relate prediction error to distance from the trainval length distribution."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analyze_length_mae as base


OUT_DIR = base.ROOT / "analysis/length_ood_mae"
K = 10
RNG_SEED = 42


def knn_ood_score(test: np.ndarray, train: np.ndarray, k: int = K) -> np.ndarray:
    """Median absolute distance to k train lengths, divided by train IQR."""
    iqr = np.subtract(*np.percentile(train, [75, 25]))
    if iqr <= 0:
        raise ValueError("Training-length IQR must be positive")
    distances = np.abs(test[:, None] - train[None, :])
    nearest = np.partition(distances, k - 1, axis=1)[:, :k]
    return np.median(nearest, axis=1) / iqr


def permutation_pvalue(x: np.ndarray, y: np.ndarray, observed: float, n: int = 10_000) -> float:
    rng = np.random.default_rng(RNG_SEED)
    count = 0
    for _ in range(n):
        permuted = stats.spearmanr(x, rng.permutation(y)).statistic
        count += abs(permuted) >= abs(observed)
    return (count + 1) / (n + 1)


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n: int = 5_000) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        value = stats.spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(value):
            values.append(value)
    return tuple(np.percentile(values, [2.5, 97.5]))


def quantile_trend(x: np.ndarray, y: np.ndarray, groups: int = 5) -> list[tuple[float, float, int]]:
    """Return median x/error in quantile groups, tolerating tied scores."""
    ranked = pd.Series(x).rank(method="first")
    bins = pd.qcut(ranked, q=min(groups, len(x)), labels=False)
    frame = pd.DataFrame({"x": x, "y": y, "bin": bins})
    return [
        (group.x.median(), group.y.median(), len(group))
        for _, group in frame.groupby("bin", sort=True)
    ]


def analyze(frames: dict[str, pd.DataFrame], trainval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations, summaries = [], []
    for dataset, frame in frames.items():
        result = frame.copy()
        for column in base.LENGTH_COLUMNS:
            score_col = column.replace("_length", "_length_ood")
            result[score_col] = knn_ood_score(
                result[column].to_numpy(float), trainval[column].to_numpy(float)
            )
            x = result[score_col].to_numpy(float)
            y = result["absolute_error"].to_numpy(float)
            rho = stats.spearmanr(x, y).statistic
            ci_low, ci_high = bootstrap_ci(x, y)
            summaries.append({
                "dataset": dataset,
                "length_measure": column,
                "n": len(result),
                "k_neighbors": K,
                "spearman_rho": rho,
                "permutation_p": permutation_pvalue(x, y, rho),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "median_ood": np.median(x),
                "max_ood": np.max(x),
            })
        observations.append(result)
    return pd.concat(observations, ignore_index=True), pd.DataFrame(summaries)


def make_svg(df: pd.DataFrame, dataset: str, summary: pd.DataFrame, output: Path) -> None:
    width, height = 1320, 470
    panel_w, margin_l, margin_t, plot_w, plot_h = 430, 62, 106, 330, 280
    point_color = {"testAB": "#2563EB", "test_fabs": "#D97706"}[dataset]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:20px;font-weight:700}.label{font-size:12px}.small{font-size:11px;fill:#4B5563}.grid{stroke:#E5E7EB;stroke-width:1}.axis{stroke:#6B7280;stroke-width:1.2}</style>',
        f'<text x="20" y="28" class="title">Error vs distance from trainval length distribution — {html.escape(dataset)}</text>',
    ]
    y = df["absolute_error"].to_numpy(float)
    ymax = max(y) * 1.1
    for panel, (length_col, label) in enumerate(base.LENGTH_COLUMNS.items()):
        ox = panel * panel_w
        score_col = length_col.replace("_length", "_length_ood")
        x = df[score_col].to_numpy(float)
        xmax = max(x.max() * 1.08, 0.01)
        sx = lambda value: ox + margin_l + value / xmax * plot_w
        sy = lambda value: margin_t + plot_h - value / ymax * plot_h
        for i in range(6):
            gy = margin_t + plot_h * i / 5
            parts.append(f'<line x1="{ox+margin_l}" y1="{gy:.1f}" x2="{ox+margin_l+plot_w}" y2="{gy:.1f}" class="grid"/>')
            parts.append(f'<text x="{ox+margin_l-8}" y="{gy+4:.1f}" text-anchor="end" class="small">{ymax*(5-i)/5:.1f}</text>')
        parts.append(f'<line x1="{ox+margin_l}" y1="{margin_t}" x2="{ox+margin_l}" y2="{margin_t+plot_h}" class="axis"/>')
        parts.append(f'<line x1="{ox+margin_l}" y1="{margin_t+plot_h}" x2="{ox+margin_l+plot_w}" y2="{margin_t+plot_h}" class="axis"/>')
        for i in range(5):
            gx = ox + margin_l + plot_w * i / 4
            parts.append(f'<text x="{gx:.1f}" y="{margin_t+plot_h+18}" text-anchor="middle" class="small">{xmax*i/4:.2f}</text>')
        for xv, yv in zip(x, y):
            parts.append(f'<circle cx="{sx(xv):.2f}" cy="{sy(yv):.2f}" r="3.3" fill="{point_color}" fill-opacity="0.55"/>')
        trend = quantile_trend(x, y)
        path = " ".join(("M" if i == 0 else "L") + f" {sx(tx):.2f} {sy(ty):.2f}" for i, (tx, ty, _) in enumerate(trend))
        parts.append(f'<path d="{path}" fill="none" stroke="#DC2626" stroke-width="2.3"/>')
        for tx, ty, _ in trend:
            parts.append(f'<circle cx="{sx(tx):.2f}" cy="{sy(ty):.2f}" r="4" fill="#DC2626"/>')
        row = summary[(summary.dataset == dataset) & (summary.length_measure == length_col)].iloc[0]
        parts.append(f'<text x="{ox+margin_l}" y="{margin_t-47}" class="label" font-weight="700">{html.escape(label)}</text>')
        parts.append(f'<text x="{ox+margin_l}" y="{margin_t-27}" class="small">Spearman ρ={row.spearman_rho:.3f}; permutation p={row.permutation_p:.3g}</text>')
        parts.append(f'<text x="{ox+margin_l}" y="{margin_t-11}" class="small">bootstrap 95% CI [{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}]</text>')
        parts.append(f'<text x="{ox+margin_l+plot_w/2}" y="{margin_t+plot_h+39}" text-anchor="middle" class="label">10-NN distance / trainval IQR</text>')
        if panel == 0:
            parts.append(f'<text x="16" y="{margin_t+plot_h/2}" text-anchor="middle" class="label" transform="rotate(-90 16 {margin_t+plot_h/2})">Absolute error (kcal/mol)</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = {name: base.load_dataset(name, path) for name, path in base.DATASETS.items()}
    trainval = base.load_trainval_lengths()
    observations, summary = analyze(frames, trainval)
    observations.to_csv(OUT_DIR / "length_ood_observations.csv", index=False)
    summary.to_csv(OUT_DIR / "length_ood_statistics.csv", index=False)
    for name in frames:
        make_svg(observations[observations.dataset == name], name, summary, OUT_DIR / f"{name}_length_ood_vs_mae.svg")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
