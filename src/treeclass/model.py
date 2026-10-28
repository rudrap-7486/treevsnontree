"""Random Forest training / prediction, label smoothing, and canopy maps.

ADOPTED: scikit-learn RandomForestClassifier (Pedregosa et al., JMLR 12, 2011).
DEVELOPED BY TEAM: training-set subsampling, kNN majority smoothing, canopy maps.
"""

from __future__ import annotations

import joblib
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier


def train_rf(X: np.ndarray, y: np.ndarray, n_estimators: int = 100, max_depth: int | None = None,
             class_weight: str | None = "balanced", random_state: int = 42,
             max_train_points: int | None = 500_000, min_samples_leaf: int = 5,
             n_jobs: int = -1) -> RandomForestClassifier:
    """Fit a Random Forest. Large training sets are randomly subsampled (fixed seed) for speed."""
    rng = np.random.default_rng(random_state)
    if max_train_points and len(X) > max_train_points:
        keep = rng.choice(len(X), size=max_train_points, replace=False)
        X, y = X[keep], y[keep]
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                class_weight=class_weight, random_state=random_state,
                                min_samples_leaf=min_samples_leaf, n_jobs=n_jobs)
    rf.fit(X, y)
    return rf


def predict(rf: RandomForestClassifier, X: np.ndarray, chunk: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, probability of tree)."""
    proba = np.concatenate([rf.predict_proba(X[i:i + chunk]) for i in range(0, len(X), chunk)])
    tree_col = list(rf.classes_).index(1) if 1 in rf.classes_ else None
    p_tree = proba[:, tree_col] if tree_col is not None else np.zeros(len(X))
    return (p_tree >= 0.5).astype(np.int8), p_tree.astype(np.float32)


def smooth_labels(xyz: np.ndarray, p_tree: np.ndarray, k: int = 10) -> np.ndarray:
    """Majority vote: average tree probability over each point's k nearest neighbours."""
    _, idx = cKDTree(xyz).query(xyz, k=k)
    return (p_tree[idx].mean(axis=1) >= 0.5).astype(np.int8)


def save_model(path, rf, meta: dict) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "meta": meta}, path, compress=3)


def load_model(path) -> tuple[RandomForestClassifier, dict]:
    bundle = joblib.load(path)
    return bundle["model"], bundle["meta"]


def canopy_maps(xyz: np.ndarray, is_tree: np.ndarray, hag: np.ndarray | None,
                cell: float = 1.0, block: int = 10) -> dict:
    """2D products from a classified cloud, measured over the SCANNED area only.

    A street-level scan covers strips along the scanner's path; the rest of its bounding box
    holds no data, so it must not count as "no trees".
    cover_pct : per block x block cells (10 m by default), % of the SCANNED 1 m cells that hold
                tree points; NaN (drawn blank) where the block holds no scanned cell
    height    : max height above ground of tree points per 1 m cell (NaN where no tree)
    canopy_cover_total_pct : tree cells / scanned cells, in %
    """
    xy0 = xyz[:, :2].min(axis=0)
    ij = np.floor((xyz[:, :2] - xy0) / cell).astype(np.int64)
    nx, ny = ij.max(axis=0) + 1
    scanned = np.zeros((ny, nx), dtype=bool)
    scanned[ij[:, 1], ij[:, 0]] = True              # any point at all = this cell was scanned
    tree_mask = np.zeros((ny, nx), dtype=bool)
    t = is_tree.astype(bool)
    tree_mask[ij[t, 1], ij[t, 0]] = True

    height = np.full((ny, nx), np.nan, dtype=np.float32)
    if hag is not None and t.any():
        flat = ij[t, 1] * nx + ij[t, 0]
        hmax = np.full(nx * ny, -np.inf)
        np.maximum.at(hmax, flat, hag[t])
        hmax[~np.isfinite(hmax)] = np.nan
        height = hmax.reshape(ny, nx).astype(np.float32)

    by, bx = int(np.ceil(ny / block)), int(np.ceil(nx / block))
    pad_tree = np.zeros((by * block, bx * block), dtype=float)
    pad_scan = np.zeros((by * block, bx * block), dtype=float)
    pad_tree[:ny, :nx] = tree_mask
    pad_scan[:ny, :nx] = scanned
    n_tree = pad_tree.reshape(by, block, bx, block).sum(axis=(1, 3))
    n_scan = pad_scan.reshape(by, block, bx, block).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        cover = np.where(n_scan > 0, n_tree / n_scan * 100.0, np.nan)
    n_scanned = int(scanned.sum())
    return {"cover_pct": cover, "height": height, "origin": xy0, "cell": cell, "block": block,
            "scanned_area_m2": float(n_scanned * cell * cell),
            "canopy_cover_total_pct": float(tree_mask.sum() / max(n_scanned, 1) * 100.0)}
