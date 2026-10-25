"""Eigenvalue (covariance) shape features at one or more neighbourhood radii.

ADOPTED (feature definitions):
    M. Weinmann et al., "Semantic point cloud interpretation based on optimal
    neighborhoods, relevant features and efficient classifiers," ISPRS J.
    Photogramm. Remote Sens., vol. 105, pp. 286-304, 2015.
ADOPTED (idea of multi-scale neighbourhoods on a voxel-downsampled pyramid):
    T. Hackel, J. D. Wegner, K. Schindler, "Fast semantic segmentation of 3D point
    clouds with strongly varying density," ISPRS Annals, vol. III-3, pp. 177-184, 2016.
DEVELOPED BY TEAM: this vectorised NumPy implementation, the radius/voxel scheme
    below, and the choice of radii.

How a scale works here
----------------------
For radius r, the cloud is first voxel-downsampled with voxel size r / 3 so that
the neighbourhood holds a manageable number of points whatever the scan density.
Then, for every ORIGINAL point, we take up to `k_max` nearest downsampled points
lying within distance r, compute their 3x3 covariance matrix, and derive the
features from its eigenvalues l1 >= l2 >= l3 (normalised to sum to 1: e1, e2, e3).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

EIGEN_FEATURES = [
    "linearity",      # (e1 - e2) / e1
    "planarity",      # (e2 - e3) / e1
    "sphericity",     # e3 / e1
    "omnivariance",   # (e1 e2 e3)^(1/3)
    "anisotropy",     # (e1 - e3) / e1
    "eigenentropy",   # -sum(ei ln ei)
    "curvature",      # change of curvature: e3 / (e1 + e2 + e3)
    "verticality",    # 1 - |nz|, n = eigenvector of the smallest eigenvalue
]


def voxel_downsample(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Replace all points inside each voxel by their centroid."""
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, xyz)
    return sums / counts[:, None]


def _features_from_neighbours(nbr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """nbr: (n, k, 3) neighbour coordinates, valid: (n, k) bool. Returns (n, 8)."""
    w = valid[..., None].astype(np.float64)
    count = w.sum(axis=1)                                     # (n, 1)
    mean = (nbr * w).sum(axis=1) / np.maximum(count, 1)
    centred = (nbr - mean[:, None, :]) * w
    cov = np.einsum("nki,nkj->nij", centred, centred) / np.maximum(count, 1)[:, :, None]
    evals, evecs = np.linalg.eigh(cov)                        # ascending order
    evals = np.clip(evals, 0.0, None)
    l3, l2, l1 = evals[:, 0], evals[:, 1], evals[:, 2]
    total = l1 + l2 + l3
    ok = (count[:, 0] >= 3) & (total > 1e-12)
    s = np.where(ok, total, 1.0)
    e1, e2, e3 = l1 / s, l2 / s, l3 / s
    e1s = np.where(e1 > 0, e1, 1.0)
    eps = 1e-12
    feats = np.stack([
        (e1 - e2) / e1s,
        (e2 - e3) / e1s,
        e3 / e1s,
        np.cbrt(e1 * e2 * e3),
        (e1 - e3) / e1s,
        -(e1 * np.log(e1 + eps) + e2 * np.log(e2 + eps) + e3 * np.log(e3 + eps)),
        e3 / np.where(ok, e1 + e2 + e3, 1.0),
        1.0 - np.abs(evecs[:, 2, 0]),                         # z of smallest eigenvector
    ], axis=1)
    feats[~ok] = 0.0                                          # too few neighbours: neutral values
    return feats.astype(np.float32)


def eigen_features(xyz: np.ndarray, radius: float, k_max: int = 32,
                   chunk: int = 100_000) -> np.ndarray:
    """(N, 8) eigenvalue features of every point at one radius."""
    support = voxel_downsample(xyz, radius / 3.0)
    tree = cKDTree(support)
    k = min(k_max, len(support))
    out = np.empty((len(xyz), len(EIGEN_FEATURES)), dtype=np.float32)
    for start in range(0, len(xyz), chunk):
        q = xyz[start:start + chunk]
        dist, idx = tree.query(q, k=k, distance_upper_bound=radius)
        if k == 1:
            dist, idx = dist[:, None], idx[:, None]
        valid = np.isfinite(dist)
        idx = np.where(valid, idx, 0)
        out[start:start + chunk] = _features_from_neighbours(support[idx], valid)
    return out


def build_feature_matrix(xyz: np.ndarray, radii: list[float], hag: np.ndarray | None = None,
                         extras: dict | None = None, use_extras: list[str] | None = None
                         ) -> tuple[np.ndarray, list[str]]:
    """Stack every requested feature into one (N, F) matrix plus the column names."""
    blocks, names = [], []
    for r in radii:
        blocks.append(eigen_features(xyz, r))
        names += [f"{f}_r{r:g}" for f in EIGEN_FEATURES]
    if hag is not None:
        blocks.append(hag[:, None].astype(np.float32))
        names.append("height_above_ground")
    for key in use_extras or []:
        if not extras or key not in extras:
            raise KeyError(f"Extra attribute '{key}' requested but not present in the data.")
        values = extras[key]
        values = values[:, None] if values.ndim == 1 else values
        blocks.append(values.astype(np.float32))
        names += [key] if values.shape[1] == 1 else [f"{key}_{c}" for c in "rgb"[: values.shape[1]]]
    return np.hstack(blocks), names
