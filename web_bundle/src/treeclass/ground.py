"""Ground filtering with CSF and height above ground (HAG).

ADOPTED: the Cloth Simulation Filter algorithm and its Python binding
    W. Zhang et al., "An easy-to-use airborne LiDAR data filtering method based on
    cloth simulation," Remote Sens., vol. 8, no. 6, Art. no. 501, 2016.
    Library: cloth-simulation-filter (PyPI).
DEVELOPED BY TEAM: height-above-ground interpolation (inverse-distance weighting
    of the nearest ground points) and parameter choices.
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from scipy.spatial import cKDTree


@contextlib.contextmanager
def _quiet():
    """CSF prints progress from C++ straight to file descriptor 1; silence it so logs
    and notebooks stay readable."""
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        saved = os.dup(1)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


def csf_ground_mask(xyz: np.ndarray, cloth_resolution: float = 0.5, rigidness: int = 2,
                    class_threshold: float = 0.5, slope_smooth: bool = False,
                    iterations: int = 500) -> np.ndarray:
    """Return a boolean mask, True where the point is ground."""
    import CSF  # imported here so the rest of the package works without it

    csf = CSF.CSF()
    csf.params.bSloopSmooth = slope_smooth
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.class_threshold = class_threshold
    csf.params.interations = iterations          # (sic) the library's own spelling
    # CSF works best near the origin; shift to avoid float precision issues with UTM coordinates
    csf.setPointCloud(xyz - xyz.min(axis=0))
    ground, non_ground = CSF.VecInt(), CSF.VecInt()
    with _quiet():
        csf.do_filtering(ground, non_ground, False)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[np.asarray(ground, dtype=np.int64)] = True
    return mask


def height_above_ground(xyz: np.ndarray, ground_mask: np.ndarray, k: int = 8) -> np.ndarray:
    """Z minus the ground elevation below each point (IDW of k nearest ground points in XY)."""
    ground = xyz[ground_mask]
    if len(ground) < k:
        raise ValueError("Too few ground points found; check the CSF parameters for this tile.")
    tree = cKDTree(ground[:, :2])
    dist, idx = tree.query(xyz[:, :2], k=k)
    w = 1.0 / np.maximum(dist, 1e-3) ** 2
    z_ground = (w * ground[idx, 2]).sum(axis=1) / w.sum(axis=1)
    return (xyz[:, 2] - z_ground).astype(np.float32)
