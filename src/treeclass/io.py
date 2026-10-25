"""Reading labelled tiles and writing classified point clouds.

Labelling convention (simplest to produce in CloudCompare):
    data/tiles/tile01_tree.las      -> every point is TREE (label 1)
    data/tiles/tile01_nontree.las   -> every point is NON-TREE (label 0)

For a tile without labels (e.g. the whole campus), just use tile.las.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np


@dataclass
class Cloud:
    xyz: np.ndarray                      # (N, 3) float64, real-world coordinates
    labels: np.ndarray | None = None     # (N,) int8, 1 = tree, 0 = non-tree, None if unlabelled
    extras: dict = field(default_factory=dict)  # optional per-point attributes, e.g. intensity, rgb

    def __len__(self) -> int:
        return len(self.xyz)


def _read_las(path: Path) -> tuple[np.ndarray, dict]:
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    names = set(las.point_format.dimension_names)
    extras = {}
    if "intensity" in names and np.any(las.intensity):
        extras["intensity"] = np.asarray(las.intensity, dtype=np.float32)
    if {"red", "green", "blue"} <= names and np.any(las.red):
        rgb = np.vstack([las.red, las.green, las.blue]).T.astype(np.float32)
        if rgb.max() > 255:          # LAS usually stores 16-bit colour
            rgb /= 256.0
        extras["rgb"] = rgb
    return xyz, extras


def thin_cloud(cloud: Cloud, voxel: float) -> Cloud:
    """Keep the first original point in every voxel (same thinning as make_labels.py)."""
    if voxel <= 0:
        return cloud
    ijk = np.floor((cloud.xyz - cloud.xyz.min(axis=0)) / voxel).astype(np.int64)
    dims = ijk.max(axis=0) + 1
    keys = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    _, first = np.unique(keys, return_index=True)
    keep = np.sort(first)
    return Cloud(xyz=cloud.xyz[keep],
                 labels=None if cloud.labels is None else cloud.labels[keep],
                 extras={k: v[keep] for k, v in cloud.extras.items()})


def read_cloud(path: str | Path) -> Cloud:
    """Read one unlabelled .las/.laz file."""
    xyz, extras = _read_las(Path(path))
    return Cloud(xyz=xyz, labels=None, extras=extras)


def read_labelled_tile(tiles_dir: str | Path, tile: str) -> Cloud:
    """Read <tile>_tree.las + <tile>_nontree.las (or .laz) and merge them with labels."""
    tiles_dir = Path(tiles_dir)
    parts = []
    for suffix, label in (("tree", 1), ("nontree", 0)):
        candidates = [tiles_dir / f"{tile}_{suffix}.{ext}" for ext in ("las", "laz")]
        found = [p for p in candidates if p.exists()]
        if not found:
            raise FileNotFoundError(f"Missing {candidates[0]} (or .laz). See README, 'Labelling convention'.")
        xyz, extras = _read_las(found[0])
        parts.append((xyz, np.full(len(xyz), label, dtype=np.int8), extras))

    xyz = np.vstack([p[0] for p in parts])
    labels = np.concatenate([p[1] for p in parts])
    extras = {}
    for key in ("intensity", "rgb"):
        if all(key in p[2] for p in parts):
            extras[key] = np.concatenate([p[2][key] for p in parts])
    return Cloud(xyz=xyz, labels=labels, extras=extras)


# ASPRS standard LAS point classification codes (same in LAS 1.2-1.4)
ASPRS_UNCLASSIFIED = 1
ASPRS_HIGH_VEGETATION = 5


def write_classified(path: str | Path, xyz: np.ndarray, fields: dict[str, np.ndarray],
                     extras: dict | None = None) -> None:
    """Write a .las with extra per-point fields (e.g. pred, label, error_class, hag).

    If a "pred" field is given, the standard LAS classification is also set
    (5 = High Vegetation for trees, 1 = Unclassified for the rest), so CloudCompare,
    QGIS and other LiDAR software show the result without any setup.
    Original colour and intensity are kept when passed in `extras`.
    Open the result in CloudCompare and colour by any of these scalar fields.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    for name, values in fields.items():
        dtype = np.float32 if np.issubdtype(np.asarray(values).dtype, np.floating) else np.int32
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for name, values in fields.items():
        las[name] = np.asarray(values)
    if "pred" in fields:
        las.classification = np.where(np.asarray(fields["pred"]) == 1,
                                      ASPRS_HIGH_VEGETATION, ASPRS_UNCLASSIFIED).astype(np.uint8)
    if extras:
        if "intensity" in extras:
            las.intensity = np.clip(extras["intensity"], 0, 65535).astype(np.uint16)
        if "rgb" in extras:
            rgb = np.clip(np.asarray(extras["rgb"]) * 256.0, 0, 65535).astype(np.uint16)
            las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.write(path)


def write_outputs(out_dir: str | Path, stem: str, cloud: Cloud, pred: np.ndarray,
                  p_tree: np.ndarray, hag: np.ndarray | None) -> dict[str, Path]:
    """The three files a user gets from a classification run:

    <stem>_classified.las  all points, ASPRS classification + pred / p_tree / hag fields
    <stem>_trees.las       only the tree points
    <stem>_nontrees.las    only the non-tree points
    """
    out_dir = Path(out_dir)
    pred = np.asarray(pred).astype(np.int32)
    fields = {"pred": pred, "p_tree": np.asarray(p_tree, dtype=np.float32)}
    if hag is not None:
        fields["hag"] = np.asarray(hag, dtype=np.float32)
    paths = {}
    for key, mask in (("classified", np.ones(len(pred), bool)), ("trees", pred == 1), ("nontrees", pred == 0)):
        if not mask.any():
            continue
        sub_extras = {k: v[mask] for k, v in cloud.extras.items()}
        paths[key] = out_dir / f"{stem}_{key}.las"
        write_classified(paths[key], cloud.xyz[mask], {k: v[mask] for k, v in fields.items()}, sub_extras)
    return paths
