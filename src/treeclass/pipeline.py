"""Glue: turn a cloud into a feature matrix, with on-disk caching of the slow parts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .features import build_feature_matrix
from .ground import csf_ground_mask, height_above_ground
from .io import Cloud


def _key(*parts) -> str:
    return hashlib.md5(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def compute_hag(cloud: Cloud, ground_cfg: dict, cache_dir: Path | None = None, name: str = "") -> np.ndarray:
    path = cache_dir / f"{name}_hag_{_key(ground_cfg, len(cloud))}.npy" if cache_dir and name else None
    if path is not None and path.exists():
        return np.load(path)
    mask = csf_ground_mask(cloud.xyz, **ground_cfg)
    hag = height_above_ground(cloud.xyz, mask)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, hag)
    return hag


def features_for(cloud: Cloud, exp: dict, ground_cfg: dict, cache_dir: Path | None = None,
                 name: str = "") -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """Feature matrix for one cloud under one experiment definition.

    exp keys: radii (list of metres), hag (bool), extras (list: 'intensity', 'rgb')
    Returns (X, feature_names, hag or None).
    """
    hag = compute_hag(cloud, ground_cfg, cache_dir, name) if exp.get("hag") else None
    blocks, names = [], []
    for r in exp["radii"]:                      # cache each radius separately: E1..E6 share them
        path = cache_dir / f"{name}_eig_r{r:g}_{_key(r, len(cloud))}.npz" if cache_dir and name else None
        if path is not None and path.exists():
            d = np.load(path, allow_pickle=False)
            Xr, nr = d["X"], list(d["names"])
        else:
            Xr, nr = build_feature_matrix(cloud.xyz, [r])
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, X=Xr, names=np.array(nr))
        blocks.append(Xr)
        names += nr
    if hag is not None or exp.get("extras"):
        Xo, no = build_feature_matrix(cloud.xyz, [], hag=hag, extras=cloud.extras,
                                      use_extras=exp.get("extras") or [])
        blocks.append(Xo)
        names += no
    return np.hstack(blocks), names, hag


def classify_cloud(cloud: Cloud, model_path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply a trained model (models/<EXP>.joblib) to any cloud.

    Returns (pred labels, tree probability, height above ground, model metadata).
    """
    from .model import load_model, predict, smooth_labels

    rf, meta = load_model(model_path)
    exp = {k: meta.get(k) for k in ("radii", "hag", "extras")}
    X, names, hag = features_for(cloud, exp, meta["ground"])
    if names != meta["feature_names"]:
        raise ValueError("Feature columns differ from the ones the model was trained with "
                         f"(model: {meta['feature_names']}, data: {names}).")
    pred, p_tree = predict(rf, X)
    if meta.get("smoothing_k"):
        pred = smooth_labels(cloud.xyz, p_tree, k=meta["smoothing_k"])
    if hag is None:                       # maps need height even if the model did not use it
        hag = compute_hag(cloud, meta["ground"])
    return pred, p_tree, hag, meta
