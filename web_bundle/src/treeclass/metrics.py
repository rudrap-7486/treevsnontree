"""Accuracy assessment for binary tree / non-tree labels (1 = tree, 0 = non-tree)."""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    return {
        "TP": int(np.sum(y_true & y_pred)),
        "FP": int(np.sum(~y_true & y_pred)),
        "FN": int(np.sum(y_true & ~y_pred)),
        "TN": int(np.sum(~y_true & ~y_pred)),
    }


def _safe(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision, recall, F1 and IoU for both classes, plus mIoU, OA and Cohen's kappa."""
    c = confusion(y_true, y_pred)
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    n = tp + fp + fn + tn
    out = {"n_points": n, **c}
    # tree is the positive class; for non-tree the roles of FP and FN swap
    for name, (t, f_pos, f_neg) in {"tree": (tp, fp, fn), "nontree": (tn, fn, fp)}.items():
        p = _safe(t, t + f_pos)
        r = _safe(t, t + f_neg)
        out[f"precision_{name}"] = p
        out[f"recall_{name}"] = r
        out[f"f1_{name}"] = _safe(2 * p * r, p + r) if np.isfinite(p) and np.isfinite(r) else float("nan")
        out[f"iou_{name}"] = _safe(t, t + f_pos + f_neg)
    out["miou"] = float(np.nanmean([out["iou_tree"], out["iou_nontree"]]))
    out["overall_accuracy"] = _safe(tp + tn, n)
    p_exp = _safe((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), n * n)
    out["kappa"] = _safe(out["overall_accuracy"] - p_exp, 1 - p_exp)
    return out


def error_classes(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-point code for error maps: 0 = correct non-tree, 1 = correct tree,
    2 = false tree (FP), 3 = missed tree (FN)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    code = np.zeros(len(y_true), dtype=np.int32)
    code[y_true & y_pred] = 1
    code[~y_true & y_pred] = 2
    code[y_true & ~y_pred] = 3
    return code
