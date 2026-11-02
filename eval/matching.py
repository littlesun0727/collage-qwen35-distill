"""Geometry-only diagnostics and class-aware end-to-end matching."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(gt, pred):
    """Invalid boxes occupy a slot but have zero overlap with every box."""
    result = np.zeros((len(gt), len(pred)), dtype=np.float64)
    gi = [i for i, o in enumerate(gt) if o.box is not None]
    pi = [i for i, o in enumerate(pred) if o.box is not None]
    if not gi or not pi:
        return result
    a = np.asarray([gt[i].box for i in gi])
    b = np.asarray([pred[i].box for i in pi])
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                    - np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = wh.prod(axis=2)
    union = ((a[:, 2:] - a[:, :2]).prod(axis=1)[:, None]
             + (b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - inter)
    result[np.ix_(gi, pi)] = inter / union
    return result


def assign(ious, allowed=None, threshold=None):
    """Hard: maximize cardinality, then IoU. Soft: maximize summed IoU.

    Row/column order is input order. SciPy's deterministic assignment resolves
    exact ties; no class, ID, relationship or order-based tie-breaking is added.
    """
    if not ious.size:
        return []
    eligible = ious > 0
    if allowed is not None:
        eligible &= allowed
    if threshold is not None:
        eligible &= ious >= threshold
    bonus = min(ious.shape) + 1 if threshold is not None else 0
    weights = np.where(eligible, bonus + ious, 0.0)
    rows, cols = linear_sum_assignment(-weights)
    return [(int(g), int(p), float(ious[g, p]))
            for g, p in zip(rows, cols) if eligible[g, p]]


def match(gt, pred, threshold=0.5, class_aware=False):
    ious = iou_matrix(gt, pred)
    allowed = None
    if class_aware:
        allowed = np.asarray([[g.kind == p.kind for p in pred] for g in gt],
                             dtype=bool).reshape(ious.shape)
    return assign(ious, allowed, threshold)


def potential_ties(ious, matches):
    """Flag local equal-IoU alternatives, not a proof of global ambiguity."""
    flags = []
    for g, p, value in matches:
        gs = [int(i) for i in np.flatnonzero(np.isclose(
            ious[:, p], value, atol=1e-12, rtol=0)) if i != g]
        ps = [int(i) for i in np.flatnonzero(np.isclose(
            ious[g, :], value, atol=1e-12, rtol=0)) if i != p]
        if gs or ps:
            flags.append({"gt_index": g, "pred_index": p,
                          "alternative_gt_indices": gs,
                          "alternative_pred_indices": ps})
    return flags
