"""Eight evaluation dimensions, with explicit denominators and failure details."""

from collections import Counter

from .matching import iou_matrix, match, potential_ties
from .schema import KINDS, BACKGROUND_TYPES, INVALID_KIND, RELATIONS, validate_occlusions


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def prf(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def confusion():
    return {g: {p: 0 for p in (*KINDS, INVALID_KIND)} for g in KINDS}


def per_template_macro(samples):
    """Equal weight per applicable template; always expose the denominator.

    None is undefined, not a zero or a perfect score. In particular, conditional
    classification/IoU means exclude templates without matches, and must be read
    together with detection, coverage and eligible_images.
    """
    def mean(*path):
        values = []
        for sample in samples:
            value = sample
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        return {"mean": ratio(sum(values), len(values)),
                "eligible_images": len(values), "total_images": len(samples)}

    return {
        "detection": {key: mean("detection", key) for key in ("precision", "recall", "f1")},
        "classification": {"accuracy": mean("classification", "accuracy")},
        "localization": {key: mean("localization", key) for key in ("mean_iou", "coverage")},
        "relations": {r: {key: mean("relations", r, key) for key in ("precision", "recall", "f1")}
                      for r in RELATIONS},
        "occlusion_order": {key: mean("occlusion_order", key)
                            for key in ("accuracy", "conditional_accuracy", "pair_coverage")},
        "background": {"accuracy": mean("background", "correct")},
        "validity": {key + "_rate": mean("validity", key)
                     for key in ("present", "json_valid", "structure_valid")},
        "auxiliary_iou_f1": {kind: mean("auxiliary_iou_f1", kind, "iou_weighted_f1")
                             for kind in KINDS},
    }


def evaluate_sample(sample_id, gt, pred, occlusion_pairs=None, threshold=0.5):
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if not gt.valid:
        raise ValueError(f"Invalid GT {sample_id}: {'; '.join(gt.errors)}")
    validate_occlusions(occlusion_pairs, gt)
    diagnostic = match(gt.objects, pred.objects, threshold, class_aware=False)
    structural = match(gt.objects, pred.objects, threshold, class_aware=True)
    gt_unique, pred_unique = gt.unique_ids(), pred.unique_ids()
    # References require unique, valid IDs at both endpoints.
    p_to_g = {p: g for g, p, _ in structural
              if gt_unique.get(gt.objects[g].id) == g
              and pred_unique.get(pred.objects[p].id) == p}
    g_to_p = {g: p for p, g in p_to_g.items()}
    n_gt, n_pred, n_matched = len(gt.objects), len(pred.objects), len(diagnostic)
    matrix = confusion()
    for g, p, _ in diagnostic:
        matrix[gt.objects[g].kind][pred.objects[p].kind] += 1
    class_correct = sum(matrix[k][k] for k in KINDS)
    iou_sum = sum(v for _, _, v in diagnostic)
    detection = prf(n_matched, n_pred - n_matched, n_gt - n_matched)
    detection.update({"gt_count": n_gt, "pred_count": n_pred})
    per_class = {}
    for kind in KINDS:
        total = sum(o.kind == kind for o in gt.objects)
        matched = sum(matrix[kind].values())
        per_class[kind] = {"gt_count": total, "matched": matched,
                           "recall": ratio(matched, total),
                           "classification_accuracy": ratio(matrix[kind][kind], matched)}

    def gt_edge(edge):
        if edge.target == gt.background.get("id") and gt.background.get("type") == "photographic":
            target = ("background", 0)
        else:
            target = ("object", gt_unique[edge.target])
        return edge.source, target

    def pred_edge(edge):
        if not edge.syntax_valid or edge.source not in p_to_g:
            return None
        if (pred.background.get("type") == "photographic"
                and gt.background.get("type") == "photographic"
                and edge.target == pred.background.get("id")):
            target = ("background", 0)
        else:
            target_idx = pred_unique.get(edge.target)
            if target_idx not in p_to_g:
                return None
            target = ("object", p_to_g[target_idx])
        return p_to_g[edge.source], target

    relations = {}
    relation_details = {}
    for relation in RELATIONS:
        expected = Counter(gt_edge(e) for e in gt.edges[relation])
        remaining = expected.copy()
        predictions = []
        tp = 0
        for edge in pred.edges[relation]:
            mapped = pred_edge(edge)
            correct = mapped is not None and remaining[mapped] > 0
            if correct:
                tp += 1
                remaining[mapped] -= 1
            predictions.append({"source_index": edge.source,
                                "source_id": pred.objects[edge.source].id,
                                "target": edge.target, "correct": correct,
                                "mapped_gt_edge": mapped})
        ng, np = len(gt.edges[relation]), len(pred.edges[relation])
        relations[relation] = {**prf(tp, np - tp, ng - tp),
                               "gt_count": ng, "pred_count": np}
        relation_details[relation] = {
            "predictions": predictions,
            "missing_gt_edges": [
                {"source_id": gt.objects[source].id,
                 "target_id": (gt.background["id"] if target[0] == "background"
                               else gt.objects[target[1]].id)}
                for (source, target), count in remaining.items() for _ in range(count)]}

    order_counts = Counter(pred.order)
    ranks = {oid: i for i, oid in enumerate(pred.order) if order_counts[oid] == 1}
    order_details = []
    correct = matched_pairs = ordered_pairs = 0
    for lower, upper in occlusion_pairs or []:
        p_lower = g_to_p.get(gt_unique[lower])
        p_upper = g_to_p.get(gt_unique[upper])
        outcome = "unmatched_endpoint"
        if p_lower is not None and p_upper is not None:
            matched_pairs += 1
            lo, hi = pred.objects[p_lower].id, pred.objects[p_upper].id
            if lo in ranks and hi in ranks:
                ordered_pairs += 1
                if ranks[lo] < ranks[hi]:
                    correct += 1
                    outcome = "correct"
                else:
                    outcome = "reversed"
            else:
                outcome = "missing_or_duplicate_order_id"
        order_details.append({"lower": lower, "upper": upper, "outcome": outcome})
    pair_count = len(occlusion_pairs or [])
    order = {"reviewed": occlusion_pairs is not None, "gt_pairs": pair_count,
             "correct": correct, "matched_pairs": matched_pairs, "ordered_pairs": ordered_pairs,
             "accuracy": ratio(correct, pair_count),
             "conditional_accuracy": ratio(correct, ordered_pairs),
             "pair_coverage": ratio(ordered_pairs, pair_count)}
    auxiliary = {}
    for kind in KINDS:
        g = [o for o in gt.objects if o.kind == kind]
        p = [o for o in pred.objects if o.kind == kind]
        soft_matches = match(g, p, threshold=None)
        soft_sum = sum(v for _, _, v in soft_matches)
        auxiliary[kind] = {"gt_count": len(g), "pred_count": len(p),
                           "iou_sum": soft_sum,
                           "iou_weighted_f1": ratio(2 * soft_sum, len(g) + len(p))}
    bg_pred = pred.background.get("type")
    bg_pred = bg_pred if bg_pred in BACKGROUND_TYPES else INVALID_KIND
    ties = potential_ties(iou_matrix(gt.objects, pred.objects), diagnostic)

    def describe(matches):
        return [{"gt_index": g, "gt_id": gt.objects[g].id, "gt_kind": gt.objects[g].kind,
                 "pred_index": p, "pred_id": pred.objects[p].id,
                 "pred_kind": pred.objects[p].kind, "iou": v} for g, p, v in matches]

    return {
        "id": sample_id, "detection": detection,
        "classification": {"correct": class_correct, "matched": n_matched,
                           "accuracy": ratio(class_correct, n_matched), "confusion": matrix},
        "localization": {"iou_sum": iou_sum, "matched": n_matched,
                         "mean_iou": ratio(iou_sum, n_matched),
                         "coverage": ratio(n_matched, n_gt)},
        "per_gt_class": per_class, "relations": relations, "occlusion_order": order,
        "background": {"gt": gt.background["type"], "pred": bg_pred,
                       "correct": gt.background["type"] == bg_pred},
        "validity": {"present": pred.present, "json_valid": pred.json_valid,
                     "structure_valid": pred.valid, "errors": pred.errors},
        "auxiliary_iou_f1": auxiliary,
        "details": {"diagnostic_matches": describe(diagnostic),
                    "structural_matches": describe(structural),
                    "missed_gt_ids": [o.id for o in gt.objects
                                      if o.index not in {g for g, _, _ in diagnostic}],
                    "unmatched_prediction_indices": [
                        o.index for o in pred.objects if o.index not in {p for _, p, _ in diagnostic}],
                    "potential_geometry_ties": ties,
                    "relations": relation_details, "occlusion_pairs": order_details},
    }


def aggregate(samples):
    if not samples:
        raise ValueError("GT evaluation set is empty")
    count = len(samples)
    det = {k: sum(s["detection"][k] for s in samples) for k in ("tp", "fp", "fn")}
    detection = prf(**det)
    detection.update({"gt_count": det["tp"] + det["fn"], "pred_count": det["tp"] + det["fp"]})
    matrix = confusion()
    for s in samples:
        for g in KINDS:
            for p in (*KINDS, INVALID_KIND):
                matrix[g][p] += s["classification"]["confusion"][g][p]
    class_correct = sum(matrix[k][k] for k in KINDS)
    iou_sum = sum(s["localization"]["iou_sum"] for s in samples)
    per_class = {}
    for k in KINDS:
        ng = sum(s["per_gt_class"][k]["gt_count"] for s in samples)
        matched = sum(matrix[k].values())
        per_class[k] = {"gt_count": ng, "matched": matched,
                        "recall": ratio(matched, ng),
                        "classification_accuracy": ratio(matrix[k][k], matched)}
    relations = {}
    for r in RELATIONS:
        totals = {k: sum(s["relations"][r][k] for s in samples) for k in ("tp", "fp", "fn")}
        relations[r] = {**prf(**totals), "gt_count": totals["tp"] + totals["fn"],
                        "pred_count": totals["tp"] + totals["fp"],
                        "images_with_gt": sum(s["relations"][r]["gt_count"] > 0 for s in samples),
                        "images_with_predictions": sum(s["relations"][r]["pred_count"] > 0 for s in samples)}
    order = {k: sum(s["occlusion_order"][k] for s in samples)
             for k in ("gt_pairs", "correct", "matched_pairs", "ordered_pairs")}
    order.update({"reviewed_images": sum(s["occlusion_order"]["reviewed"] for s in samples),
                  "unreviewed_images": sum(not s["occlusion_order"]["reviewed"] for s in samples),
                  "images_with_pairs": sum(s["occlusion_order"]["gt_pairs"] > 0 for s in samples),
                  "accuracy": ratio(order["correct"], order["gt_pairs"]),
                  "conditional_accuracy": ratio(order["correct"], order["ordered_pairs"]),
                  "pair_coverage": ratio(order["ordered_pairs"], order["gt_pairs"])})
    bg_matrix = {g: {p: 0 for p in (*BACKGROUND_TYPES, INVALID_KIND)} for g in BACKGROUND_TYPES}
    for s in samples:
        bg_matrix[s["background"]["gt"]][s["background"]["pred"]] += 1
    bg_correct = sum(s["background"]["correct"] for s in samples)
    validity = {k: sum(s["validity"][k] for s in samples)
                for k in ("present", "json_valid", "structure_valid")}
    validity.update({f"{k}_rate": v / count for k, v in list(validity.items())})
    auxiliary = {}
    for k in KINDS:
        totals = {v: sum(s["auxiliary_iou_f1"][k][v] for s in samples)
                  for v in ("gt_count", "pred_count", "iou_sum")}
        totals["iou_weighted_f1"] = ratio(2 * totals["iou_sum"], totals["gt_count"] + totals["pred_count"])
        auxiliary[k] = totals
    warnings = []
    if order["unreviewed_images"]:
        warnings.append(f"{order['unreviewed_images']} images lack reviewed occlusion_pairs; "
                        "order scores cover only annotated pairs, or are N/A.")
    tie_images = sum(bool(s["details"]["potential_geometry_ties"]) for s in samples)
    if tie_images:
        warnings.append(f"{tie_images} images have potential equal-IoU identity ambiguity; inspect details.")
    return {
        "num_examples": count, "detection": detection,
        "classification": {"accuracy": ratio(class_correct, det["tp"]),
                           "correct": class_correct, "matched": det["tp"], "confusion": matrix},
        "localization": {"mean_iou": ratio(iou_sum, det["tp"]), "iou_sum": iou_sum,
                         "matched": det["tp"], "coverage": ratio(det["tp"], detection["gt_count"])},
        "per_gt_class": per_class, "relations": relations, "occlusion_order": order,
        "background": {"accuracy": bg_correct / count, "correct": bg_correct,
                       "total": count, "confusion": bg_matrix},
        "validity": validity, "auxiliary_iou_f1": auxiliary,
        "potential_geometry_tie_images": tie_images, "warnings": warnings,
        "per_template_macro": per_template_macro(samples),
    }
