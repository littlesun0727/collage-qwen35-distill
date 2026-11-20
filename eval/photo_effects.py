"""SFT v3 photo effects. Match instances before scoring their attributes."""

from collections import Counter, defaultdict
from math import atan2, degrees, hypot, isfinite
import re

import numpy as np
from PIL import Image, ImageDraw

from .matching import match
from .metrics import prf, ratio

FIELDS = (
    "mode",
    "shape",
    "radius",
    "star_points",
    "points",
    "corners",
    "axis",
    "stroke",
    "fade",
    "popout",
)
CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")
SIDES = ("left", "right", "top", "bottom")
SHAPES = ("rectangle", "circle", "ellipse", "heart", "star", "other")
MODES = ("window", "cutout", "bare")
CLASSES = tuple("mode." + m for m in MODES) + tuple("shape." + s for s in SHAPES)
VERSION = "collage-photo-effects-v3.1"


def number(x, low, high):
    return type(x) in (float, int) and low <= x <= high and isfinite(x)


def color(x, alpha=False):
    return isinstance(x, str) and re.fullmatch(
        r"#[0-9a-fA-F]{6}" + ("(?:[0-9a-fA-F]{2})?" if alpha else ""), x
    )


def coordinates(value, scale, length):
    return (
        isinstance(value, list)
        and len(value) == length
        and scale is not None
        and all(number(v, 0, 1000 / scale[i % 2] + 1e-8) for i, v in enumerate(value))
    )


def valid_field(k, v, mode, shape, scale):
    window = mode == "window"
    if k == "mode":
        return v in MODES
    if k == "shape":
        return window and v in SHAPES
    if k == "radius":
        return window and shape == "rectangle" and number(v, 0, 0.5)
    if k == "star_points":
        return window and shape == "star" and type(v) is int and 3 <= v <= 12
    if k == "points":
        return (
            window
            and shape == "other"
            and isinstance(v, list)
            and 3 <= len(v) <= 32
            and all(coordinates(p, scale, 2) for p in v)
        )
    if k == "corners":
        return (
            window
            and shape == "rectangle"
            and isinstance(v, dict)
            and set(v) == set(CORNERS)
            and all(coordinates(p, scale, 2) for p in v.values())
        )
    if k == "axis":
        return (
            window
            and shape in ("heart", "ellipse", "star")
            and isinstance(v, list)
            and len(v) == 2
            and all(coordinates(p, scale, 2) for p in v)
            and v[0] != v[1]
        )
    if k == "popout":
        return window and coordinates(v, scale, 4) and v[0] < v[2] and v[1] < v[3]
    if k == "stroke":
        return (
            mode in ("window", "cutout")
            and isinstance(v, dict)
            and not set(v) - {"width", "color", "placement", "pattern", "offset"}
            and number(v.get("width"), 0.001, 0.1)
            and color(v.get("color"))
            and v.get("pattern", "solid") in ("solid", "dash", "dot")
            and v.get("placement", "outside") in ("inside", "outside", "center")
            and (mode != "cutout" or "placement" not in v)
            and number(v.get("offset", 0), 0, 0.2)
            and (v.get("placement") != "center" or v.get("offset", 0) == 0)
        )
    if k == "fade":
        return (
            window
            and isinstance(v, dict)
            and not set(v) - {*SIDES, "direction"}
            and any(v.get(s, 0) for s in SIDES)
            and all(number(v.get(s, 0), 0, 0.5) for s in SIDES)
            and v.get("direction", "inside") in ("inside", "outside")
            and (
                v.get("direction") == "outside"
                or max(v.get("left", 0) + v.get("right", 0), v.get("top", 0) + v.get("bottom", 0))
                <= 0.90000001
            )
        )


def v3_fields(obj, scale, errors):
    mode, shape = obj.get("mode", "window"), obj.get("shape", "rectangle")
    fields = {}
    for k in FIELDS:
        if k in obj:
            if valid_field(k, obj[k], mode, shape, scale):
                fields[k] = obj[k]
            else:
                errors.append(k)
    if mode == "window" and shape == "other" and "points" not in obj:
        errors.append("other requires points")
    for k in ("axis", "points", "corners", "popout"):
        if k not in fields:
            continue
        v = fields[k]

        def scaled(point):
            return [n * scale[i % 2] for i, n in enumerate(point)]

        if k == "corners":
            fields[k] = {name: scaled(point) for name, point in v.items()}
        elif k == "popout":
            fields[k] = scaled(v)
        else:
            fields[k] = [scaled(point) for point in v]
    if mode in MODES:
        fields["mode." + mode] = True
    if mode == "window" and shape in SHAPES:
        fields["shape." + shape] = True
    return fields


def attributes(structure):
    raw = structure.raw
    errors = list(structure.errors)
    if not isinstance(raw, dict):
        return {}, errors or ["missing structure"]
    if set(raw) != {"objects", "layer_order", "background"}:
        errors.append("unexpected or missing top-level keys")
    if len(structure.objects) > 160:
        errors.append("too many objects")
    bg = structure.background
    bg_keys = {"type"}
    if bg.get("type") in ("photographic", "reference_clean"):
        bg_keys.update(("id", "description"))
    if bg.get("type") == "solid_color":
        bg_keys.add("color")

    def named(row):
        return isinstance(row.get("id"), str) and re.fullmatch("[a-z][a-z0-9_-]{0,63}", row["id"])

    def described(row):
        return isinstance(row.get("description"), str) and 1 <= len(row["description"]) <= 400

    if set(bg) != bg_keys or ("id" in bg_keys and not (named(bg) and described(bg))):
        errors.append("background fields")
    photos = {}
    for e in structure.objects:
        obj = raw["objects"][e.index]
        if not isinstance(obj, dict):
            continue
        allowed = {"kind", "id", "description", "bbox"}
        issues = []
        if e.kind == "photo":
            allowed.update(FIELDS)
            photos[e.index] = v3_fields(obj, structure.scale, issues)
        if e.kind == "frame":
            allowed.add("photo_id")
        if e.kind == "text":
            allowed.update(("text", "color", "align"))
            if (
                "text" not in obj
                or not isinstance(obj["text"], (str, type(None)))
                or ("color" in obj and not color(obj["color"], True))
                or obj.get("align", "left") not in ("left", "center", "right")
            ):
                issues.append("text attributes")
        if not named(obj) or not described(obj):
            issues.append("id/description")
        issues.extend(sorted(set(obj) - allowed))
        errors.extend(f"objects[{e.index}].{k}" for k in issues)
    return photos, errors


def mask_iou(a, b):
    def mask(poly):
        image = Image.new("1", (512, 512))
        ImageDraw.Draw(image).polygon([tuple(round(v * 0.511) for v in p) for p in poly], fill=1)
        return np.asarray(image)

    left, right = mask(a), mask(b)
    return float((left & right).sum() / (left | right).sum())


def quality(g, p, size):
    values = {}

    def paired(field):
        return field in g and field in p

    if paired("popout"):
        a, b = g["popout"], p["popout"]
        inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        values["popout.iou"] = inter / (
            (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        )
    if paired("points"):
        values["points.mask_iou"] = mask_iou(g["points"], p["points"])

    def point_error(a, b):
        return hypot((a[0] - b[0]) * size[0], (a[1] - b[1]) * size[1]) / (1000 * hypot(*size))

    for field in ("corners", "axis"):
        if not paired(field):
            continue
        if field == "corners":
            a, b = ([d[field][k] for k in CORNERS] for d in (g, p))
            values["corners.mask_iou"] = mask_iou(a, b)
        else:
            if g.get("shape") != p.get("shape"):
                continue
            a, b = g[field], p[field]
        if size:

            def error(points):
                return sum(point_error(x, y) for x, y in zip(a, points)) / len(a)

            v = error(b)
            if field == "axis" and g.get("shape") == "ellipse":
                v = min(v, error(b[::-1]))
            values[field + ".diagonal_mae"] = v
            if field == "axis":

                def angle(pair):
                    return degrees(
                        atan2((pair[1][1] - pair[0][1]) * size[1], (pair[1][0] - pair[0][0]) * size[0])
                    )

                period = 180 if g.get("shape") == "ellipse" else 360
                values["axis.angle_error_deg"] = abs(
                    (angle(a) - angle(b) + period / 2) % period - period / 2
                )
    if paired("radius"):
        values["radius.mae"] = abs(g["radius"] - p["radius"])
    if g.get("shape.star") and p.get("shape.star"):
        values["star_points.accuracy"] = g.get("star_points", 5) == p.get("star_points", 5)
    if paired("stroke"):
        a, b = g["stroke"], p["stroke"]
        for k, default in [("width", 0), ("offset", 0)]:
            values["stroke." + k + "_mae"] = abs(a.get(k, default) - b.get(k, default))
        for k, default in [("pattern", "solid"), ("placement", "outside")]:
            values["stroke." + k + "_accuracy"] = a.get(k, default) == b.get(k, default)
        values["stroke.color_accuracy"] = a["color"].lower() == b["color"].lower()
    if paired("fade"):
        a, b = g["fade"], p["fade"]
        values["fade.direction_accuracy"] = a.get("direction", "inside") == b.get("direction", "inside")
        values["fade.sides_accuracy"] = all((s in a) == (s in b) for s in SIDES)
        values["fade.width_mae"] = sum(
            abs(a.get(s, 0) - b.get(s, 0)) for s in SIDES if s in a or s in b
        ) / sum(s in a or s in b for s in SIDES)
    return values


def evaluate_effects(gt, pred, size=None, finish_reason=None):
    gs, gt_errors = attributes(gt)
    ps, errors = attributes(pred)

    def elements(structure, kind):
        return [e for e in structure.objects if e.kind == kind]

    left, right = elements(gt, "photo"), elements(pred, "photo")
    pairs = [(left[i].index, right[j].index) for i, j, _ in match(left, right, 0.5)]
    paired_gt, paired_pred = {i for i, _ in pairs}, {j for _, j in pairs}
    counts = {}
    # Presence requires a legal predicted value. Malformed attempts still count as predictions.
    for k in (*CLASSES, *FIELDS[2:]):
        targets = {i for i, f in gs.items() if k in f}
        attempts = {i for i, f in ps.items() if k in f or k in pred.raw["objects"][i]}
        if k == "star_points":
            # A five-point star has identical geometry with or without an explicit count.
            targets = {i for i in targets if gs[i][k] != 5}
            attempts = {j for j in attempts if ps[j].get(k) != 5}
        correct = sum(i in targets and j in attempts and k in ps[j] for i, j in pairs)
        counts[k] = [
            len(targets),
            len(attempts),
            correct,
            len(targets & paired_gt),
            len(attempts & paired_pred),
        ]
    a, b = elements(gt, "frame"), elements(pred, "frame")
    counts["frame"] = [len(a), len(b), len(match(a, b, 0.5)), len(a), len(b)]
    measurements = defaultdict(list)
    confusion = {"mode": Counter(), "shape": Counter()}
    for i, j in pairs:
        for k, v in quality(gs[i], ps[j], size).items():
            measurements[k].append(float(v))
        for field in ("mode", "shape"):
            if field == "shape" and not gs[i].get("mode.window"):
                continue

            def category(fields):
                return next(
                    (k.split(".")[1] for k in CLASSES if k.startswith(field + ".") and k in fields),
                    "invalid/inapplicable",
                )

            confusion[field][category(gs[i]) + " -> " + category(ps[j])] += 1
    return dict(
        counts=counts,
        measurements=dict(measurements),
        confusion=confusion,
        errors=errors,
        gt_errors=gt_errors,
        present=pred.present,
        json_valid=pred.json_valid,
        background_correct=gt.background.get("type") == pred.background.get("type"),
        finish_reason=finish_reason,
    )


def summarize_effects(rows, stage):
    rows = list(rows)
    total = len(rows)
    scores = {}
    for k in rows[0]["counts"] if rows else ():
        g, p, tp, mg, mp = map(sum, zip(*(r["counts"][k] for r in rows)))
        scores[k] = dict(
            gt=g,
            predicted=p,
            gt_images=sum(r["counts"][k][0] > 0 for r in rows),
            end_to_end=prf(tp, p - tp, g - tp),
            matched=prf(tp, mp - tp, mg - tp),
        )
    measures = defaultdict(list)
    confusion = {k: Counter() for k in ("mode", "shape")}
    for r in rows:
        for k, v in r["measurements"].items():
            measures[k].extend(v)
        for k, v in r["confusion"].items():
            confusion[k].update(v)
    known = [r for r in rows if r["finish_reason"] is not None]
    valid = ratio(sum(not r["errors"] for r in rows), total)
    return dict(
        version=VERSION,
        stage=stage,
        expected=total,
        present=sum(r["present"] for r in rows),
        missing=sum(not r["present"] for r in rows),
        parsed=sum(r["present"] and r["json_valid"] for r in rows),
        valid_rate=valid,
        background_accuracy=ratio(sum(r["background_correct"] for r in rows), total),
        invalid_json_rate=ratio(
            sum(r["present"] and not r["json_valid"] for r in rows), sum(r["present"] for r in rows)
        ),
        first_pass_valid_rate=valid if stage == "raw_first" else None,
        final_valid_rate=valid if stage == "final_accepted" else None,
        gt_invalid=sum(bool(r["gt_errors"]) for r in rows),
        finish_reason_known=len(known),
        truncated_rate=ratio(sum(r["finish_reason"] == "length" for r in known), len(known)),
        detection=scores,
        confusion=confusion,
        quality={k: {"mean": sum(v) / len(v), "count": len(v)} for k, v in measures.items()},
    )


def has_effects(structure):
    objects = structure.get("objects") if isinstance(structure, dict) else None
    return isinstance(objects, list) and any(
        isinstance(o, dict) and set(FIELDS).intersection(o) for o in objects
    )


def report_lines(summary):
    def fmt(v):
        return "N/A" if v is None else f"{v:.3f}"

    lines = [
        "## v3 照片效果评测",
        "",
        f"Valid {fmt(summary['valid_rate'])} · "
        f"parsed {summary['parsed']}/{summary['expected']} · missing {summary['missing']} · GT invalid {summary['gt_invalid']}",
        "",
        f"背景准确率 {fmt(summary['background_accuracy'])} · 已返回预测的 JSON 解析失败率 {fmt(summary['invalid_json_rate'])} · 截断率 {fmt(summary['truncated_rate'])}（有结束原因 {summary['finish_reason_known']} 条）",
        "",
        "| 字段 | GT 图片 | GT 对象 | 预测对象 | Precision | Recall | F1 | 匹配照片 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, d in summary["detection"].items():
        s = d["end_to_end"]
        lines.append(
            "| "
            + " | ".join(
                map(
                    str,
                    [
                        k,
                        d["gt_images"],
                        d["gt"],
                        d["predicted"],
                        fmt(s["precision"]),
                        fmt(s["recall"]),
                        fmt(s["f1"]),
                        fmt(d["matched"]["f1"]),
                    ],
                )
            )
            + " |"
        )
    lines.extend(["", "| Parameter | Mean | Matched objects |", "|---|---:|---:|"])
    lines.extend(f"| {k} | {fmt(m['mean'])} | {m['count']} |" for k, m in summary["quality"].items())
    return lines
