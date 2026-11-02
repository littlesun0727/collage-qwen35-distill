"""Validate scored structure fields; preserve malformed predictions for scoring."""

from collections import Counter
from dataclasses import dataclass, field
import json
import math
import re

KINDS = ("photo", "frame", "overlay", "text")
BACKGROUND_TYPES = ("photographic", "reference_clean", "solid_color", "transparent")
RELATIONS = ("photo_id", "linked_to")
INVALID_KIND = "__invalid__"


def strict_loads(text):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def identifier(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass
class Element:
    index: int
    id: str | None
    kind: str
    box: list | None


@dataclass
class Edge:
    source: int
    target: object
    syntax_valid: bool = True


@dataclass
class Structure:
    objects: list = field(default_factory=list)
    background: dict = field(default_factory=dict)
    order: list = field(default_factory=list)
    edges: dict = field(default_factory=lambda: {r: [] for r in RELATIONS})
    errors: list = field(default_factory=list)
    json_valid: bool = True
    present: bool = True

    @property
    def valid(self):
        return self.json_valid and not self.errors

    def unique_ids(self):
        counts = Counter(o.id for o in self.objects if o.id is not None)
        return {o.id: o.index for o in self.objects
                if o.id is not None and counts[o.id] == 1}


def parse_structure(raw, *, present=True, error=None, coordinate_mode="normalized_1000",
                    width=None, height=None):
    s = Structure(present=present)
    if error or not present:
        s.json_valid = False
        s.errors.append(error or "missing_prediction")
        return s
    if isinstance(raw, str):
        try:
            raw = strict_loads(raw)
        except (ValueError, TypeError) as exc:
            s.json_valid = False
            s.errors.append(f"invalid_json: {exc}")
            return s
    if not isinstance(raw, dict):
        s.errors.append("structure must be a JSON object")
        return s

    mode = raw.get("coordinate_mode", coordinate_mode)
    scales = {"normalized_1000": (1.0, 1.0), "normalized_1": (1000.0, 1000.0)}
    if mode == "pixel":
        if all(isinstance(x, (int, float)) and not isinstance(x, bool)
               and math.isfinite(x) and x > 0 for x in (width, height)):
            scales["pixel"] = (1000 / width, 1000 / height)
        else:
            s.errors.append("pixel coordinates require positive finite width and height")
    scale = scales.get(mode) if isinstance(mode, str) else None
    if scale is None:
        s.errors.append(f"unsupported/unusable coordinate_mode: {mode!r}")
    objects = raw.get("objects")
    if not isinstance(objects, list):
        s.errors.append("objects must be a list")
        objects = []
    raw_objects = []
    for i, item in enumerate(objects):
        if not isinstance(item, dict):
            s.errors.append(f"objects[{i}] must be an object")
            item = {}
        raw_objects.append(item)
        oid = item.get("id")
        if not identifier(oid):
            s.errors.append(f"objects[{i}].id must be a nonempty string")
            oid = None
        kind = item.get("kind")
        if kind not in KINDS:
            s.errors.append(f"objects[{i}].kind is unknown")
            kind = INVALID_KIND
        box = item.get("bbox")
        if (scale is not None and isinstance(box, list) and len(box) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        and math.isfinite(v) for v in box)):
            box = [float(v) * scale[j % 2] for j, v in enumerate(box)]
            if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
                box = None
        else:
            box = None
        if box is None:
            s.errors.append(f"objects[{i}].bbox is invalid (no clipping or repair)")
        s.objects.append(Element(i, oid, kind, box))

    ids = [o.id for o in s.objects if o.id is not None]
    if len(set(ids)) != len(ids):
        s.errors.append("object IDs must be unique")
    bg = raw.get("background")
    if not isinstance(bg, dict):
        s.errors.append("background must be an object")
        bg = {}
    s.background = bg
    bg_type = bg.get("type")
    if bg_type not in BACKGROUND_TYPES:
        s.errors.append("unknown/missing background.type")
    if bg_type in ("photographic", "reference_clean"):
        if not identifier(bg.get("id")):
            s.errors.append("photographic/reference_clean background requires id")
        elif bg["id"] in ids:
            s.errors.append("background.id must not duplicate an object ID")
    if bg_type == "solid_color":
        if not isinstance(bg.get("color"), str) or not re.fullmatch(
                r"#[0-9a-fA-F]{6}", bg["color"]):
            s.errors.append("solid_color background requires #RRGGBB color")

    order = raw.get("layer_order")
    if not isinstance(order, list) or not all(identifier(x) for x in order):
        s.errors.append("layer_order must be a list of nonempty string IDs")
    else:
        s.order = order
        if len(set(order)) != len(order):
            s.errors.append("layer_order contains duplicate IDs")
        if set(order) != set(ids) or len(order) != len(s.objects):
            s.errors.append("layer_order must contain every object ID exactly once")

    unique = s.unique_ids()
    for i, item in enumerate(raw_objects):
        if item.get("kind") == "frame" and not item.get("photo_id"):
            s.errors.append(f"objects[{i}] frame requires nonempty photo_id")
        for relation in RELATIONS:
            if relation not in item:
                continue
            value = item[relation]
            syntax_valid = isinstance(value, list) if relation == "photo_id" else identifier(value)
            targets = value if relation == "photo_id" and isinstance(value, list) else [value]
            if not syntax_valid or not targets:
                s.errors.append(f"objects[{i}].{relation} has invalid format")
            if item.get("kind") != ("frame" if relation == "photo_id" else "photo"):
                s.errors.append(f"objects[{i}].{relation} has an invalid source kind")
                syntax_valid = False
            seen = set()
            for target in targets:
                edge_ok = syntax_valid and identifier(target)
                if identifier(target):
                    if target in seen:
                        s.errors.append(f"objects[{i}].{relation} contains duplicate target")
                    seen.add(target)
                    target_idx = unique.get(target)
                    bg_target = (relation == "linked_to" and bg_type == "photographic"
                                 and target == bg.get("id") and target not in ids)
                    valid_obj = (target_idx is not None and target_idx != i
                                 and s.objects[target_idx].kind == "photo")
                    if not (bg_target or valid_obj):
                        edge_ok = False
                else:
                    edge_ok = False
                if not edge_ok:
                    s.errors.append(f"objects[{i}].{relation} has invalid reference: {target!r}")
                s.edges[relation].append(Edge(i, target, edge_ok))
    return s


def validate_occlusions(pairs, gt):
    """None = not reviewed; [] = reviewed, with no applicable occlusion pairs."""
    if pairs is None:
        return
    if not isinstance(pairs, list):
        raise ValueError("occlusion_pairs must be a list, or omitted")
    ids = gt.unique_ids()
    rank = {oid: i for i, oid in enumerate(gt.order)}
    seen = set()
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or not all(identifier(x) and x in ids for x in pair)):
            raise ValueError(f"invalid occlusion pair: {pair!r}")
        lower, upper = pair
        if lower == upper or tuple(pair) in seen:
            raise ValueError(f"self/duplicate occlusion pair: {pair!r}")
        if rank[lower] >= rank[upper]:
            raise ValueError(f"occlusion pair contradicts GT layer_order: {pair!r}")
        seen.add(tuple(pair))
