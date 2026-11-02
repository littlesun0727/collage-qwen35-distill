"""Read the existing delivery manifest, structure directories, or JSONL envelopes."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

from .schema import identifier, strict_loads


@dataclass
class Record:
    id: str
    raw: object
    meta: dict = field(default_factory=dict)
    source: str = ""
    source_hash: str = ""
    error: str | None = None
    aliases: list = field(default_factory=list)


def digest(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_structure(sample_id, path, meta=None):
    path = Path(path)
    record = Record(sample_id, None, meta or {}, str(path), aliases=[path.parent.name])
    try:
        data = path.read_bytes()
        record.source_hash = hashlib.sha256(data).hexdigest()
        record.raw = strict_loads(data.decode("utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as exc:
        record.error = f"cannot read structure: {exc}"
    return record


def add_unique(records, record):
    if not identifier(record.id):
        raise ValueError("every record needs a nonempty string ID")
    if record.id in records:
        raise ValueError(f"duplicate record ID: {record.id}")
    records[record.id] = record


def load_records(path):
    path = Path(path).resolve()
    records, metadata = {}, {"input": str(path)}
    if path.is_file() and path.name == "manifest.json":
        path = path.parent
    if path.is_dir():
        manifest = path / "manifest.json"
        if manifest.exists():
            value = strict_loads(manifest.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
                raise ValueError("delivery manifest must have a cases list")
            metadata.update({k: value.get(k) for k in ("dataset", "annotation_status")})
            metadata["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            if "total" in value and value["total"] != len(value["cases"]):
                raise ValueError("manifest total does not equal cases count")
            for entry in value["cases"]:
                if not isinstance(entry, dict):
                    raise ValueError("manifest case must be an object")
                sample_id = entry.get("test_id", entry.get("id"))
                ref = entry.get("structure.json")
                if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
                    raise ValueError(f"missing structure.json path in manifest for {sample_id}")
                record = read_structure(sample_id, path / ref["path"], entry)
                if ref.get("sha256") and record.source_hash and ref["sha256"] != record.source_hash:
                    raise ValueError(f"structure file SHA-256 does not match manifest: {sample_id}")
                add_unique(records, record)
        else:
            files = sorted(path.rglob("structure.json"))
            if not files:
                files = sorted(path.glob("*.json"))
            for f in files:
                sample_id = f.parent.name if f.name == "structure.json" else f.stem
                add_unique(records, read_structure(sample_id, f))
    elif path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                envelope = strict_loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL envelope: {exc}") from exc
            if not isinstance(envelope, dict) or not identifier(envelope.get("id")):
                raise ValueError(f"{path}:{line_number}: envelope requires string id")
            if "structure" in envelope and "structure_path" in envelope:
                raise ValueError(f"{path}:{line_number}: choose structure OR structure_path")
            if "structure_path" in envelope:
                if not isinstance(envelope["structure_path"], str):
                    raise ValueError("structure_path must be a string")
                record = read_structure(envelope["id"], path.parent / envelope["structure_path"], envelope)
            else:
                record = Record(envelope["id"], envelope.get("structure"), envelope,
                                f"{path}:{line_number}", digest(envelope))
                if "structure" not in envelope:
                    record.error = "missing structure field"
            add_unique(records, record)
    elif path.is_file() and path.suffix.lower() == ".json":
        sample_id = path.parent.name if path.name == "structure.json" else path.stem
        add_unique(records, read_structure(sample_id, path))
    else:
        raise ValueError(f"unsupported/missing input: {path}")
    return records, metadata


def align_prediction_ids(predictions, gt):
    aliases = {}
    for sample_id, record in gt.items():
        for alias in record.aliases:
            aliases.setdefault(alias, set()).add(sample_id)
    aligned = {}
    for sample_id, record in predictions.items():
        if sample_id in gt:
            target = sample_id
        else:
            candidates = aliases.get(sample_id, set())
            if len(candidates) != 1:
                raise ValueError(f"unexpected or ambiguous prediction ID: {sample_id}")
            target = next(iter(candidates))
        if target in aligned:
            raise ValueError(f"duplicate prediction after ID alignment: {target}")
        aligned[target] = record
    return aligned


def load_occlusions(path, gt):
    if path is None:
        return {}
    pairs = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = strict_loads(line)
        if not isinstance(row, dict) or not identifier(row.get("id")):
            raise ValueError(f"occlusion line {line_number}: requires id")
        sample_id = row["id"]
        if sample_id not in gt or sample_id in pairs:
            raise ValueError(f"unknown/duplicate occlusion ID: {sample_id}")
        if "occlusion_pairs" not in row:
            raise ValueError(f"occlusion line {line_number}: requires occlusion_pairs")
        pairs[sample_id] = row["occlusion_pairs"]
    return pairs


def records_hash(records):
    return digest([{"id": key, "source_hash": r.source_hash or digest(r.raw), "error": r.error,
                    "coordinate_mode": r.meta.get("coordinate_mode", "normalized_1000"),
                    "width": r.meta.get("width"), "height": r.meta.get("height"),
                    "occlusion_pairs": r.meta.get("occlusion_pairs")}
                   for key, r in sorted(records.items())])
