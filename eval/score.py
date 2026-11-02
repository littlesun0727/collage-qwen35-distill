"""CLI: python -m eval.score --gt DELIVERY --pred PREDICTIONS --out runs/eval/run1"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy
import scipy

from . import PROTOCOL_VERSION
from .io import align_prediction_ids, digest, load_occlusions, load_records, records_hash
from .metrics import aggregate, evaluate_sample
from .schema import KINDS, parse_structure
from .primary import grouped_metrics, summarize, markdown_primary


def parse_record(record):
    if record is None:
        return parse_structure(None, present=False)
    return parse_structure(
        record.raw, error=record.error,
        coordinate_mode=record.meta.get("coordinate_mode", "normalized_1000"),
        width=record.meta.get("width"), height=record.meta.get("height"))


def evaluate_dataset(gt, predictions, *, occlusions=None, threshold=0.5, require_occlusions=False,
                     prediction_stage="unspecified"):
    if not gt:
        raise ValueError("GT evaluation set is empty")
    predictions = align_prediction_ids(predictions, gt)
    occlusions = occlusions or {}
    if set(occlusions) - gt.keys():
        raise ValueError("occlusion annotations contain unknown IDs")
    samples, effective_pairs = [], {}
    for sample_id in sorted(gt):
        record = gt[sample_id]
        inline = record.meta.get("occlusion_pairs")
        if sample_id in occlusions and inline is not None and inline != occlusions[sample_id]:
            raise ValueError(f"conflicting inline and sidecar occlusion annotations: {sample_id}")
        pairs = occlusions.get(sample_id, inline)
        if require_occlusions and pairs is None:
            raise ValueError(f"missing reviewed occlusion annotation: {sample_id}")
        effective_pairs[sample_id] = pairs
        parsed_gt, parsed_pred = parse_record(record), parse_record(predictions.get(sample_id))
        sample = evaluate_sample(sample_id, parsed_gt, parsed_pred, pairs, threshold)
        sample["grouped_metrics"] = grouped_metrics(parsed_gt, parsed_pred)
        samples.append(sample)
    summary = aggregate(samples)
    summary["primary"] = summarize(samples, summary, prediction_stage)
    summary["protocol"] = {
        "version": PROTOCOL_VERSION, "match_iou_threshold": threshold,
        "diagnostic_matching": "class_agnostic_max_cardinality_then_iou",
        "structural_matching": "class_aware_max_cardinality_then_iou",
        "auxiliary_matching": "per_class_max_sum_iou_without_hard_threshold",
        "aggregation": "top-level metrics: pooled counts/IoUs; per_template_macro: "
                       "equal-weight mean over defined per-image scores",
        "recommended_primary_view": "primary: five dimensions; legacy metrics retained for diagnostics",
        "primary_detection_iou": 0.5,
        "primary_precise_iou": 0.75,
        "primary_groups": {"photo": ["photo"], "fixed": ["frame", "overlay", "text"]},
        "equivalent_group_annotations": "unavailable; strict instance units",
        "empty_denominator": "null (N/A); no empty-image or empty-class bonuses",
        "default_coordinate_mode": "normalized_1000",
        "gt_records_sha256": records_hash(gt),
        "prediction_records_sha256": records_hash(predictions),
        "occlusion_annotations_sha256": digest(effective_pairs),
    }
    return summary, samples


def percentage(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def markdown_report(s):
    d, c, loc, rel, order = (s[k] for k in
                            ("detection", "classification", "localization", "relations", "occlusion_order"))
    macro = s["per_template_macro"]
    rows = [
        ("主指标① 检测 F1（不看类别）", percentage(d["f1"]),
         f"GT={d['gt_count']}，预测={d['pred_count']}；P={percentage(d['precision'])}，R={percentage(d['recall'])}"),
        ("主指标② 匹配元素分类准确率", percentage(c["accuracy"]), f"匹配元素={c['matched']}"),
        ("主指标③ 匹配元素平均 IoU", percentage(loc["mean_iou"]), f"覆盖率={percentage(loc['coverage'])}"),
        ("主指标④ 相框关联 F1", percentage(rel["photo_id"]["f1"]),
         f"GT关系={rel['photo_id']['gt_count']}；GT图片={rel['photo_id']['images_with_gt']}"),
        ("主指标④ 出框关联 F1", percentage(rel["linked_to"]["f1"]),
         f"GT关系={rel['linked_to']['gt_count']}；GT图片={rel['linked_to']['images_with_gt']}"),
        ("主指标⑤ 确定遮挡对正确率", percentage(order["accuracy"]),
         f"GT对数={order['gt_pairs']}；已审核图片={order['reviewed_images']}/{s['num_examples']}"),
        ("基础检查 背景类型准确率", percentage(s["background"]["accuracy"]), f"图片={s['num_examples']}"),
        ("基础检查 JSON合法率", percentage(s["validity"]["json_valid_rate"]), ""),
        ("基础检查 核心结构合法率", percentage(s["validity"]["structure_valid_rate"]), ""),
    ]
    for kind in KINDS:
        item = s["auxiliary_iou_f1"][kind]
        rows.append((f"辅助 {kind} IoU-weighted F1", percentage(item["iou_weighted_f1"]),
                     f"GT={item['gt_count']}；预测={item['pred_count']}"))
    macro_values = [
        macro["detection"]["f1"], macro["classification"]["accuracy"],
        macro["localization"]["mean_iou"], macro["relations"]["photo_id"]["f1"],
        macro["relations"]["linked_to"]["f1"], macro["occlusion_order"]["accuracy"],
        macro["background"]["accuracy"], macro["validity"]["json_valid_rate"],
        macro["validity"]["structure_valid_rate"],
    ] + [macro["auxiliary_iou_f1"][kind] for kind in KINDS]
    text = ["# 拼贴结构评测报告", "", f"协议：{PROTOCOL_VERSION}；样本数：{s['num_examples']}。",
            "", "所有分数衡量与所提供 GT 的一致性；教师候选标注自检不代表模型真实能力。",
            "", "主看逐模板均值：每个适用模板权重相同。全量累计保留作为对照，元素/关系多的模板权重更大。",
            "分类准确率与平均 IoU 是条件分数：无匹配的图不进入条件均值，必须同时看检测、覆盖率与适用图片数。",
            "", "| 指标 | 逐模板均值 | 适用图片数 | 全量累计 | 全量分母与覆盖 |",
            "| --- | --- | --- | --- | --- |"]
    text += [f"| {name} | {percentage(m['mean'])} | {m['eligible_images']}/{m['total_images']} "
             f"| {value} | {support} |"
             for (name, value, support), m in zip(rows, macro_values)]
    text += ["", f"逐模板平均匹配覆盖率：{percentage(macro['localization']['coverage']['mean'])}；"
             f"逐模板平均检测召回率：{percentage(macro['detection']['recall']['mean'])}。",
             "每模板先算 F1 再平均，不是对平均 Precision / Recall 再算 F1；"
             "没有任何关系的图不奖励满分，但无 GT 却虚构关系的图以 F1=0 进入均值。"]
    text += ["", "## 分类混淆矩阵（仅已匹配元素）", "",
             "| GT / 预测 | photo | frame | overlay | text | invalid |",
             "| --- | --- | --- | --- | --- | --- |"]
    for kind, counts in c["confusion"].items():
        text.append(f"| {kind} | " + " | ".join(str(v) for v in counts.values()) + " |")
    text += ["", f"已匹配且顺序可判定的元素对：{order['ordered_pairs']}；"
             f"条件排序正确率：{percentage(order['conditional_accuracy'])}；"
             f"对覆盖率：{percentage(order['pair_coverage'])}。",
             "", f"缺失预测：{s['num_examples'] - s['validity']['present']}。",
             "", "逐图匹配、漏检、错连、反序及合法性错误见 per_sample.jsonl；"
             "协议、输入指纹与运行信息见 summary.json。"]
    if s["warnings"]:
        text += ["", "## 覆盖与输入提示", ""] + [f"- {w}" for w in s["warnings"]]
    diagnostic_text = "\n".join(text).replace("# 拼贴结构评测报告", "# 兼容 v1 诊断报告")
    diagnostic_text = diagnostic_text.replace("主指标", "旧版诊断")
    return markdown_primary(s) + "\n---\n\n" + diagnostic_text + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--occlusions", type=Path)
    parser.add_argument("--require-occlusions", action="store_true")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--prediction-stage", choices=("unspecified", "raw_first", "final_accepted"), default="unspecified",
                        help="Declare raw first responses only; never label repaired outputs raw_first")
    args = parser.parse_args(argv)
    try:
        gt, gt_meta = load_records(args.gt)
        predictions, pred_meta = load_records(args.pred)
        pairs = load_occlusions(args.occlusions, gt)
        summary, samples = evaluate_dataset(
            gt, predictions, occlusions=pairs, threshold=args.iou_threshold,
            require_occlusions=args.require_occlusions, prediction_stage=args.prediction_stage)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    code_dir = Path(__file__).parent
    summary["run"] = {
        "name": args.run_name, "model": args.model,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
        "gt_input": gt_meta, "prediction_input": pred_meta,
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(code_dir.glob("*.py"))},
    }
    if gt_meta.get("annotation_status") in ("teacher_prediction_unreviewed", "mixed_candidate_annotations_unreviewed"):
        summary["warnings"].append("GT contains unreviewed candidate annotations; scores measure agreement, not verified accuracy.")
    if summary["protocol"]["gt_records_sha256"] == summary["protocol"]["prediction_records_sha256"]:
        summary["warnings"].append("GT and predictions have identical fingerprints: this is a self-consistency check.")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.out / "per_sample.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False, allow_nan=False) + "\n" for s in samples),
        encoding="utf-8")
    (args.out / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({"num_examples": summary["num_examples"], "output": str(args.out.resolve()),
                      "detection_f1_micro": summary["detection"]["f1"],
                      "detection_f1_per_template": summary["per_template_macro"]["detection"]["f1"]["mean"],
                      "occlusion_order_per_template": summary["per_template_macro"]["occlusion_order"]["accuracy"]["mean"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
