"""Five primary dimensions; v1 metrics remain available as diagnostics."""

from .matching import match


def grouped_metrics(gt, pred):
    result = {}
    for name, kinds in (("photo", {"photo"}), ("fixed", {"frame", "overlay", "text"})):
        g = [o for o in gt.objects if o.kind in kinds]
        p = [o for o in pred.objects if o.kind in kinds]
        denominator = len(g) + len(p)
        soft = match(g, p, threshold=None)
        result[name] = {
            "gt_count": len(g), "pred_count": len(p),
            "f1_50": 2 * len(match(g, p, .5)) / denominator if denominator else None,
            "f1_75": 2 * len(match(g, p, .75)) / denominator if denominator else None,
            "soft_iou_f1": 2 * sum(v for _, _, v in soft) / denominator if denominator else None,
            "soft_matches": [{"gt_id": g[a].id, "pred_id": p[b].id, "iou": v}
                             for a, b, v in soft],
        }
    return result


def summarize(samples, legacy, prediction_stage):
    def mean(group, key):
        values = [s["grouped_metrics"][group][key] for s in samples
                  if s["grouped_metrics"][group][key] is not None]
        return {"mean": sum(values) / len(values) if values else None,
                "eligible_images": len(values), "total_images": len(samples)}

    macro = legacy["per_template_macro"]
    return {
        "output_usability": {
            "first_pass_core_valid_rate": (legacy["validity"]["structure_valid_rate"]
                                          if prediction_stage == "raw_first" else None),
            "final_core_valid_rate": (legacy["validity"]["structure_valid_rate"]
                                      if prediction_stage == "final_accepted" else None),
            "prediction_stage": prediction_stage,
            "scope": "Strict JSON and eval/schema.py core contract; not full upstream schema",
        },
        "recognition": {**{g: mean(g, "f1_50") for g in ("photo", "fixed")},
                        "background_accuracy": macro["background"]["accuracy"]},
        "localization": {g: {k: mean(g, k) for k in ("soft_iou_f1", "f1_75")}
                         for g in ("photo", "fixed")},
        "treatment": {
            "kind_agreement_proxy": macro["classification"]["accuracy"],
            "matched_gt_coverage": macro["localization"]["coverage"],
            "reviewed_treatment_accuracy": None,
            "note": "Kind agreement on geometry matches is a proxy, not reviewed reconstruction suitability",
        },
        "relations": {**macro["relations"], "occlusion_order": macro["occlusion_order"]},
    }


def report_rows(summary):
    p = summary["primary"]
    rows = [("① 首次核心结构合法率", p["output_usability"]["first_pass_core_valid_rate"], summary["num_examples"])]
    rows.append(("① 修复流程后核心结构合法率", p["output_usability"]["final_core_valid_rate"], summary["num_examples"]))
    for g, label in (("photo", "照片"), ("fixed", "固定设计")):
        v = p["recognition"][g]
        rows.append((f"② {label} F1@0.50", v["mean"], v["eligible_images"]))
    v = p["recognition"]["background_accuracy"]
    rows.append(("② 背景类型准确率", v["mean"], v["eligible_images"]))
    for g, label in (("photo", "照片"), ("fixed", "固定设计")):
        for key, metric in (("soft_iou_f1", "软 IoU-F1"), ("f1_75", "F1@0.75")):
            v = p["localization"][g][key]
            rows.append((f"③ {label} {metric}", v["mean"], v["eligible_images"]))
    v = p["treatment"]["kind_agreement_proxy"]
    rows.append(("④ 处理方式代理：匹配类别一致率", v["mean"], v["eligible_images"]))
    v = p["treatment"]["matched_gt_coverage"]
    rows.append(("④ 匹配覆盖率（条件分数配套）", v["mean"], v["eligible_images"]))
    rows.append(("④ 人工审核处理方式正确率", None, 0))
    for key in ("photo_id", "linked_to"):
        v = p["relations"][key]["f1"]
        rows.append((f"⑤ {key} F1", v["mean"], v["eligible_images"]))
    v = p["relations"]["occlusion_order"]["accuracy"]
    rows.append(("⑤ 确定遮挡对正确率", v["mean"], v["eligible_images"]))
    return rows


def markdown_primary(summary):
    rows = report_rows(summary)
    lines = ["# 五项主指标（v2）", "",
             "首次合法率仅用于 raw_first 输入；修复流程后合法率仅用于 final_accepted 输入（含程序规范化和模型修复）。合法率检查核心结构契约，不是完整上游 schema。",
             "首次响应的可解析有效部分继续评分，不因其他字段非法整图归零。最终流程失败按缺失预测计分：GT有元素的组F1和软IoU-F1为0；无匹配的条件mIoU/类别分为N/A，必须结合覆盖率。",
             "固定设计合并 frame/overlay/text 做几何匹配，类别差异由处理方式代理体现。尚无人工等价分组标注，不自动合并或忽略元素。",
             "所有均值按适用图片计算；失败预测保留。当前教师参考未经核验时，分数只表示一致性。",
             "", "| 指标 | 逐图均值 | 适用图片数 |", "| --- | --- | --- |"]
    for label, value, n in rows:
        display = "N/A" if value is None else f"{100 * value:.2f}%"
        lines.append(f"| {label} | {display} | {n}/{summary['num_examples']} |")
    return "\n".join(lines) + "\n"
