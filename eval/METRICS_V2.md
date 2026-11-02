# 五项主评测协议 v2.1

v2 为当前主视图，v1 字段保留为诊断和历史兼容。主视图位于 summary.json 的 primary；不合成总分。

| 主项 | 定义 |
| --- | --- |
| ① 输出合法率 | 分开报告原始首次响应与规范化/修复流程后的核心结构合法率，分母均为固定全集。分别认 prediction_stage=raw_first 和 final_accepted；来源不明为 N/A。 |
| ② 元素识别 | photo 和 fixed 分别计算一对一几何匹配 F1@0.50；fixed 包含 frame/overlay/text，组内不区分类别；背景类型准确率单列。 |
| ③ 定位质量 | 两组分别计算软 IoU-F1=2ΣIoU/(GT数+预测数)，匹配最大化正 IoU 总和，不设硬门槛；另报 F1@0.75。 |
| ④ 处理方式 | 当前仅以所有对象的几何匹配@0.50中的类别一致率做代理，同时报告匹配覆盖率；人工审核的处理方式正确率尚未实现，标为 N/A。 |
| ⑤ 关键关系 | 沿用 v1 的 photo_id/linked_to F1 和人工确定遮挡对正确率；不把几何相交当遮挡标注。 |

## 口径与限制

- 主视图均为逐图计算后平均，输出适用图片数。无匹配的条件类别分数为 N/A，不能据此忽略配套覆盖率。
- 缺失、坏 JSON 的预测保留在固定评测集，标注中有元素时对应识别和定位计零。双方均无该组对象为 N/A，不奖励空集。
- 最终交付视图：流程status非complete的样本作为缺失预测，不读取失败响应或可能残留的structure.json；complete但最终文件缺失/非法会报错，先检查来源。覆盖全部固定图片，不只统计修复成功图片。
- 首次诊断视图：结构非法不意味着所有有效部分都归零。例如layer_order错误但照片框有效，照片指标仍计分。这与最终交付失败按缺失处理不同；两列差值不能直接解释为修复带来的视觉质量变化。
- 无预测时：GT有元素的组F1/软IoU-F1为0；背景准确率为0；GT有关系/遮挡约束时其端到端分数为0。双方均无该类元素/关系为N/A；无匹配的条件mIoU及类别一致率为N/A，必须同时报告覆盖率和适用图片数。
- 部分字段非法时，合法性不通过；有效元素仍可计分。不要把模型结构得分误解为整图可执行率。
- photo 与 fixed 跨组不匹配。fixed 中 text/overlay 混淆不再扣几何分，但会降低④的类别一致代理。
- 低于0.5的差框仍能获得软IoU贡献；重复预测占分母；未知类别留在v1检测/分类/合法性诊断中，不得只看分组指标选模型。
- ①只检查现有评分器核心契约，不检查完整上游 schema（如全部description/text样式字段）。原始输出把背景ID放入layer_order会判非法，不自动删除、补齐或去重。
- 目前没有人工认可的等价分组标注，仍按已有实例一对一计分，不能将整组任意合并。这个版本未实现多答案/分组等价评分。
- ④的类别代理并不检验description中的摄影环境保留、抠图语义或重建效果。需人工补标后另行实现完整处理方式评测。
- 主指标阈值固定为0.50/0.75。CLI --iou-threshold 仍只控制v1诊断与关系端点匹配，默认0.50；改变时需注明，不得混比。
- 教师未审核参考下的成绩为一致性，不是能力准确率。历史教师重复运行不是教师自评满分，也不是性能上限；没有首次原始响应证据，不报告教师首次合法率。

## 通用运行

首次视图输入必须是首次原始响应，不能拿经过规范化的 structure.json 或修复后响应冒充首次结果：

```powershell
python -m eval.score --gt gt.jsonl --pred first-responses.jsonl --prediction-stage raw_first --out runs/eval/my-v2
```

不声明 --prediction-stage 时仍计算结构质量，但首次合法率为 N/A。

最终交付视图使用 --prediction-stage final_accepted，输入仅包含最终交付结果；失败图从预测输入中省略，但必须保留在GT全集中，由评分器作为缺失预测计分。

## 当前 covers 基线复现

从仓库根目录运行：

```powershell
python -m scripts.evaluate_covers_v2
```

以 runs/compare/teacher-two-runs-20261030/per_sample.json 的130张图及其当前教师结构为共同参考；9B/4B从对应case的 workflow/vlm/analyze/*/response-0.json 提取content，不读取analyze_repair或accepted结果。多份候选首次响应会报错，缺失响应按缺失预测计分。

v2.1另加入各模型-final列，从status=complete的case读取最终structure.json；包含程序规范化及至多一次模型修复。失败样本也保留在130张的分母中，适用组计零。

输出 runs/eval/covers-v2-first-pass-20261102/index.html、comparison.md、comparison.json，以及各运行summary.json、per_sample.jsonl、sources.json和report.md。保存输入哈希、响应路径/哈希、采样参数、评分代码哈希及环境版本。固定对比集外的8张图片未纳入本次比较；原始全部138张的流程摘要不能与本报告直接混比。
