# 结构评测：运行说明

当前主协议为 [五项主指标 v2.1](METRICS_V2.md)，首次原始响应与修复流程后最终交付结果分列报告。旧版 [8项诊断指标](METRICS_V1.md) 保留兼容。评分器仅离线计算，不调用模型、不需要GPU。

主结果在 summary.json 的 primary 中，下文 per_template_macro 与旧版示例属于兼容诊断视图。私人图片、教师结果和运行报告不随公开仓库分发；请用自己的数据生成报告。

## 安装与最小演示

Python 3.10或更新版本，从仓库根目录运行：

```powershell
python -m pip install -r eval/requirements.txt
python -m eval.score --gt eval/examples/gt.jsonl --pred eval/examples/predictions.jsonl --occlusions eval/examples/occlusions.jsonl --out runs/eval/example-v1 --run-name synthetic-example
```

演示为3个人工构造的最小结构：完美预测、错分类/错连/反序、缺失预测。不是实际模型成绩。
预期结果：

| 指标 | 逐模板均值（主视图） | 全量累计（对照） |
| --- | --- | --- |
| 检测F1 | 66.67% | 80% |
| 分类准确率 | 90%，仅2/3张图有匹配 | 90% |
| 匹配mIoU | 100%，仅2/3张图有匹配 | 100% |
| 匹配覆盖率 | 66.67% | 66.67% |
| photo_id F1 | 33.33% | 40% |
| linked_to F1 | 66.67% | 80% |
| 遮挡对正确率 | 50% | 50% |

缺失预测仍进入检测和关系成绩；分类/mIoU属于条件分数，不为无匹配图片虚构数值。

| 输出文件 | 用途 |
| --- | --- |
| report.md | 两种汇总并列的指标表、分类混淆矩阵、分母与覆盖提示 |
| summary.json | 完整汇总、各类分项、协议和输入指纹 |
| per_sample.jsonl | 每图匹配、漏检、多检、错连、反序及合法性错误 |

JSON分数为0–1，null为N/A。
v1.1主视图使用 `summary.per_template_macro`，每个读数含mean、eligible_images、total_images。旧顶层指标仍保留全量累计口径。
每图先算F1再平均，不用平均P/R推导平均F1。

## 评价当前1000张图上的学生预测

现有delivery可直接读入，不移动或修改原始数据：

```powershell
python -m eval.score --gt data/teacher-delivery --pred runs/student-step-1000/predictions.jsonl --out runs/eval/student-step-1000 --model Qwen-checkpoint-path --run-name student-step-1000
```

这批GT是未核验教师标注，成绩代表与教师的一致性。没有人工遮挡对时，其余指标正常计算，层级为N/A。

补好遮挡审核文件后：

```powershell
python -m eval.score --gt data/teacher-delivery --pred runs/student-step-1000/predictions.jsonl --occlusions data/eval/occlusions.jsonl --require-occlusions --out runs/eval/student-step-1000-reviewed
```

全部图片完成审核后才使用 --require-occlusions；确认无关系的图填空列表。

## 输入格式

推荐JSONL，每行一个样本。外层id是固定样本ID，当前delivery使用test_id，如 `"0001"`：

```json
{"id":"0001","structure":{"background":{"type":"transparent"},"objects":[{"id":"photo_a","kind":"photo","bbox":[100,100,500,800]}],"layer_order":["photo_a"]}}
```

structure也可以是原始模型文本字符串；不要丢掉模型输出坏JSON的样本：

```json
{"id":"0002","structure":"模型输出了无法解析的文本"}
```

也可引用已有文件，路径相对JSONL：

```json
{"id":"0001","structure_path":"cases/0001/structure.json"}
```

structure与structure_path二选一。原始structure内coordinate_mode优先；缺省使用外层coordinate_mode，再缺省为normalized_1000。像素坐标记录提供画布尺寸：

```json
{"id":"0001","width":1200,"height":1800,"coordinate_mode":"pixel","structure_path":"cases/0001/structure.json"}
```

如果引用文件内已声明normalized_1000，外层pixel不会覆盖它。

其他支持的输入：

- delivery目录：读取manifest.json中的cases[*].test_id与structure.json路径，校验声明的SHA-256；manifest损坏时报错，不退回扫描。
- 无manifest目录：递归读取 `<case>/structure.json`，父目录名作为id。
- 没有structure.json的目录：读取顶层 `<id>.json`。
- 单个 `<id>.json` 或 `<case>/structure.json`。
- 当前delivery的唯一case名（如0001_db-curated）可映射到test_id（0001）。重复、歧义、多余预测id会报错；缺失预测仍按GT计分。

## 遮挡审核文件

单独保存occlusions.jsonl，不修改模型训练目标：

```json
{"id":"0001","occlusion_pairs":[["photo_a","sticker_b"]]}
{"id":"0002","occlusion_pairs":[]}
{"id":"0003","occlusion_pairs":null}
```

方向为“下方、上方”。第一条有已确认约束，第二条已审核无适用约束，第三条未审核。
也支持GT JSONL外层occlusion_pairs；与sidecar冲突时会报错。
不能把未审图片统一填[]，也不能从完整排序或bbox相交自动伪造人工约束。

## 参数、代码与测试

```powershell
python -m eval.score --help
python -m unittest discover -s tests -p test_eval.py -v
```

- --iou-threshold默认0.50，影响几何/结构匹配，不影响辅助软IoU-F1。改变门槛后需重评基线。
- --model、--run-name仅记录checkpoint和运行标识，不加载模型。
- matching.py：IoU与匹配；schema.py：核心合法性与坐标统一。
- metrics.py：计分与汇总；io.py：输入和固定ID对齐；score.py：CLI与报告。

程序接口：

```python
from eval.schema import parse_structure
from eval.metrics import evaluate_sample, aggregate

result = evaluate_sample("0001", parse_structure(gt_dict),
                         parse_structure(pred_dict), occlusion_pairs=None)
summary = aggregate([result])
```

本工具不生成教师标注、不运行训练，不将教师自检成绩当作学生能力。
