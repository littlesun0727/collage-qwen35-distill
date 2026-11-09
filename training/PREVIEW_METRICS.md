# GT 对照与六项评测

`run_preview_official.sh`、`run_lora.py --preview/--adapter` 和评测队列继续使用原命令。每次评测自动保存所选样本的 `ground_truth.jsonl`，逐批更新 `preview.html`：每行最左为 GT 框线图，右侧为基线（若指定）和当前预测；原图在 GT 下方的“查看原图”链接中。

GT 从所选数据 split 的 assistant 标注读取，在剥离答案前单独保存；传给模型的 `input.jsonl` 始终不含答案。`--images` 的自选图片没有配套 GT 时显示明确提示，不拿训练集答案冒充。

## 已有 137 张预测：只重建报告

在工程根目录执行，无需 GPU，也不重新推理：

```bash
python -u training/preview_report.py \
  --run runs/step600-official-test137 \
  --baseline runs/base-official-test137
```

GT 优先读取运行目录的快照。旧目录无快照时，按 `command.json` 中的 `source.local_dir` 和 `split` 读取原数据，只选择 `input.jsonl` 中的评测图片。数据目录已搬家或旧命令缺少来源时，显式指定同一批标注：

```bash
python -u training/preview_report.py \
  --run runs/step600-official-test137 \
  --baseline runs/base-official-test137 \
  --gt /path/to/collage-structure-sft-v1/test.jsonl
```

`--baseline` 可省略；`--gt` 支持数据集的 messages/images JSONL，或本功能保存的 ground_truth.jsonl。以完整图片路径对齐，遇到缺失或重复 GT 会报错，不按同名文件猜配。运行目录和数据整体迁移时，也需使 predictions/input 中的图片路径指向有效文件。

报告及 CSV 仅依赖 Pillow、NumPy、SciPy；训练环境依赖已列入 `training/requirements.txt`，现有环境只需补齐缺失包，无需升级训练组件。

## 六项口径

参考 [2026-11-03 归档六项评测 v3](https://github.com/littlesun0727/reports-achrive/blob/main/reports/2026-11-03/collage-demo/collage-qwen35-distill/runs/eval/covers137-six-metrics-auto-occlusions-20261102/index.html)。原有 `eval.score` 的 v2 接口保持兼容。

| 项目 | 当前报告 |
| --- | --- |
| JSON 结构通过率 | 首次原始输出的核心结构合法率；没有复检流程时，复检后为 N/A |
| photo | 召回率、已找到框平均 IoU |
| overlay/text/frame | 合为 fixed 组，报告召回率、已找到框平均 IoU，组内不区分类别 |
| 重叠元素顺序 | 固定 GT 对的顺序满足率，未找到端点、反序、排序缺失均计失败 |
| frame | 找到对应 frame 且含有效照片引用为“连上”；对应照片集合不漏、不多、不错为“全部连对” |
| linked_to | 出框主体及其来源都匹配正确 |

匹配为组内一对一 IoU ≥ 0.50，先最大化匹配数量，再最大化 IoU。主分先逐图计算再平均；GT 有该组/关系的固定图片进入分母，缺失和坏 JSON 不剔除。条件 IoU 只统计有匹配的图片，展示适用数；空组/无适用关系为 N/A。多报框、额外关系、反序等保留在诊断中，不合成总分。

保存 GT 时，已有 `occlusion_pairs` 原样使用（显式 null 表示未提供）；否则按 GT bbox 正面积重叠与 GT layer_order 一次生成，写入快照。预测不参与生成这些关系；这是一种自动近似，透明元素和空心相框可能不代表真实遮挡。

当前训练推理只生成一次输出，因此第 2–6 项读取首次原始输出，仍可计入结构非法回复里的有效元素。画框时移除空 think/代码围栏、裁剪越界坐标等仅为显示，计分不使用这些调整后的结果。归档主表的第 2–6 项读取复检后的最终结果；不能只因同为 137 张就直接混比。GT 内容、图片集合、预测阶段和推理配置均应一致，未经人工审核的教师 GT 得分表示一致性。

## 输出与汇报

| 文件 | 用途 |
| --- | --- |
| `preview.html`、`preview-assets/` | 可离线打开的总表与逐图对照；点击图片查看大图 |
| `ground_truth.jsonl` | 所选评测集的 GT、图片像素指纹和固定顺序关系快照 |
| `metrics/summary.csv` | 汇总指标，含适用图片数和 GT 数，可直接用 Excel 打开 |
| `metrics/per_sample.csv` | 逐图逐模型分数，便于筛选失败例 |
| `metrics/summary.json` | 协议、GT 指纹、来源及完整汇总 |
| `metrics/per_sample.jsonl` | 匹配框、关系成败和错误诊断 |
| `metrics/gt.jsonl` | 本次报告实际使用的 GT |

推理中页面会刷新，并标注为临时分数，未完成样本暂按缺失计入固定分母。基线和当前模型总是按当前这批 GT 重算；基线缺图按缺失计分。汇报时保留整个运行目录中的 HTML、图片和 metrics 子目录。

验证：`python -m unittest discover -s training -p 'test_preview*.py' -v`，包括 137 张的完整报告、缺失预测、非法 JSON、基线重排、GT 指纹与原始输出不变等回归用例。
