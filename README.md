# Collage Qwen3.5：A100 部署、推理与评测

在一台 Linux A100 机器上复现 Qwen3.5-4B/9B 的拼贴图片分析：图片 → `structure.json` → 校验、必要时一次模型修复 → `analysis.json` 和 bbox 预览。

仓库自带 `collage-recreate` 的 analyze 代码快照，不需要原电脑、不需要 SSH 回连，也不依赖另一个私有仓库。代码、模型、环境和输出均放在当前项目目录。当前分支已支持 LoRA SFT 和 vLLM 评测，见下方快速启动。

## LoRA 训练与 vLLM 评测快速启动

本节对应 `feat/vllm-eval-ddp-fa2` 分支的 `training/` 入口，目标是图片 → `structure.json`。评测使用 SWIFT 内部的 vLLM 引擎，无需另外启动 API 服务。

### 环境与路径

Linux x86_64、Python 3.12、glibc ≥ 2.31。当前版本：Torch 2.10.0+cu128、SWIFT 4.1.0、Transformers 5.5.3、vLLM 0.19.1；保留 flash-attn 2.8.3.post2、flash-linear-attention 0.4.1、causal-conv1d 1.7.0。依赖以 [training/requirements.txt](training/requirements.txt) 为准。已有环境无需重复安装，不要在训练运行中升级依赖。

在仓库根目录执行，替换实际路径：

```bash
git switch feat/vllm-eval-ddp-fa2
git pull --ff-only origin feat/vllm-eval-ddp-fa2
conda activate collage-qwen35-train

export MODEL_DIR=/path/to/Qwen3.5-4B
export DATA_DIR=/path/to/collage-structure-sft-v1
```

模型目录必须包含完整原始权重；数据目录包含 train.jsonl、validation.jsonl、test.jsonl 和图片。使用完整下载的本地数据，并保持数据版本一致。GPU 编号应替换为获准使用的空闲卡。

### 裸模型评测：官方非 thinking 通用配置

```bash
# 先跑 test 第一张；不会自动选择此前失败的图片
GPU=4 LIMIT=1 bash training/run_preview_official.sh

# 完整137张，显式保存基线目录
GPU=4 OUTPUT_DIR=runs/base-official-test137 bash training/run_preview_official.sh
```

[启动脚本](training/run_preview_official.sh) 默认：GPU 6、test 最多137条、每批8张、图片预算1024、输入上限12288、输出上限8192、vLLM 总上下文20480、显存预算为整卡的0.7。输出目录必须不存在；省略 `OUTPUT_DIR` 时自动生成时间戳目录。模型提前结束时不会强制生成满8192 token。

脚本使用以下六个参数：

```bash
--temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --presence-penalty 1.5 --repetition-penalty 1.0
```

出处：[Qwen3.5-4B 官方 Best Practices](https://huggingface.co/Qwen/Qwen3.5-4B#best-practices)。这是 **非 thinking 通用任务**的参考配置，不保证本任务准确率更高。官方说明 presence penalty 可以减少无限重复，但较大值可能造成语言混杂或表现下降。

图片识别、信息抽取和结构 JSON 输出，先用 **general** 配置作为实验起点；**reasoning** 配置主要作为数学、复杂逻辑、多步推导任务的参考。两者是采样建议，不是模型能力开关。这里始终设置 `enable_thinking=false`；选择 general 不代表关闭空间理解。

启动日志 `[sampling]` 显示最终生效的六个参数。普通 `run_lora.py` 不传采样参数时，仍是 temperature=0，其余未指定值沿用后端/模型默认值，不会自动套用上述组合。

```bash
# 降低 presence penalty 做对照；自动生成新目录
GPU=4 bash training/run_preview_official.sh --presence-penalty 0.5

# 增加批处理吞吐
GPU=4 INFER_BATCH_SIZE=8 bash training/run_preview_official.sh
```

`temperature`、`repetition-penalty`、`top-p`、`top-k` 支持两个评测后端；`presence-penalty`、`min-p` 当前仅支持本仓库的 vLLM 入口。SWIFT 4.1.0 CLI 没有直接暴露后两项，本仓库将它们应用到请求的 vLLM SamplingParams。所有采样参数只影响推理，不改变 SFT loss 或权重；训练命令误传会报错。详见 [推理采样参数.txt](training/推理采样参数.txt)。

### 双卡 LoRA 训练

先等占用6、7号卡的评测结束，再运行：

```bash
python training/run_lora.py \
  --model "$MODEL_DIR" --data "$DATA_DIR" --gpus 6,7 \
  --epochs 2 --batch-size 2 --grad-accum 4 \
  --learning-rate 5e-5 --max-length 12288 --image-tokens 1024 \
  --attn-impl flash_attention_2 \
  --save-steps 100 --eval-steps 100 --master-port 29501 \
  --output runs/train-2gpu-bs16-2epochs
```

有效 batch=2卡×每卡2条×累积4步=16。脚本使用 BF16、LoRA rank16/alpha32、冻结视觉编码器和对齐层、梯度检查点、5% warmup；SWIFT 4.1.0 默认 cosine 学习率衰减。5272条训练数据每轮约330次更新，2轮约660次，实际以编码后保留数量和日志为准。显存不足时改为 `--batch-size 1 --grad-accum 8`，有效 batch 不变。

训练只读 train/validation，test 不参与训练或 checkpoint 选择。2轮是首轮实验起点，用验证 loss 和验证图片生成结果选择 checkpoint。训练中的 max-length 包含输入、图片和标准答案；推理中该参数限制输入，输出由 max-new-tokens 单独限制。

续训在原命令中增加 `--resume /实际路径/checkpoint-300`，并使用新的 output 目录。保持原数据和配置；`--epochs 2` 表示累计完成2轮，不是额外2轮。checkpoint 必须保存完整训练状态、含 trainer_state.json；不能使用合并后的模型目录。

### LoRA 评测与基线对比

选择已保存完毕的 checkpoint，在空闲卡运行：

```bash
GPU=7 ADAPTER_DIR=/实际路径/checkpoint-600 \
  BASELINE_DIR=runs/base-official-test137 \
  OUTPUT_DIR=runs/step600-official-test137 \
  bash training/run_preview_official.sh
```

脚本默认让 vLLM 直接挂载 LoRA，不合并或生成完整模型副本。确实需要合并时，才在 shell 命令末尾显式追加 `--merge-lora`。BASELINE_DIR 可省略。对比应保持数据版本、prompt、图片预算、输出上限、采样参数和后端一致。

固定137张 test 用作最终回归；调参和 checkpoint 选择用 validation。可在脚本末尾追加 `--preview-split validation --limit 32`，基线也使用相同验证图片。

产物包括 preview.html、preview-assets/、predictions.jsonl、run.log、command.json。预览尽量画每个可用 bbox：重复 ID 或关联字段问题不阻止其他框显示。原始输出保留；结构通过率不等于定位准确率。每批完成后更新预览。

预览最左列现在显示 **GT 标注框线图**，右侧为基线（可选）和当前模型；顶部及逐图下方展示归档口径的六项指标。自动保存 `ground_truth.jsonl`，并导出 `metrics/summary.csv` 与 `metrics/per_sample.csv` 供汇报使用。GT 不进入模型输入；未运行复检时复检后通过率为 N/A。完整口径、旧结果补建及文件说明见 [GT 对照与六项评测](training/PREVIEW_METRICS.md)。

只重建页面，不重新推理、不需要 GPU：

```bash
python -u training/preview_report.py --run runs/step600-official-test137 \
  --baseline runs/base-official-test137
```

旧运行找不到 GT 时追加 `--gt /path/to/collage-structure-sft-v1/test.jsonl`；仍然只读取已有预测，不重新推理。新运行及能从 `command.json` 找回原数据的旧运行无需追加。

会逐张打印读取、画框进度。预览图最长边限制为1600像素，不改变模型输入或预测坐标。命令参数已做 CPU 回归测试，启动脚本通过 Bash 语法检查；完整 A100 推理及采样效果需在目标机器验证。

## 历史独立 vLLM 服务部署流程

以下 `.venv` 流程使用另一套依赖。训练与共享评测环境请按上方快速启动操作，不要混用两套安装步骤。

## 一、获取与安装

需要 Linux、现有 NVIDIA 驱动、可用 A100、`git`、`curl`、支持 venv 的 `python3`，建议预留至少 60 GiB。网络需能访问 Python 包源和模型下载站；代码可从 GitHub 下载 ZIP。脚本不会安装或修改系统 NVIDIA 驱动/CUDA。

```bash
git clone https://github.com/littlesun0727/collage-qwen35-distill.git
cd collage-qwen35-distill
nvidia-smi
bash scripts/setup.sh
```

安装器在项目内下载 Python 3.12 并创建 `.venv`，首次需要联网。若包源较慢：

```bash
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple bash scripts/setup.sh
```

核心复现版本：vLLM 0.17.1、PyTorch 2.10.0、Transformers 4.57.6、huggingface-hub 0.36.2。原测试机的完整包列表见 [tested-environment.txt](docs/tested-environment.txt)，用作参考；本机实际安装清单自动保存在 `logs/environment.txt`、`logs/pip-packages.json`、`logs/install-commands.log`。安装依赖解析可能因平台和包源变化而不同。

## 二、下载模型（在 A100 机器上）

下载器默认固定到原基线使用的 revision，支持断点续传，完成后校验每个文件；权重不在 GitHub 仓库里。

```bash
.venv/bin/python scripts/download_curl.py --model Qwen/Qwen3.5-9B
# 或下载 4B
.venv/bin/python scripts/download_curl.py --model Qwen/Qwen3.5-4B
```

直连 Hugging Face 较慢可加 `--endpoint https://hf-mirror.com`。也可用 `scripts/download_model.py`（HF_ENDPOINT 环境变量控制镜像）。下载模型需遵守对应模型仓库的使用条款。

## 三、选择 GPU 和启动服务

默认选择**物理 GPU 0**，如果要用第八张卡改为 `--gpu 7`。仅单卡，无张量并行。

```bash
.venv/bin/python scripts/service.py configure --model Qwen3.5-9B --gpu 0
.venv/bin/python scripts/service.py ensure
.venv/bin/python scripts/service.py status
tail -f logs/server.log
```

`ensure` 返回 `starting` 时正在加载；`status` 返回 `ready` 才可用。服务只监听 `127.0.0.1:8000`，不需要外部机器连接。

切换模型必须先 stop；不要同时为两种模型运行批处理：

```bash
.venv/bin/python scripts/service.py stop
.venv/bin/python scripts/service.py configure --model Qwen3.5-4B --gpu 0
.venv/bin/python scripts/service.py ensure
```

## 四、先做烟测，再跑图片目录

仓库不包含私人 covers 图片。可以先生成一张原创合成图检查链路，再将自己的图片放到 `data/covers/`。若要复现原 138 张逐图结果，需要自行拷贝那批图片；合成图只用于检查程序能否运行。

```bash
.venv/bin/python scripts/make_smoke_image.py
.venv/bin/python scripts/run_covers.py --model Qwen3.5-4B --input data/smoke --output runs/smoke-4b --workers 1
.venv/bin/python scripts/run_covers.py --model Qwen3.5-4B --input data/covers --output runs/covers-4b --workers 4
```

9B 使用同一入口，先切换服务，再把 `--model` 改为 `Qwen3.5-9B`，输出目录改为 `runs/covers-9b`。输入目录非递归扫描支持的图片；可用 `--limit 3` 先试少量。

复现配置：BF16，单卡，max-model-len=32768，vLLM max-num-seqs=8，本地4路请求，thinking=false，temperature=0，max_tokens=16000，json_object，normalized_1000。运行保存模型 revision、prompt、源码与输入文件哈希。A100 80G 可沿用 gpu-memory-utilization=0.90；它会预留 KV cache，因此显存显示较高不表示权重实际占满。

如 GPU 进程被 kill，下一次请求会检查并按需启动/恢复服务，最多重试两次传输失败。没有常驻监督进程、定时任务或空闲自动重启。模型输出不合法采用 analyze 工作流的修复规则，和传输重试分别记录。若整台机器或批处理进程也被杀，需手动重新运行批处理命令。

## 五、结果与复跑

`runs/covers-4b/index.html` 是完成后的原图/bbox 预览。每张图目录含 `structure.json`（模型坐标）、`analysis.json`（像素坐标）、`analysis.bbox.png`、`workflow/vlm/`（请求、原始响应和修复记录）。失败图保留日志，可能没有最终结构。

首次出现 `batch-summary.json` 后可在第二个终端生成实时页面：

```bash
.venv/bin/python scripts/live_report.py runs/covers-4b --watch
# 浏览器可访问本机时，使用本机静态服务；远端网页访问由你自己的端口转发决定
.venv/bin/python -m http.server 8080 --bind 127.0.0.1 --directory runs
```

相同命令可恢复未完成样本；已有 case-summary（包括失败）不重复调用模型。需要重新试失败图时另选目录：

```bash
.venv/bin/python scripts/run_covers.py --model Qwen3.5-4B --input data/covers --output runs/covers-4b-retry --workers 4 --retry-failures-from runs/covers-4b
.venv/bin/python scripts/audit_prompts.py runs/covers-4b
.venv/bin/python scripts/compare_covers.py runs/covers-9b runs/covers-4b
```

`comparison.json` 对比通过率、输出数量、坐标整十比例和归一化→像素换算；这些不是定位准确率。temperature=0 也不保证跨 GPU、批次调度和软件环境逐 token 一致。

## 六、离线评测与标注

五项主指标 v2.1：输出可用性、元素识别、定位、处理方式代理、关系/层级。完整定义见 [eval/README.md](eval/README.md) 和 [METRICS_V2.md](eval/METRICS_V2.md)。

先运行不需要 GPU 的合成数据演示：

```bash
.venv/bin/python -m eval.score --gt eval/examples/gt.jsonl --pred eval/examples/predictions.jsonl --occlusions eval/examples/occlusions.jsonl --out runs/eval-example
```

真实批次需要自己的人工核验 GT。默认 GT 的 `id` 为输入文件名（带扩展名）；否则提供 JSON 映射，例如 `{"cover.jpg":"0001"}`。首次回复与修复后结果必须分开报告：

```bash
.venv/bin/python -m scripts.evaluate_batch --batch runs/covers-4b --gt data/gt.jsonl --stage raw_first --out runs/eval-4b-first
.venv/bin/python -m scripts.evaluate_batch --batch runs/covers-4b --gt data/gt.jsonl --stage final_accepted --out runs/eval-4b-final
# 如有审核过的遮挡对，增加 --occlusions data/occlusions.jsonl
# 如 GT 使用编号 ID，增加 --id-map data/id-map.json
```

失败样本不从评测分母删除。未经核验的教师结构只能测教师一致性，不能当作真实准确率。GT/测试集不得混入后续训练集。

现有标注工具也已包含：`.venv/bin/python -m annotation.server --dataset data/annotations --port 8765`，数据格式与功能见 [annotation/README.md](annotation/README.md)。

## 七、验证与停止

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/service.py stop
.venv/bin/python scripts/service.py status
```

停止仅作用于本项目记录的进程组；不会清理其他用户 GPU 任务。模型、环境和输出保留在磁盘。

## 已有基线与代码范围

| 同一新 prompt，138张 | 9B | 4B |
|---|---:|---:|
| 首次通过 |126|111|
| 修复后通过 |9|9|
| 最终失败 |3|18|
| 批次耗时（秒） |1204.66|1718.54|
| 输出 token |255536|590692|

这些是此前机器上经 SSH 调用的历史值；本仓库改为同机 HTTP，硬件与调度不同，耗时不保证相同。失败常见于重复生成、输出触顶和关联/层级约束。没有逐图人工 GT 时，不应把结构通过率当作识别准确率。

`vendor/collage-recreate/` 保留原 analyze 及其依赖的源文件，哈希清单见 `docs/upstream-snapshot.json`。其中后续渲染、素材处理、其他模型 provider 模块仅作为源码依赖快照；本仓库支持的入口是本文列出的 analyze、评测、标注流程。私人 SSH 工具、历史图片报告、原始数据不在公开包内。 训练和评测入口见 `training/`。

## 串行评测多个 base / LoRA checkpoint

单次脚本和队列默认 infer-batch-size=8。复制清单后填写本机路径，任务按 `jobs` 顺序执行，前一个模型进程退出后才启动下一个。

```bash
cp training/eval_queue.example.json eval_queue.local.json
# 先编辑 eval_queue.local.json 中的 model/data/adapter 路径
python -u training/run_eval_queue.py --config eval_queue.local.json --gpu 4 --dry-run
python -u training/run_eval_queue.py --config eval_queue.local.json --gpu 4 --output-root runs/eval-checkpoints
```

清单见 [eval_queue.example.json](training/eval_queue.example.json)。没有 `adapter` 的任务跑裸模型；有 `adapter` 的任务默认直接挂载 LoRA。仅在明确需要时，给该任务添加 `"merge_lora": true` 才合并。每个任务可覆盖 `model`，必须与 adapter 的基础模型对应。`compare_to` 填前面任务的名字，例如 `base`，即可生成并排对比。

JSON 顶层的 `sampling` 设置全局采样参数，每个任务也可添加 `sampling` 覆盖其中部分参数。例如全局 `"sampling": {"temperature": 0.2, "presence_penalty": 0}`；某个任务设置 `"sampling": {"temperature": 0.7, "presence_penalty": 1.5}` 可保持旧配置。支持 `temperature`、`top_p`、`top_k`、`min_p`、`presence_penalty`、`repetition_penalty`，未设置的项保持原官方默认值。`--dry-run` 和 `plan.json` 展示合并后的参数。参数变化会触发该任务归档重跑；保留不同采样实验时使用不同任务名。

每个任务保存到 `output-root/任务名/`，队列状态见 `queue-status.json`。重复使用同一个 output-root 时，按任务名自动核验并复用完整且配置一致的预测，只重建预览；新增任务继续推理。不完整或配置不同的旧结果先归档到 `_history/任务名/时间戳/`，再重新跑该任务。已有独立 base 结果可整体复制到 `output-root/base/`，保留原有输入、配置及预测文件，且图片路径仍需可访问。`--dry-run --output-root ...` 可提前查看复用判断。校验不包含权重和图片内容哈希，不要原地替换同路径文件。默认失败即停止；加 `--continue-on-error` 可继续独立任务，基线失败时跳过依赖它的任务。仅使用已完整保存的 checkpoint，不会等待训练生成未来的 checkpoint。

完整说明见 [串行评测说明](training/串行评测.txt)。这只是评测排队，不改变训练配置。
