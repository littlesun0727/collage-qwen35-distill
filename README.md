# Collage Qwen3.5：A100 部署、推理与评测

在一台 Linux A100 机器上复现 Qwen3.5-4B/9B 的拼贴图片分析：图片 → `structure.json` → 校验、必要时一次模型修复 → `analysis.json` 和 bbox 预览。

仓库自带 `collage-recreate` 的 analyze 代码快照，不需要原电脑、不需要 SSH 回连，也不依赖另一个私有仓库。代码、模型、环境和输出均放在当前项目目录。当前**没有实现微调/蒸馏训练**。

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

`vendor/collage-recreate/` 保留原 analyze 及其依赖的源文件，哈希清单见 `docs/upstream-snapshot.json`。其中后续渲染、素材处理、其他模型 provider 模块仅作为源码依赖快照；本仓库支持的入口是本文列出的 analyze、评测、标注流程。私人 SSH 工具、历史图片报告、原始数据和训练代码不在公开包内。
