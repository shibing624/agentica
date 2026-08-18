# 评测

Agentica CLI 与 Codex CLI 在同一套题目、同一套 pytest 上的对照。底模都是 `deepseek-v4-flash-official`，题集是 [Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) 的 **Python 全量 34 题**（官方全榜 225 题含其它语言；本仓库 runner 只跑 Python 子集）。不对 Docker / SWE-bench 声称对齐。

判分在 harness 外面跑：对错、墙钟、crash、误报完成、误改文件都不读对方 TUI。原始机器输出仍在本地 `evaluation/code_benchmark/outputs/`（含 workdir，不入库）；GitHub 上只放 `summary.json` 和 `predictions.jsonl`。

## Agentica CLI vs Codex CLI

| 指标 | Agentica CLI | Codex CLI | 对比 |
|---|---|---|---|
| 准确率 | **34/34（100%）** | **34/34（100%）** | 平 |
| 平均墙钟 / 题 | **60.0s** | 68.6s | Agentica 快，Codex 慢 1.14× |
| 总墙钟 | **2039s** | 2332s | 34 min vs 39 min |
| crash / timeout | **0/34** | **0/34** | 平 |
| 误报完成（声称 done 但 pytest 红） | **0/34** | **0/34** | 平 |
| 误改文件 | **0** | **0** | 平 |
| tool calls | 247（均 7.3） | 168（均 4.9） | Codex 步数更少 |
| 输入 token | **2,595,107** | 2,848,928 | Codex 多 10% |
| 输出 token | 227,554 | 201,421 | Codex 少 11% |
| cache hit | 85.6% | 86.8% | 都约 86% |

两边质量打平：全对、0 crash、0 误报、0 误改。差距在速度和输入——同一裁判下 Agentica CLI 平均少等 8.6 秒/题，少吃约 25 万输入 token。Codex 工具步数更少，但墙钟没有因此更好。

| 产品 | 跑次 | summary | predictions |
|---|---|---|---|
| **Agentica CLI** | `20260817-223855-polyglot` | [summary.json](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/results/20260817-223855-polyglot/summary.json) | [predictions.jsonl](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/results/20260817-223855-polyglot/predictions.jsonl) |
| **Codex CLI** | `20260817-215956-polyglot` | [summary.json](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/results/20260817-215956-polyglot/summary.json) | [predictions.jsonl](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/results/20260817-215956-polyglot/predictions.jsonl) |

仓库内路径：`evaluation/code_benchmark/results/<run-id>/`。`summary.json` 的 `metrics` 与上表对应；`predictions.jsonl` 一行一题，含通过与否、墙钟、工具次数、token 和模型输出。

## 设定

- 模型：`deepseek-v4-flash-official`，`extra_body` 为空（Codex 隔离配置 `model_reasoning_effort = high`）
- 题集：`--bench polyglot --language python --max-samples 0`（34 题）
- 裁判：题目目录里的 pytest（`--rootdir=. --noconftest`），两边同一条命令
- Agentica：本仓库 `run.py --agent agentica`（`DeepAgent` + 文件工具 + `execute`）
- Codex：`--agent codex --agent-timeout 600`（`codex exec --json`，隔离 `CODEX_HOME`）
- 串行、同一 Venus 兼容端点；评测把 `AGENTICA_HOME` 指到输出目录，不写 `~/.agentica`

复现（key / base-url 自己导出，不要写进命令行）：

```bash
python evaluation/code_benchmark/run.py \
  --bench polyglot --max-samples 0 --language python \
  --agent agentica --model deepseek-v4-flash-official

python evaluation/code_benchmark/run.py \
  --bench polyglot --max-samples 0 --language python \
  --agent codex --agent-timeout 600 --model deepseek-v4-flash-official
```

评测入口说明见 [`evaluation/code_benchmark/README.md`](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/README.md)。
