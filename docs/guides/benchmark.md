# 评测

Agentica 与 Codex CLI 在同一套题目、同一套 pytest 上的对照。底模都是 `deepseek-v4-flash-official`，题集是 [Aider Polyglot](https://github.com/Aider-AI/polyglot-benchmark) 的 **Python 全量 34 题**（官方全榜 225 题含其它语言；本仓库 runner 只跑 Python 子集）。不对 Docker / SWE-bench 声称对齐。

判分在 harness 外面跑：对错、墙钟、crash、误报完成、误改文件都不读对方 TUI。原始机器输出仍在本地 `evaluation/code_benchmark/outputs/`（含 workdir，不入库）；GitHub 上只放 `summary.json` 和 `predictions.jsonl`。

## Agentica vs Codex

两边质量打平：全对、0 crash、0 误报、0 误改。差距在效率——同一模型下 Agentica 更快、更少工具、更少 token，更省钱。

![Agentica_pk_Codex](https://github.com/shibing624/agentica/blob/main/docs/assets/polyglot-agentica-vs-codex.png)

| 指标 | Agentica | Codex | 对比 |
|---|---|---|---|
| 准确率 | **34/34（100%）** | **34/34（100%）** | 平 |
| 平均墙钟 / 题 | **43.7s** | 68.6s | Agentica 快，Codex 慢 **1.57×** |
| 总墙钟 | **1486s** | 2332s | 25 min vs 39 min |
| tool calls | **139**（均 4.1） | 168（均 4.9） | Codex 多 **1.21×** |
| 输入 token | **1,236,109** | 2,848,928 | Codex 多 **2.30×** |
| 输出 token | **172,705** | 201,421 | Codex 多 **1.17×** |
| crash / timeout | **0/34** | **0/34** | 平 |
| 误报完成（声称 done 但 pytest 红） | **0/34** | **0/34** | 平 |
| 误改文件 | **0** | **0** | 平 |
| cache hit | 80.4% | 86.8% | 平 |

| 产品 | 跑次 | summary | predictions |
|---|---|---|---|
| **Agentica** | `20260819-195326-polyglot` | [summary.json](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/results/20260819-195326-polyglot/summary.json) | [predictions.jsonl](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/results/20260819-195326-polyglot/predictions.jsonl) |
| **Codex CLI** | `20260817-215956-polyglot` | [summary.json](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/results/20260817-215956-polyglot/summary.json) | [predictions.jsonl](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/results/20260817-215956-polyglot/predictions.jsonl) |

仓库内路径：`evaluation/code_benchmark/results/<run-id>/`。`summary.json` 的 `metrics` 与上表对应；`predictions.jsonl` 一行一题，含通过与否、墙钟、工具次数、token 和模型输出。

## 设定

- 模型：`deepseek-v4-flash-official`（思考开：Agentica `reasoning.effort=high`，Codex `model_reasoning_effort=high`）
- 题集：`--bench polyglot --language python --max-samples 0`（34 题）
- 裁判：题目目录里的 pytest（`--rootdir=. --noconftest`），两边同一条命令
- Agentica：`run.py --agent agentica`（默认 `--wire-api responses`；`DeepAgent` 文件读写 + `execute`，无 todo / `ls` / `glob` / `grep`）
- Codex：`--agent codex --agent-timeout 600`（`codex exec --json`，隔离 `CODEX_HOME`）
- 串行、同一兼容端点；评测把 `AGENTICA_HOME` 指到输出目录，不写 `~/.agentica`

复现（key / base-url 自己导出，不要写进命令行）：

```bash
python evaluation/code_benchmark/run.py \
  --bench polyglot --max-samples 0 --language python \
  --agent agentica --model deepseek-v4-flash-official \
  --extra-body '{"reasoning": {"effort": "high"}}'

python evaluation/code_benchmark/run.py \
  --bench polyglot --max-samples 0 --language python \
  --agent codex --agent-timeout 600 --model deepseek-v4-flash-official
```

评测入口说明见 [`evaluation/code_benchmark/README.md`](https://github.com/shibing624/agentica/blob/main/evaluation/code_benchmark/README.md)。

## Data analysis（InfiAgent-DABench）

[InfiAgent-DABench](https://github.com/InfiAgent/InfiAgent) 的 DAEval **validation** 封闭题：给 CSV，agent 用 pandas / sklearn 算出 `@name[value]`。判分是精确匹配（浮点 1e-6），与官方 `eval_closed_form.py` 同口径。这是 **data analysis** 类别，不是 Polyglot 那种改代码 + pytest。Agentica 走 `--wire-api responses`，评测 agent 与 Polyglot 同一套（无 todo / `ls` / `glob` / `grep`）；prompt 点名 CSV、禁止题面没要求的清洗、算出 `@tag` 立刻停。

下面是同一模型、同一前 10 题的探路（n=10，不是全量 257 题）。两边都挂在 id=8（按舱等 fare 的 population std 和金标差 ~0.2）；Codex 另外挂了 id=7（sklearn 线性回归）。全量以 `--max-samples 0` 为准。

| 指标 | Agentica | Codex CLI | 对比 |
|---|---|---|---|
| 准确率 | **9/10（90%）** | 8/10（80%） | Agentica |
| 平均墙钟 / 题 | **25.7s** | 26.5s | 平 |
| 总墙钟 | **257s** | 265s | 平 |
| crash / timeout | **0/10** | **0/10** | 平 |
| tool calls | **35**（均 3.5） | 40（均 4.0） | Agentica 略少 |
| 输入 token | **304,152** | 502,578 | Codex 多 **1.65×** |
| 输出 token | 24,343 | **11,840** | Codex 更少 |
| cache hit | 87.8% | 90.3% | 平 |

| 产品 | 跑次（本地 `outputs/`） |
|---|---|
| Agentica | `evaluation/code_benchmark/outputs/20260820-132753-dabench` |
| Codex CLI | `evaluation/code_benchmark/outputs/20260820-133213-dabench` |

```bash
python evaluation/code_benchmark/run.py \
  --bench dabench --max-samples 0 --agent agentica --wire-api responses \
  --model deepseek-v4-flash-official \
  --extra-body '{"reasoning": {"effort": "high"}}'

python evaluation/code_benchmark/run.py \
  --bench dabench --max-samples 0 --agent codex --agent-timeout 600 \
  --model deepseek-v4-flash-official \
  --extra-body '{"reasoning": {"effort": "high"}}'
```

