# Code Benchmarks（无 Docker）

本目录跑五套**本机直接执行**的 coding / data-analysis benchmark，用来给 agentica 出第一批数字。

公榜上最有公信力的 TB2.1 / SWE-bench 都绑 Docker。这里刻意避开它们：判分在本机跑，PK 公信力会打折，但一天内能出分。**Aider Polyglot** 测改代码直到 pytest 绿；**InfiAgent-DABench** 测读 CSV 做数据分析直到打出 `@name[value]`。

| Benchmark | 测什么 | 默认用法 | 本机依赖 |
|---|---|---|---|
| **polyglot**（主，coding） | agent 改文件直到 pytest 绿 | `--bench polyglot` | git + pytest |
| **dabench**（data analysis） | agent 读 CSV，封闭题 `@tag[value]` | `--bench dabench` | pandas / numpy / scipy / sklearn |
| **livecodebench**（副） | 单轮算法生成 + 执行 | `--bench livecodebench` | 仅 python |
| **bigcodebench** | 函数级 + 真实库 | `--bench bigcodebench` | `pip install datasets` + 题面里的第三方库 |
| **evalplus** | HumanEval+，只当管道 smoke | `--bench evalplus` | 仅 python |

官方 Aider runner 其实也建议 Docker；这里**不用 aider**，只克隆 [polyglot-benchmark](https://github.com/Aider-AI/polyglot-benchmark) 的 Python 练习，用 agentica `DeepAgent`（`read_file` / `apply_patch` / `write_file` / `execute`，无 todo / `glob` / `grep`）改代码，再用 pytest 判分。默认 `--wire-api responses`。

## 快速开始

```bash
# 先验证判分管道（不调 LLM）
python evaluation/code_benchmark/run.py --dry-run --bench all

# Polyglot Python 子集，1 题冒烟
python evaluation/code_benchmark/run.py --bench polyglot --max-samples 1 --model gpt-4o-mini

# LiveCodeBench 裸模基线
python evaluation/code_benchmark/run.py --bench livecodebench --max-samples 5 --lcb-start-date 2024-08-01

# DABench（data analysis）。agentica 默认 Responses；评测 agent 与 Polyglot 同一套（无 todo / ls / glob / grep）
python evaluation/code_benchmark/run.py --bench dabench --max-samples 0 --agent agentica --wire-api responses --model deepseek-v4-flash-official --extra-body '{"reasoning": {"effort": "high"}}'
python evaluation/code_benchmark/run.py --bench dabench --max-samples 0 --agent codex --agent-timeout 600 --model deepseek-v4-flash-official --extra-body '{"reasoning": {"effort": "high"}}'
```

OpenAI 兼容端点用 `--base-url` / `--api-key`，或环境变量 `OPENAI_BASE_URL` / `OPENAI_API_KEY`。

## 推荐组合

1. **Polyglot 主分数（coding）**：`--language python --tries 2`。
2. **DABench（data analysis）**：`--bench dabench`。公开的是 DAEval validation（本仓库缓存 **257** 题 / 55 个 CSV）；判分是 `@name[value]` 精确匹配（浮点 1e-6），与 [InfiAgent](https://github.com/InfiAgent/InfiAgent) 的 `eval_closed_form.py` 同口径。agentica 与 Polyglot 共用评测 agent（无 todo，schema 去掉 `ls` / `glob` / `grep`），prompt 点名 CSV、禁止多余清洗、打出 `@tag` 立刻停。全量 `--max-samples 0`，默认 `--wire-api responses`。
3. **LiveCodeBench 裸分**：同一底模、单轮生成。`Polyglot − LCB ≈ harness 增益`。
4. **EvalPlus**：只确认管道没坏。头部模型已经 90%+，别拿出去 PK。
5. **BigCodeBench**：要 `datasets`，且题面会 `import` 真实第三方库；缺库的题会判失败。

Python 子集大约 34 题（全量 Polyglot 是 225，含 JS/Go/Rust/Java/C++）。先跑 `--max-samples 5` 估成本，再拉满。

已发布的 Agentica vs Codex CLI 对照见 [`docs/guides/benchmark.md`](../../docs/guides/benchmark.md)，原始 `summary.json` / `predictions.jsonl` 在 [`results/`](results/)。

## 单测

harness 自己的不调 LLM 的断言：

```bash
python -m pytest evaluation/code_benchmark/tests/ -q
```

覆盖评测 agent 的工具白名单、DABench `@tag[value]` 判分、prompt 约束、以及 `--dry-run` 的 gold/broken/empty 三条路径。

## 输出

`evaluation/code_benchmark/outputs/<time>-<bench>/`

- `predictions.jsonl` — 每题通过与否、耗时、模型输出、**metrics**
- `summary.json` — `passed / total / accuracy` + 与 TB2.1/Pro 并列的指标表

### 指标（`summary.json` → `metrics`）

必报（和准确率同一张表）：

| 字段 | 含义 |
|---|---|
| `avg_wall_clock_s` | 每任务端到端墙钟 |
| `avg_tool_calls` / `avg_api_calls` | 平均工具步数 / API 次数 |
| `crash_timeout_rate` | crash、判分超时、runner abort（max_turns 等）占比 |
| `completion_honesty_fail_rate` | 嘴上说 tests pass / done，pytest 实际未过 |

强烈建议：

| 字段 | 含义 |
|---|---|
| `avg_collateral_files` / `avg_collateral_lines` | 改了任务无关文件/行（`git diff --numstat`，solution 文件不算） |
| `error_recovery_rate` | 先红后绿 / 出错任务数 |
| `human_intervention_rate` | 调用了 `ask_user_question` 等需人介入的工具 |
| `cache_hit_rate` | `cached_input / (fresh + cached + write)` |

用量：`model`、`sum_api_calls` / `sum_tool_calls`、`sum_input_tokens` / `sum_fresh_input_tokens` / `sum_cached_input_tokens` / `sum_cache_write_tokens` / `sum_output_tokens` / `sum_wall_clock_s` / `sum_cost_usd`。`summary.json` 的 `tasks[]` 以及终端 `--- per task ---` 表按题列出 `tool_calls` 和 `api_calls`（两列分开，不混在一起）。缺测的格子是 `-` / `n/a`，不当 0。

Polyglot 评测关掉了 `ask_user_question`，所以 `human_intervention_rate` 应为 0；非 0 说明有人介入工具漏进了 agent。

## 和 Claude Code / Codex CLI 对比

不要从它们的交互式 TUI 里抠内部指标。正确做法是：**同一套题目 + 同一套 pytest 判分**，把对方当成 headless 子进程包进来。墙钟、对错、crash/timeout、honesty、collateral 是我们在外面量的，不依赖它们开没开 telemetry。

```bash
# 同一题、同一模型：agentica 默认 Responses；Codex CLI 只走 Responses
python evaluation/code_benchmark/run.py --bench polyglot --max-samples 5 --agent agentica \
  --model deepseek-v4-flash-official --extra-body '{"reasoning": {"effort": "high"}}'
python evaluation/code_benchmark/run.py --bench polyglot --max-samples 5 --agent codex \
  --model deepseek-v4-flash-official --extra-body '{"reasoning": {"effort": "none"}}'
```

`--agent codex` 且给了 `--base-url` 时，会在输出目录写一份隔离的 `CODEX_HOME`（`wire_api = "responses"`，`requires_openai_auth = false`），不读也不写 `~/.codex`。Codex CLI 从 2026-02 起不再支持 Chat Completions，所以该代理必须有 `/v1/responses`。

| 指标 | 怎么拿到 | 能否 PK |
|---|---|---|
| 对错 / accuracy | 我们的 pytest | 必报 |
| wall-clock / task | 我们的计时器包着子进程 | 必报 |
| crash / timeout | CLI 退出码、我们的 `--agent-timeout`、JSON `is_error` | 必报 |
| API calls | Claude：`num_turns`；Codex：`turn.completed` 条数 | 必报（口径是「模型回合」，不是 HTTP 次数） |
| cost / tokens / cache | Claude result JSON 的 `usage` / `total_cost_usd`；Codex `turn.completed.usage` | 有 JSON 就能报；订阅账号可能是估算 |
| tool calls | Codex jsonl 的 `command_execution` / `file_change` 等；Claude `--output-format json` **没有**工具次数 | Claude 这一格是 `-`，不要拿 0 去比 |
| honesty / collateral / recovery | 看它们的最终文本 + 工作区 `git diff` + 我们的 pytest 回灌 | 可 PK（不是它们内部的数） |

Claude：`claude --print --output-format json --dangerously-skip-permissions`。Codex：`codex exec --json --sandbox workspace-write -c approval_policy=never`（`exec` 没有 `--ask-for-approval`）。带 `--base-url` 的路径不读 `~/.codex`、也不走 ChatGPT 登录。这是产品对产品的 harness 对比，不是同一套 SDK 的 ablation。

评测过程把 `AGENTICA_HOME` 指到输出目录，不会写 `~/.agentica`。

## 和官方 runner 的差异

- **Polyglot**：题目与 pytest 判分对齐 aider；agent 是 agentica，不是 aider 的 edit-format。分数可对表，但不能直接贴官方 leaderboard。
- **DABench**：题目、CSV、`@tag[value]` 金标对齐 InfiAgent DAEval validation；agent 是 agentica / Codex CLI，不是官方 Docker sandbox。分数可对表，但不能直接贴官方 leaderboard。
- **LCB / EvalPlus / BCB**：本机 subprocess 执行，不是官方 Docker / remote evaluator。`--dry-run` 用 canonical / 坏答案自检管道。LiveCodeBench 官方 jsonl 约 1.2GB，runner **按行流式读取**，`--max-samples N` 不会整包下载。
