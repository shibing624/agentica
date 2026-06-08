# VaG Evaluation 实验执行 TODO

本目录是 VaG 论文的实验复现区。核心代码和可复用 SDK 能力仍在 `agentica/`；这里只放论文实验 runner、adapter、数据、baseline、分析脚本和结果摄取逻辑。

当前主任务：把 VaG 论文实验做完，尤其是把现在的 deterministic semi-real 结果升级为真实 Agentica / Terminal-Bench 2.0 trace 支撑的结果。

## 0. 机器分工

| 机器 | 用途 | 原因 | 预期产物 |
|---|---|---|---|
| Mac 本机 | 写代码、生成 split、跑小型 synthetic smoke、汇总表格、写论文 | 不应假设能稳定构建全部 TB2 Linux Docker 镜像 | `evaluation/vag/data/*.jsonl`、`results/vag/*.json`、论文表格 |
| dev CPU Linux 机 | 跑 Harbor + Terminal-Bench 2.0 Docker 任务、oracle smoke、真实 TB2 agent replay | TB2 每个 task 多数是独立 Linux/Docker 环境，CPU 足够跑多数 terminal/build/debug 任务 | TB2 pass/fail logs、holdout replay JSONL |
| GPU 服务器 | 可选：跑本地 LLM judge / 本地 agent backend | TB2 本身不一定需要 GPU；只有本地模型推理或大模型服务需要 GPU | 真实 `AgentCritic` cross-model 结果、低成本本地 judge 结果 |

结论：**真实 TB2 计算执行放到 Linux/Docker/Daytona；Mac 只做实验编排和结果摄取。**

## 1. 当前本地可跑 smoke

### 1.1 Pilot Regression Injection

```bash
python evaluation/vag/runners/regression_injection.py
```

目的：快速验证 gate pipeline 是否能跑通。

预期结果：生成 `60` 条 pilot candidates：

- `evaluation/vag/data/pilot_candidates.jsonl`
- `evaluation/vag/data/pilot_labels.jsonl`
- `evaluation/vag/data/pilot_ablation_results.json`

预期现象：`vag_full` 的 harmful admission 明显低于单 critic。

### 1.2 200-sample semi-real 表格

```bash
python evaluation/vag/runners/regression_injection.py --set extended
```

目的：生成当前论文表 2 的 deterministic semi-real 结果。

预期产物：

- `evaluation/vag/data/candidates.jsonl`
- `evaluation/vag/data/labels.jsonl`
- `results/vag/hot_pollution_rate.json`

当前预期趋势：

- `ungated`: harmful admission = `1.0`
- `skillsvote_style`: harmful admission = `1.0`
- `exec_agent`: harmful admission 约 `0.125`
- `vag_full`: harmful admission = `0.0`
- benign retention 当前为 `1.0`，但真实 trace 后可能下降，需要专门分析 false rejection。

### 1.3 附加本地实验

```bash
python evaluation/vag/runners/case_studies.py
python evaluation/vag/runners/cross_model.py
python evaluation/vag/runners/downstream_stress.py
python evaluation/vag/runners/holdout_replay.py --synthetic
python evaluation/vag/analysis/make_paper_tables.py --out results/vag/paper_tables.json
```

目的：生成论文 case study、mock judge sensitivity、downstream stress 和 paper table JSON。

预期产物：

- `evaluation/vag/data/case_studies/provenance.jsonl`
- `evaluation/vag/data/case_studies/report.md`
- `evaluation/vag/data/cross_model_sensitivity.json`
- `results/vag/downstream_stress.json`
- `results/vag/holdout_replay_decisions.json`
- `results/vag/paper_tables.json`

## 2. 真实 Terminal-Bench 2.0 环境准备

TB2 官方运行入口是 Harbor。用户给出的两个来源内容一致：

- GitHub: `https://github.com/harbor-framework/terminal-bench-2`
- HuggingFace: `https://huggingface.co/datasets/harborframework/terminal-bench-2.0/tree/main`

这些 task 是一个目录一个任务，里面包含 Docker / test / 资源文件。不要把它们当成本地 Python 单测。

### 2.1 在 Linux dev CPU 机准备 Harbor 和 Docker

在 Linux dev 机上完成：

```bash
# 安装 Docker，并确认 daemon 正常
sudo docker ps

# 安装 Harbor，按官方文档执行
# 具体安装命令以 Harbor 文档为准
harbor --help
```

目的：确认 Linux 机能构建和运行 TB2 task image。

预期结果：

- `docker ps` 正常返回；
- `harbor --help` 能输出 CLI 帮助；
- 机器有足够磁盘空间缓存 TB2 镜像和依赖。

### 2.2 克隆 TB2 数据集仓库，可选但推荐

```bash
git clone https://github.com/harbor-framework/terminal-bench-2.git /data/terminal-bench-2
```

目的：本地检查 task 目录、Dockerfile、test 文件和大资源需求。

预期结果：

```text
/data/terminal-bench-2/
  build-pov-ray/
  fix-git/
  compile-compcert/
  ...
```

如果不 clone，也可以直接通过 Harbor 的 dataset id 跑：

```bash
terminal-bench/terminal-bench-2
```

## 3. 生成 TB2 split 和 Harbor 运行计划

在 Mac 或 Linux dev 机的本 repo 中执行：

```bash
python evaluation/vag/runners/tb2_harbor_plan.py \
  --dataset-root /data/terminal-bench-2 \
  --event-size 50 \
  --holdout-size 14 \
  --test-size 25 \
  --out results/vag/tb2_harbor_plan.json
```

如果暂时没有 clone TB2 repo，可以先不用 `--dataset-root`，脚本会用文档页任务快照生成 dry-run split：

```bash
python evaluation/vag/runners/tb2_harbor_plan.py \
  --event-size 50 \
  --holdout-size 14 \
  --test-size 25
```

目的：固定论文实验 split，避免后续结果不可复现。

预期产物：

- `evaluation/vag/data/tbench2_split_event.txt`
- `evaluation/vag/data/tbench2_split_holdout.txt`
- `evaluation/vag/data/tbench2_split_test.txt`
- `evaluation/vag/data/tbench2_split_manifest.json`
- `results/vag/tb2_harbor_plan.json`

## 4. 在 Linux/Docker/Daytona 上跑 TB2 oracle smoke

先跑官方 oracle，确认不是环境问题：

```bash
harbor run -d terminal-bench/terminal-bench-2 -a oracle
```

目的：验证 Harbor、Docker、TB2 dataset 访问、task image 构建、test 判定全部正常。

预期结果：

- Harbor 能成功启动多个 TB2 task；
- oracle agent pass rate 应接近官方参考表现；
- 如果 oracle 都失败，先修 Harbor / Docker / 数据集环境，不要继续跑 VaG。

## 5. 真实 VaG 实验总流程

真实 TB2 实验不是一次命令完成，而是四段：

```text
Event split   -> 生成真实 candidate skills
Holdout split -> without_skill / with_skill replay，得到 ExecCritic-replay 判定
Gate ablation -> 每个 gate 冻结一份 hot skill store
Test split    -> 用 frozen hot skills 跑 pass@1 / tokens / bad-skill failures
```

### 5.1 Event split：生成真实 candidates

在 TB2 Linux/Docker/Daytona 环境跑 Agentica / Claude-code style agent，让 agent 在 event split 上执行任务并记录：

- tool error
- tool recovery
- task success
- user correction / oracle correction
- final patch / command history

目的：从真实执行轨迹生成 candidate `SKILL.md`，替换当前 semi-real seeds。

预期产物：

```text
results/vag/tb2_event_traces/
  <task_id>/run.jsonl
  <task_id>/final_status.json
```

然后转换为：

```text
evaluation/vag/data/candidates.jsonl
evaluation/vag/data/labels.jsonl
```

要求：

- `candidates.jsonl` 不能包含 `label`；
- `labels.jsonl` 单独保存人工标签；
- 两人标注，保留一致样本；
- 目标仍然是 `120 benign + 80 harmful`，如果真实样本不够，就用真实 benign 经过公开 mutation protocol 生成 harmful。

### 5.2 Holdout split：跑 `ExecCritic-replay`

对每条 candidate skill，选 `3-10` 个 holdout tasks，跑两组：

```text
without_skill: agent 不注入该 skill
with_skill:    agent 注入该 skill
```

每个 task 输出一行：

```json
{"candidate_id":"skill_0042","task_id":"fix-git","without_skill_pass":true,"with_skill_pass":false}
```

保存为：

```text
results/vag/tb2_holdout_replay_records.jsonl
```

然后在本 repo 中执行：

```bash
python evaluation/vag/runners/holdout_replay.py \
  --input results/vag/tb2_holdout_replay_records.jsonl \
  --out results/vag/tb2_holdout_replay_decisions.json
```

目的：把 `ExecCritic-static` 替换成论文主 claim 需要的 `ExecCritic-replay`。

准入规则：

```text
approve iff pass_rate_with_skill >= pass_rate_without_skill - epsilon
```

建议：

- `epsilon = 0` 作为主结果；
- appendix 可报告 `epsilon = 0.05` 的敏感性。

预期结果：

- 错误命令、环境错配、破坏性操作、跳过测试等行为型 harmful skill 应被 replay 拒绝；
- 部分语义型 harmful skill 仍需要 `AgentCritic`，这正好支持异质组合。

### 5.3 Gate ablation：冻结不同 hot skill store

对同一批 candidates 跑所有 gate：

```bash
python evaluation/vag/runners/regression_injection.py \
  --candidates evaluation/vag/data/candidates.jsonl \
  --labels evaluation/vag/data/labels.jsonl \
  --gates ungated,llm_only,skillsvote_style,peek_style,schema,exec,agent,schema_exec,schema_agent,exec_agent,vag_full \
  --report results/vag/hot_pollution_rate.json
```

目的：生成论文表 2 和表 3。

预期结果：

- `ungated` harmful admission 最高；
- `schema` 主要拦格式；
- `exec` 主要拦行为回归；
- `agent` 主要拦语义风险；
- `vag_full` harmful admission 最低；
- 真实数据下 `benign_retention` 可能低于 100%，需要分析 false rejection。

### 5.4 Test split：跑真实 downstream pass@1

对每个 gate 冻结 hot skill store，在 test split 上跑同一 agent。

需要至少比较：

- `Agentica-seed`
- `Ungated evolution`
- `LLM-only soft gate`
- `Exec-only`
- `Agent-only`
- `Exec + Agent`
- `VaG-full`

目的：回答论文最关键的下游问题：准入更安全是否减少 bad-skill-induced failures，并保持或提升 pass@1。

预期结果表：

```text
Gate | Pass@1 | Avg tokens | Tool calls | Bad-skill-induced failures | Hot skills
```

判定标准：

- 不要求 `VaG-full` 大幅超过所有 baseline；
- 必须证明 `VaG-full` 相比 `Ungated` 显著减少 bad-skill-induced failures；
- pass@1 至少不明显退化，最好小幅提升；
- 若 pass@1 下降，要看是否因为 false rejection 过多，然后调 `AgentCritic` style 或 replay epsilon。

## 6. 消融实验怎么做

### 6.1 Critic ablation

必须跑：

```text
schema
exec
agent
schema_exec
schema_agent
exec_agent
vag_full
```

目的：证明 VaG 不是单个 `AgentCritic`，而是结构 / 行为 / 语义互补。

预期结果：

- `schema` 只拦 format；
- `exec` 拦 command/env/destructive；
- `agent` 拦 PII/overgeneralization/contradiction/wrong_precondition；
- `exec_agent` 仍漏 format；
- `vag_full` 最稳。

### 6.2 Gate baseline ablation

必须跑：

```text
ungated
llm_only
skillsvote_style
peek_style
vag_full
```

目的：证明 soft gate / success gate 不等于 explicit verification。

预期结果：

- `skillsvote_style` 对来自 success trajectory 的 harmful mutation 基本无能为力；
- `peek_style` 能挡一部分 PII/stale，但挡不住 exec regression；
- `llm_only` 漏执行不可复现和环境错配；
- `vag_full` 最低 harmful admission。

### 6.3 AgentCritic model sensitivity

当前 mock 版本：

```bash
python evaluation/vag/runners/cross_model.py
```

真实版本要换成真实 judge：

```text
gpt-4o / deepseek / claude / 本地 judge model
```

目的：证明 VaG 不依赖某一个完美 reviewer。

预期结果：

- judge 越弱，harmful admission 越高；
- 但因为 `SchemaCritic` + `ExecCritic` 兜底，不会退化到 `ungated`。

### 6.4 `ExecCritic-replay` epsilon sensitivity

跑：

```text
epsilon = 0
epsilon = 0.05
epsilon = 0.10
```

目的：平衡 false rejection 和 false admission。

预期结果：

- `epsilon=0` 最保守，harmful admission 最低；
- `epsilon=0.05` 可能提高 benign retention；
- `epsilon=0.10` 可能开始放进有害 skill。

### 6.5 `AgentCritic` style sensitivity

跑：

```text
strict
neutral
lenient
```

目的：找投稿默认参数。

预期结果：

- `strict` harmful admission 低，但 false rejection 高；
- `lenient` benign retention 高，但 harmful admission 高；
- `neutral` 可能是默认最稳。

## 7. 真实 TB2 跑完后还要做什么

### 7.1 结果摄取

把 Linux/Daytona 上的结果同步回本 repo：

```text
results/vag/tb2_event_traces/
results/vag/tb2_holdout_replay_records.jsonl
results/vag/tb2_test_results_<gate>.json
```

### 7.2 重新生成论文表格

```bash
python evaluation/vag/runners/holdout_replay.py \
  --input results/vag/tb2_holdout_replay_records.jsonl \
  --out results/vag/tb2_holdout_replay_decisions.json

python evaluation/vag/runners/regression_injection.py \
  --candidates evaluation/vag/data/candidates.jsonl \
  --labels evaluation/vag/data/labels.jsonl \
  --report results/vag/hot_pollution_rate.json

python evaluation/vag/analysis/make_paper_tables.py \
  --out results/vag/paper_tables.json
```

如果有真实 TB2 test split 结果，需要补一个 analysis 脚本或扩展 `make_paper_tables.py`，生成：

```text
Table 4: Terminal-Bench 2 pass@1 / tokens / bad-skill failures
```

### 7.3 人工审查失败案例

至少抽样：

- `10` 个 false accepted harmful skills；
- `10` 个 false rejected benign skills；
- 所有 bad-skill-induced test failures。

每个案例记录：

```text
candidate_id
source_trace
label
gate decision
critic verdicts
failure task
why it failed
how provenance helps rollback
```

目的：支撑 Industry Track 的 case study 和 rebuttal。

### 7.4 更新论文

需要更新：

- `docs/paper_vag_industry_track_zh.md` 的表 2、表 3、表 4；
- `§5.5 Case Studies` 的真实案例；
- `§6.3 可信边界`，删除已完成项，保留真实 limitation；
- 摘要中的实验数字；
- 结论中的数字。

注意：`docs/paper_*.md` 是保密文件，不放入 git 跟踪。

## 8. 最小投稿包判断标准

达到以下条件，VaG 论文实验基本够投 Industry Track：

- [ ] `200` 条 labelled Regression Injection，来自真实 traces + mutation protocol；
- [ ] `VaG-full` harmful admission 明显低于所有单 critic 和 soft gate；
- [ ] `benign_retention` 不低于 `80%`，并有 false rejection 分析；
- [ ] `ExecCritic-replay` 使用真实 holdout replay，而不只是 static predicate；
- [ ] Terminal-Bench 2 test split 有 pass@1、tokens、bad-skill-induced failures；
- [ ] 至少 `5` 个 provenance-backed case studies；
- [ ] 至少 `2-3` 个真实 AgentCritic judge model sensitivity；
- [ ] 所有脚本可复现，`pytest evaluation/vag/tests -q` 和 `check_ast.py` 通过。

## 9. 每次改代码后的验证

```bash
python -m pytest evaluation/vag/tests -q
python ~/.agents/rules/check_ast.py .
```

预期结果：

```text
all tests passed
All files passed
```
