# 实习生功能建议评审（agentica × A5 × Grok）

评审对象：实习生对当前 agentica 仓库的「已有能力」判断，以及 10 条功能建议。对照物：仓库现状、A5 技术博文（Hugging Face，2026-10-23，*Introducing A5*）、*Grok Bot Guide*（工程场记，24 页）、*Grok Bot Marketing Guide*（14 页）。

结论先说：这不是一份可以按原顺序排期的需求单。文件定位大体是对的，产品方向（不要把 agentica 做成 Grok 托管产品的本地复刻）也是对的。错在把几种已经分开的机制揉成一个缺口，并把场记里的提示词习惯写成了平台规格。

---

## 1. Executive take

实习生显然读过记忆、经验、cron、peer、权限这几条主路径，不是凭博客空想。十条里没有一条是「仓库里完全不存在、对方却说已经做完」的幻觉；反过来，「已有、不必再提」里有一条默认是关的。最大的事实错误不是漏文件，而是**粒度**：把记忆的 `type: project` 当成没有项目边界（边界确实没有，但常驻规则有另一条链）、把经验卡的冷热归档说成记忆完全没有归档、把 Grok「无事保持安静 / 用户投稿不可信」说成必须落地的字段和 XML 标签、把 `PeerConflictChecker` 说成 Diff Engine 的半成品。延迟数字 `p50 1.2s` 和 `<untrusted source=...>` 这个标签，仓库和两份 Grok 指南里都没有。

对 agentica 的策略应当收紧，而不是加宽。它是本地优先的 CLI / SDK / Gateway 框架：一个用户、一份工作区记忆、按 cwd 切开的会话、用 skill 而不是「每 bot 一台云 VM」来分发做法。A5 值得学的是**范围、便宜的预取、任务中途的资料变化、空闲整理、以及没话就别出声的主动建议**。Grok 值得学的是**窄角色、先读后写的放权阶梯、纠正要写成原则、例行任务要便宜且无事不扰、工具读到的第三方文本当数据**。不值得学的是每 bot 一台电脑、市场模板即产品、凭证保险库、以及不等用户点头的全自动。

因此排序要从实习生的 `2 → 5 → 1 → 4` 改掉。先把召回的语料切对（用户级 vs 当前仓库），再给这次选择加一层本地粗筛；第三方工具结果的隔离可以并行，而且不要包住用户自己打的字。Cron 拆开：无事不报和把已有的 token 用量记进 `TaskRun` 值得做；事件触发和 IM `deliver` 先不做。纠正矛盾检查先于 Autodream。只读档、长期专家模板、Inbox 都不是下一刀。

---

## 2. Corrections to intern facts

### 判定表

| 主张 | 判定 | 证据 |
|---|---|---|
| 权限档 + 审批已有 | **VERIFIED** | `agentica/agent/permissions.py`：`ask` / `auto` / `allow-all`。工具三档都留在 schema 里，差别是执行前是否停车（`agentica/agent/approvals.py` 的 allow / ask / deny，以及「拒绝类似」）。CLI `/permissions`、SDK、Gateway 同一套词。 |
| 重复纠正自动生成 skill | **PARTIAL** | 管道在：`SkillEvolutionManager.maybe_spawn_skill`，默认 `min_repeat_count=3` 且要 hot 层（`agentica/experience/skill_upgrade.py`，`SkillUpgradeConfig`）。CLI 默认关：`--enable-skill-upgrade` 是 `store_true`（`agentica/cli/runtime.py`），`DeepAgent` 把 `skill_upgrade=None`（`agentica/agent/deep.py`）。来源不限于纠正，工具错误卡也能进。产物是 shadow `SKILL.md`，不是当场改行为。 |
| `verify_completion` | **VERIFIED** | `agentica/tools/goal_tool.py`。挂在 goal 上（跑测试 / 对标准），不是每次普通对话都有的工具。 |
| 跨 agent 消息带来源 | **VERIFIED**（仅 peer） | `format_for_model`（`agentica/peers/__init__.py`）区分「用户从另一个会话转发」和「另一个 agent」。不是 `<untrusted>` 标签。 |
| 本轮已写记忆则跳过后台抽取 | **VERIFIED** | `MemoryExtractHooks._flush`（`agentica/agent/hooks.py`）：窗口里任一 turn 调过 `save_memory` 就跳过抽取 LLM，对话归档仍写。旁路：`on_post_compact` 挖 session notes 时不看这个标记。 |
| 1. 只有 `users/{id}/`，项目记忆混在用户级，召回串项目 | **PARTIAL** | 记忆正文确实全在 `users/{user_id}/memory/` + `MEMORY.md`（`agentica/workspace.py`，默认工作区 `~/.agentica/workspace`）。`memory_type` 含 `project`，但是同一索引上的标签，召回不按仓库过滤（`agentica/memory/recall.py`）。常驻规则不是这样：用户级 `users/{id}/AGENTS.md` 与仓库链上的 `AGENTS.md` / `CLAUDE.md` / `.cursorrules` 是分开的（`Workspace._collect_agent_md_sources`）。会话日志按 cwd 分项目，记忆不分。 |
| 2. 每轮都调辅助模型（p50 1.2s）；`compute_relevance_score` 已在，缺 BM25 粗筛 | **PARTIAL**，两处事实错 | CLI / Web / `DeepAgent` 默认 `recall_memories=True`，有未露面的索引条目时才会调辅助模型（`build_memory_recall`）。空索引、本窗口已召回过、连续失败 3 次熔断，都不调。延迟注释写的是 p95：chat 1.7s、Anthropic 2.9s，超时 5s，约 4.5% 在 7–11s（`WorkspaceMemoryConfig.recall_timeout`）。**没有 p50 1.2s。** `compute_relevance_score` 是词重叠 + 字符 bigram，不是 BM25；生产召回不用它。`search_memory` 和示例里的 `get_relevant_memories` 才用。 |
| 3. 只有边界抽取；无去重/冲突/归档；记忆目录不进 git；`memory_candidates/` 只能手提升 | **PARTIAL** | 边界抽取属实（默认每 10 轮或 compact，间隔 60s）。**经验卡**有 hot/warm/cold 和 `archive_days`（`CompiledExperienceStore.run_lifecycle`），记忆文件没有。记忆不在项目 git 里，也没有任何对 `MEMORY.md` 的 git 版本化。候选区更窄：自动抽取 `source="auto_extract"` 进 `memory_candidates/`；模型自己的 `save_memory` 默认 `source="verified"`，直接进正式记忆。`promote_memory_candidate` 只是 `Workspace` 方法，CLI / 工具 / UI 都没接。`search_memory` 会把候选和正式记忆一起搜出来并标明 `source`。 |
| 4a. 无事可报时应沉默 | **PARTIAL** | 运行结果总会写入 run 记录（`mark_job_run`）。没有「空报告不打扰」的运行时约定。Grok 指南是让人在 routine 的提示词里写 “stay quiet on a no-op”，不是平台字段。 |
| 4b. `TaskRun` 无 token/费用，`/cron list` 不显示 | **VERIFIED** | `TaskRun` 只有 run_id、status、起止时间、attempt、result、error（`agentica/cron/types.py`）。`CliAgentRunner.run` 只返回 `response.content`。`/cron` 列表是 id / 名字 / 状态 / 下次运行；`/cron runs` 是状态和时间，不显示用量，也不显示结果正文。模型侧 `Usage` 已存在，没有接到 cron。 |
| 4c. 无事件触发（文件变化 / webhook） | **VERIFIED** | 调度只有 `at` / `every` / `cron`。`/webhook/feishu` 是渠道入站，不是任务触发器。 |
| 4d. `deliver` 存了不用 | **VERIFIED** | `CronJob.deliver` 默认 `local`，创建 API 会存。`scheduler._execute_job` 不读它。`docs/guides/cron_scheduler.md` 写明不会推到 IM。 |
| 5. 只有 peer 有来源标签；`fetch_url` / `web_search` / IM 入站应包进 `<untrusted source=...>` | **PARTIAL** | peer 标签属实。`fetch_url` / `web_search` 原样返回 JSON（`agentica/tools/builtin/web_tools.py`）。IM 入站把用户自己的文字当作 `message` 送进 `agent_service.chat`（`gateway/main.py` 的 `_process_channel_message`）。Grok 指南的原话是：别人提交的反馈文本不可信，**在搭建流水线的提示词里说清楚**，并防 prompt injection。指南里没有这个 XML 标签。把用户自己的 IM 包成 untrusted 是设计错误。 |
| 6. `PeerConflictChecker` 在，但没接到文件工具；应用 mtime/hash 跟踪 `read_file` | **PARTIAL** | 前半句对：`agentica/peers/conflicts.py` 模块文档写明**故意不**接到 `BuiltinFileTool`，只回答「另一个活会话的 git dirty 是否包含这个路径」，且不阻断写入。这不是 A5 Diff Engine（任务进行中，已连接资料变了，下一次模型调用前注入）。`read_file` 没有 mtime/hash 失效。类注释里的 “content cache” 没有对应实现。 |
| 7. `.agentica/agents/*.md` 只有 prompt+tools，无 `memory_scope`，无导入导出 | **PARTIAL** | 文件比「prompt+工具」富：`description`、`allowed_tools` / `denied_tools`、`execute_policy`（`inherit` \| `read_only`）、`model_tier`、`max_turns`、`timeout`、`inherit_context/workspace/knowledge`（`agentica/subagents/loader.py`，示例 `bundled/code.md`）。它们是 `task` 子代理，用完即走。`save_memory` 在子代理 `BLOCKED_TOOLS` 里。没有 `memory_scope`，也没有「去掉凭证和聊天记录再导出」的模板。Grok 的长期专家是持久 bot；agentica 这边更接近的是 peer 会话 + 全局 skill + 一份用户记忆。 |
| 8. 权限只有 ask/auto/allow-all，缺只读档 | **PARTIAL** | 枚举确实只有这三档。`ask` 已经自动放行读（含工作区外只读 shell），写和联网才停车，所以它不是「不能写」。子代理另有 `execute_policy: read_only`。无人值守（cron / IM / `--print`）遇到本该停车的调用，直接写成 `Tool call denied by user.`，不会等人。Grok 的阶梯是 rollout 政策（先只读摘要，再内部草稿，再执行），不是第四个全局枚举。 |
| 9. 经验卡压成 ≤8 词；高置信单次纠正立即生效；无矛盾检查 | **PARTIAL** | ≤8 词只约束分类器的 `rule` 字段，且仅当用户没有给出可逐字抄的引号规则（`prompts/memory/md/feedback_classify.md`）。卡片正文仍保留 why / how_to_apply。工具错误卡和成功模式卡不是这个格式。置信度 ≥ 0.8（默认）且 `should_persist` 会写卡片；显式规则前缀预过滤置信度 1.0，不必等第二次。但「立即生效」过了：LLM 评判默认每 10 轮 / 60 秒才跑；经验索引在会话开始时冻结，本会话系统提示词看不到新卡。矛盾检查确实没有。去重是 `correction_key` 词干撞文件名、`repeat_count` +1，不是语义冲突。 |
| 10. 没有 Proposal Engine / Inbox | **VERIFIED** | 无建议历史、无用户收件箱。peer mailbox 是 agent 之间的信，不是给用户确认的建议队列。 |

### 需要改口的具体句子

- 「只有 `users/{id}/`」应改成：记忆正文在用户分区；项目级**规则**已经有仓库 `AGENTS.md` 链；缺的是项目级**事实**的存放和召回过滤。
- 「BM25 粗筛，因为 `compute_relevance_score` 已经在」应改成：现有打分不是 BM25，而且生产召回路径根本没调用它。p50 1.2s 请删掉，改引用配置注释里的 p95，或者先量一次再写进需求。
- 「记忆没有去重/归档」应改成：记忆文件没有；经验卡有生命周期。Autodream 不该从零设计，先问能不能复用 `run_lifecycle`。
- 「`memory_candidates/` 只能手工提升」应改成：提升函数存在，产品表面没入口；同时模型调用的 `save_memory` 绕过隔离区，直接当可信写入。
- 「Grok 要求 `<untrusted source=...>`」应改成：Grok 要求把**别人写的文本**当成不可信输入，并在提示词里写明。标签格式是实习生自己的机制建议，可以保留为实现草案，不能写成外部规格。用户自己从 IM 发来的话不是第三方文本。
- 「把 `PeerConflictChecker` 接到文件工具，再记 mtime」应拆开。Checker 解决的是两个活会话改同一仓库文件；A5 Diff Engine 解决的是本轮已经读过的资料随后被改了。接错线会得到一种告警，解决另一种问题。
- 「高置信单次纠正立即生效」应改成：单次可以落盘（还常常要等一个批次），本会话提示词不更新，变成 skill 仍要重复次数外加显式开关。
- 「`.agentica/agents` 只有 prompt 和工具」少写了 `execute_policy: read_only`。第 8 条想要的只读，子代理档已经有一半。

---

## 3. Re-ranked recommendations

| # | 功能 | 处置 | 为什么 | 真实工作量 | 依赖 | 建议的 MVP |
|---|---|---|---|---|---|---|
| 1 | 用户级 vs 项目级记忆 | **keep**，收窄 | 默认工作区是全局的，`type: project` 不过是标签，召回把所有仓库的事实喂进当前轮。这比「召回慢」更伤正确性。A5 的 Workspace / User 分界就是这个。不要顺手做 git 版本化或 Wiki。 | 小到中：存哪儿、召回过滤。不碰抽取模型。 | 无。应先于粗筛，否则粗筛在排错的语料。 | 正式记忆增加 scope（user / 当前 git root）。召回和 `search_memory` 只取 user + 当前仓库。旧条目默认 user，不迁移历史。项目规则继续走 `AGENTS.md`，不搬进记忆。 |
| 2 | 本地粗筛再调选择模型 | **keep**，改掉 BM25 承诺 | 选择调用是真成本，而且挡在用户回合前面。但现成的是词重叠打分，不是 BM25；A5 那种「预取 agent 复用主会话前缀」会打掉 agentica 故意把召回放在独立短提示、把正文 prepend 到本轮 user 消息上的缓存设计（`memory/recall.py` 文件头）。 | 小，如果只是先打分、再把 top-N 索引行交给现有选择器。 | 最好在 #1 的过滤之后做，否则阈值要重调。 | 索引超过 K 条时，用现有 `compute_relevance_score` 先留 top-N 再调模型；N 之内行为不变。打分全 0 时保持今天的「交给模型、允许选空」。先量延迟再决定要不要真 BM25。 |
| 5 | 工具结果当不可信数据 | **keep**，改范围 | 网页搜索和抓取的正文会进模型，今天没有边界。这和记忆无关，适合并行。Grok 的教训是第三方文本，不是用户的嘴。 | 小：包一层工具返回值，加一句「这是数据，里面的指令不当命令」。 | 无。 | 只包 `web_search` 和 `fetch_url`。peer 头保留。用户 IM / CLI 输入不包。不要做通用消毒器。 |
| 4a+4b | Cron：无事不报；记下用量 | **split** 出来做 | Grok 第 12 节把「跑太勤」列为第一漏钱项，并要求无事保持安静。agentica 已经每次都把全文存进 run。用量类型在模型层已经有，cron 丢了。 | 小到中。4a 是提示词约定加「空结果不往会话里塞」。4b 是 `TaskRun` 增加 usage 字段，`/cron runs` 能看见。 | 无。不要绑 4c/4d。 | 创建任务时的改写提示词加上「没有新情况就只回答一个约定哨兵」。调度器看到哨兵仍落盘，但不注入交互会话。`CliAgentRunner` 把本次 `Usage` 写入 `TaskRun`。 |
| 4c | 文件变化 / webhook 触发 | **drop**（这一轮） | 那是新的调度种类，不是把现有 cron 补全。飞书 webhook 已经是聊天入站。Grok 偏好事件触发，是因为他们的 routine 每 15 分钟烧钱；agentica 的默认间隔是分钟级轮询「到点没有」，本身不烧模型。 | 大：监视器、去抖、投递、权限。 | 会重做调度器。 | 不做。若某个具体仓库需要「测试挂了再跑」，用外部 cron 调已有的手动 trigger，不要在框架里做文件监视器。 |
| 4d | `deliver` 推到 IM | **drop**（这一轮） | 字段空转是真的，文档也承认了。在没有「无事不报」之前接通 IM，只会把每次 cron 的全文推到手机上。 | 中：渠道寻址、失败、幂等。 | 依赖 4a，否则是噪音放大器。 | 不做。结果继续留在 run 历史；prompt 里需要发消息时，让任务自己调用已有渠道工具。 |
| 9 | 纠正：原则化 + 写前矛盾检查 | **keep**，砍掉「立即生效」 | ≤8 词和单次落盘已经大致在了。缺的是：新规则和已有卡片相反时不要静默并存。这比 Autodream 小，而且保护你刚切开的记忆。Grok 的原话是「写成原则，删掉那一次事故」，不是「再压短」。 | 小：写卡前对同 key / 相邻卡做一次对照；冲突则留下两边并标待确认，不覆盖。 | 最好在 #1 之后，这样项目规则不会和用户习惯比。 | 只检查 `correction` 卡。不改 8 词提示词。不把单次纠正提前注入本会话系统提示词（那会打掉前缀缓存，`freeze_snapshots` 是故意的）。 |
| 3 | 空闲整理 / Autodream | **merge** 进已有生命周期，推迟 | A5 的空闲整理（去重、过期、归档、把可迁移经验升到用户级）值得做，但经验卡已经有一版。记忆侧缺的是归档和使用情况，不是另一个后台 agent。候选区今天连提升入口都没有，整理器会堆在一个没人看的目录上。 | 中，如果复用 lifecycle；大，如果新开一个会改 `MEMORY.md` 的空闲 agent。 | 依赖 #1 的 scope，否则会把 A 仓库的事实并进用户级。 | 先做两件确定性的事：候选区一个显式提升/丢弃命令；长期未召回的记忆标 cold，默认召回跳过。git 提交记忆目录先不做（那是家目录，不是仓库）。 |
| 6 | 任务中途文件变化 | **keep**，换机制 | 多会话写冲突和「我读过的文件被人改了」是两件事。Checker 不接文件工具是书面决定，不要推翻它来假装做 Diff Engine。 | 中：本轮 `read_file` 记路径和 mtime/hash，下一次模型调用前如果变了，插一条短注。 | 无，但优先级低于记忆正确性。 | 只跟踪本轮读过的工作区文件。变了就告诉模型，不阻断，不看别的会话的 git dirty。 |
| 8 | 全局只读权限档 | **drop** | 第四个模式会和 `ask`（读已经自动放行）以及子代理 `execute_policy: read_only` 重叠。Grok 阶梯是「新 bot 先只读」，不是每个人的默认档。真正别扭的是无人值守把「询问」变成「假装用户拒绝」。 | 若硬做枚举：中，而且要改三端。若只给 cron 一个预设：小。 | 若做 cron 预设，和 4a 一起想，但不阻塞 4a。 | 不做新档。需要天花板时，给单个 cron 任务一个「拒绝写」预设，而不是全局只读。 |
| 7 | 长期专家 + 记忆 + 导入导出模板 | **drop**（框架层） | 子代理定义已经能表达只读专家。持久人格、独立记忆、可分享模板是 Grok 的产品拓扑（记忆 per bot，skill 全局，模板去掉凭证和聊天记录）。agentica 把做法分发在 skill hub（GitHub / skills.sh / LobeHub），把并行会话放在 peer。再做一套 bot 市场是第二条产品线。 | 大。 | 看起来依赖 #1 的 `memory_scope`，那会把 scope 做成 per-bot，和「一个用户一份记忆」对着干。 | 不做。要窄角色，写一个 `.agentica/agents/*.md` 或一个 skill。 |
| 10 | 主动建议 + Inbox | **drop**（这一轮） | A5 自己说：没有值得推进的新信息就保持安静，被拒绝过的建议要说明发生了什么实质变化。agentica 还没有安静的 cron，也没有分范围的记忆。薄 Inbox 会变成第二条通知管道。 | 大，即便「薄」。 | 依赖 #3 和 4a，实习生这点判断是对的，所以更不该排在前面。 | 不做。以后若做，第一条规则是默认不发，而不是先做收件箱 UI。 |

---

## 4. Better sequencing

实习生的 `2 → 5 → 1 → 4` 把延迟放在正确性前面，又把 cron 当成一个功能。建议改成：

**先 #1 的过滤，紧接着 #2 的粗筛，#5 与它们并行。然后只做 cron 的 4a 和 4b。再做 #9 的矛盾检查。#6 和瘦身版 #3 放在这之后。8、7、10、4c、4d 不做。**

理由按风险、杠杆和依赖分开说。

记忆串项目是正确性风险：召回越快，错误事实进上下文越勤。粗筛的候选集就是索引，scope 一切，分数和阈值都变。所以 #2 不是 #1 的前置，#1 是 #2 的前置。两者都可以很小，没有必要分成两个季度；只是不要先做一个会把全部项目记忆排进 top-N 的检索器。

#5 的杠杆高、依赖为零，但实习生把它排成第二是因为把它理解成「Grok 安全规格」。改完范围之后它是一个工具返回值的包装，可以和记忆改动同一周落地，不必占据主路径的顺序。

Cron 的杠杆在 Grok 经济性那一节，不在 `deliver` 字段。无事仍全文落盘、用量看不见，这两件事让人无法回答「这个定时任务值不值」。事件触发和 IM 投递会打开新的失败面（监视器、渠道、重复通知），而且在 4a 之前接通是放大噪音。拆开之后，4a/4b 没有记忆依赖，可以跟在 #1/#2 后面，而不是和它们抢第一。

#9 依赖「已经知道这条规则属于用户还是这个仓库」，所以放在 #1 之后。它比重写 Autodream 便宜，而且能拦住「两条相反原则都注入提示词」。#3 要等 scope 稳定，否则空闲任务会把项目事实晋升成用户事实——A5 把这件事当成整理器的小心点，不是副产品。

#6 是真问题，频率低于「每次召回都可能串项目」。用读集失效，不要接 `PeerConflictChecker`。

这个顺序也尊重 agentica 已经为缓存做的选择：召回是独立短调用，正文贴在最新 user 消息上，系统提示词前缀不动。A5 为了预取缓存命中去复用主会话前缀，在这里是倒退，不要排进任何一刀。

---

## 5. Missing opportunities

实习生的「先别抄」清单是对的，但漏了几件和 A5 / Grok **以及现有缺口**都对得上、又比 Inbox 更值的事。

- **把验证从 goal 里请出来。** Grok 指南把验证写成三天里唯一该带走的建造指令：可重复的检查手段、机器可读的功能地图、证据跟着结果走。agentica 的 `verify_completion` 只在 goal 上。普通回合和 cron 仍然可以口头说完成。比 Inbox 更贴这个仓库的一刀是：cron / 任务的完成态要带上实际跑过的命令结果，而不是再做一个建议队列。
- **仓库怎么被验证，写成 agent 读得懂的说明。** 仓库里没有 `llms.txt`。Grok 场记里 Thursday Arena 的 `/rules` 和 llms.txt 式端点，是为了让 agent 不用猜规则。agentica 已有 `AGENTS.md` 链，缺的是「测什么、命令是什么、入口在哪」这种地图，而不是 Workspace Wiki。营销指南里的 agent-readable docs 落在这里，不落在新频道。
- **分发单位继续是 skill，不是 bot 模板。** 营销指南的 “template as distribution”（报纸 bot、`Create a template of yourself`、模板带着记忆和 routine、去掉凭证和聊天记录）是 Grok 市场的增长回路。agentica 已经有 skill hub。高价值的对应物是：导出一个 skill 加一小段 `AGENTS.md`，而不是导出一个带记忆的人。不要做 bot 市场。Proof vault（证言、播放量、截图、能否对外引用）是营销资产库；工程上的对应物是验证证据，不是一个新的保险库产品。凭证库同理，见下一节。
- **无人值守的权限语义比新的只读档更急。** cron、IM、`--print` 在需要审批时直接假装用户拒绝。这让 `ask` 在定时任务上既不像只读、也不像自动。定一个任务级预设（拒绝写，或明确 allow-all）比增加全局第四档清楚。实习生把这收成了「缺 read-only」。
- **不要「修复」召回的缓存隔离。** A5 预取复用主前缀是为了他们的缓存。agentica 把选择器放在独立提示里，是为了主前缀不变。下一份建议书如果把这写成缺口，是在用另一套产品的约束改这套产品的优点。

没有更该插队的大模块。Context 语义索引、Workspace Wiki、RSI，都比上面这几条远。

### 营销附录（来自 Grok Bot Marketing Guide，不单独立项）

- **模板即分发。** 报纸 bot 的产品就是别人能装上的模板；项目经理 bot 被要求「做成我可以分享的模板」。对 agentica，分享物应是 skill 和规则片段。记忆和聊天记录按指南也不该进模板。
- **Proof vault。** 「没人相信任何事，所以你得有证据」：证言、数据、截图、是否核实、能否公开。不要在框架里做这个库。若要借鉴，只借鉴「结果旁边有证据」——测试输出、命令退出码——接到已有的 `verify_completion` 和 cron run。
- **给 agent 读的文档。** 工程指南里的 llms.txt 比营销指南更具体。营销侧只是反复强调 bot 要有明确的真相来源。agentica 的真相来源已经是 `AGENTS.md` + 仓库本身；补一张验证地图即可，不必做站点生成器。

---

## 6. Explicitly affirm or reject the "do not copy" list

**同意不抄这三件。**

- **每 bot 一台云 VM + computer use。** Grok 的隔离单位是「一台 Linux VM、一份记忆、共享文件系统」。他们自己的经济性章节把浏览器自动化列为第二漏钱项，并说有 API 就不要用电脑操作。agentica 跑在用户自己的机器上，进程权限就是 OS 用户。再套一层 VM 专家，得到的是托管产品，不是更好的本地框架。并行用现有 peer / worktree，而不是每人一台电脑。
- **全自动。** Grok 收场时把克制当成必备技能；直播里一个自主的坏 SQL 打挂过生产。A5 的建议也要进 Inbox 等用户点头，没有新信息就不出声。agentica 已有显式的 `allow-all`。把它变成主动默认，是他们当场看过的失败模式。
- **凭证保险库。** Grok 的保险库是因为 bot 有自己的电脑、要重新登录，而且模板不能带走密码。agentica 以当前 OS 用户运行，写入记忆前会打码密钥（`docs/concepts/memory.md`）。保险库是 1Password 那类产品面。不要在记忆或模板里存秘密，也不要为了「对齐 Grok」做一套金库。

**这张清单还不完整，建议一并不要抄：**

- bot 市场，以及「带记忆的可分享人格」。
- 为了像 A5 而让召回复用主会话前缀。
- 把 `PeerConflictChecker` 接到写文件上，当作 Diff Engine。
- Workspace Wiki、候选索引的发布仪式、RSI。这些是 A5 的长期研究，不是这个框架的下一刀。

---

## 7. Concrete next ask

请拍板下面三件产品决定。定了再写代码。

1. **记忆要不要按仓库切开？** 默认工作区是 `~/.agentica/workspace`，所有项目的事实进同一份 `users/{id}/memory/`，召回不看当前 git root。建议：用户级一份，当前仓库一份；召回只带用户级 + 当前仓库。旧记忆一律视为用户级，不自动归类。若你其实是「一个仓库一个 workspace 目录」，这个问题比实习生写的小，请明确说，我们就不做 scope。
2. **定时任务要不要学会闭嘴，并让人看见花了多少 token？** 建议做：无新情况不往会话里塞；`/cron runs` 能看见用量。不做：文件监视、webhook 触发、把 `deliver` 接到 IM。
3. **网页搜索和抓取的正文，要不要明确标成数据而不是指示？** 建议只包 `web_search` 和 `fetch_url`。不包用户在 CLI 或 IM 里自己发的话。peer 消息维持现在的来源头。

未列入上述三件的（只读新档、专家导入导出、Inbox、空闲大整理、读集失效、纠正矛盾检查）先不动。其中矛盾检查和读集失效是同意 #1 之后的自然下一问，不是这次要一起批的范围。
