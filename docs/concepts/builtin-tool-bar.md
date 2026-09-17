# 新增内置工具的判据

[Tools](tools.md) 讲的是**怎么写**一个工具。这篇讲**该不该把它做成内置**——
即默认装进 `DeepAgent` / CLI、每个用户每一轮都带着它的那种工具。

这不是打分表。agentica 不按「这个能力有多基础」投票，因为那种判据不可证伪：
档位的边界、词条的取舍，组合起来没有穷尽，而每一次判错都是静默的。
下面每一条判据都能在装配期用一个具体事实回答**是**或**否**。

---

## 先回答：它是不是 CLI-only

这是第一道门，因为它不是策略，是物理限制。

`Agent` 会被嵌入 gateway，而 gateway **一个进程服务多个 session**。
凡是「要改进程级状态」或「要有人在这个键盘前」的能力，就不能进
`get_builtin_tools()`，只能在 `cli/runtime.py` 里挂。

| 工具 | 为什么进不了 `get_builtin_tools` |
|---|---|
| `worktree` | 换 worktree 要换 **进程 cwd**，gateway 不能为其中一个 session 改（`agentica/tools/worktree_tool.py:12`） |
| `delegate` | 要 session 的 `BackgroundProcessRegistry` 才能追踪/等待/回报 worker |
| `list_agents` / `send_message` | 要 `PeerSession`，即「这台机器上用户开着的其它终端」 |
| `self_manage` / `cronjob` | 改的是这套安装的 `config.yaml` / `.env` 与用户的 cron |

判据：**这个能力的正确性依赖「只有一个 session」吗？** 依赖就是 CLI-only。
CLI-only 工具在 `agentica/cli/runtime.py:1078` 一带组装，从不进工厂。

## 有没有一个装配期就能查的前提

内置工具的前提必须是**装配时可判定的事实**，不是运行时的猜测。
没有前提就不挂——不是挂上去再在调用时报错。

```python
# agentica/agent/base.py:704 —— 记忆工具要有 workspace 才有意义
if self.enable_long_term_memory and self.workspace is not None:
    ...注册 BuiltinMemoryTool

# agentica/cli/runtime.py:1154 —— 没有 registry，或已经到 MAX_DEPTH
if background_process_registry is None or delegation_depth() >= MAX_DEPTH:
    ...移除 BuiltinDelegateTool
```

`ask_user_question` 是这条的极端形式：`create_agent()` 把
`include_ask_user_question` 做成 **keyword-only 且没有默认值**
（`agentica/cli/runtime.py:914`）。

给它一个默认值就等于替调用方猜「现在有没有人在终端前等着回答」，而这件事只有
调用方知道。`--query` 一次性运行显式传 `False`（`agentica/cli/main.py:262`）：
没有 TUI，工具会掉到一个 `input()` 上去问一个不存在的终端。
**必填让这个问题在读代码时就必须被回答，而不是在半夜的 cron 里挂住。**

## 不要用「运行时把它藏起来」来抵消代价

一个常见的辩护是「工具很多也没关系，不相关的那轮从 schema 里摘掉就行」。
agentica 明确不走这条路：

```python
# agentica/agent/permissions.py:57
def read_only_whitelist(mode: str) -> Optional[List[str]]:
    """Query-level tool whitelist for `mode`. Always None: every tier exposes all tools."""
```

原因是 prompt cache。工具定义在系统消息里，而 prompt cache 命中的条件是
**前缀逐字节相同**；系统消息在之后每一个 breakpoint 的前缀里。
按轮增删工具 = 每轮换一个前缀 = 把整段对话历史重新计价。
省下来的那点 schema token，换来的是全量重算。

所以内置工具的代价是**长期且全局**的，不能靠「需要时才出现」摊薄。
这也意味着：**上面第二条的条件挂载是装配期一次性的，不是每轮过滤。** 两者形状相似，
成本完全不同。

## 能不能由已有工具加一个参数完成

新工具名是给每个模型的每一轮都加一行 schema；新参数只对用到它的那次调用收费。

`analyze_image` 是这条的正面例子：底模能看图就把像素直接给它、配了
`vision_model` 就交给它、都没有才退到本地 OCR——**三条路一个工具名**。
OCR 没有单独暴露成 tool，因为「走哪条路」是这套安装的能力问题，
不该让模型替我们推理它。

反过来，`execute` 的 `background=True` 也不是独立的 `run_in_background` 工具。
但注意它连带的约束：没有 `BackgroundProcessRegistry` 时，
**那个参数根本不在 schema 里**——`agentica/tools/builtin/execute_tool.py:342`
把 `background` 从
`parameters["properties"]` 里主动 `pop` 掉，同时 `wait` 不注册。
实测两种模式的参数面确实不同：交互 CLI 是
`["background", "command", "parallel_safe", "timeout"]`，
`--query` 是 `["command", "parallel_safe", "timeout"]`。

**参数所承诺的能力必须真的存在**，否则模型会调一个注定拿不回结果的后台命令。
这也说明：盘点两种形态的差异时只比工具名是不够的，同一个工具的参数面也会变。

## 它的 docstring 是 prompt，不是开发文档

内置工具的 docstring 进每一轮 context，并且**同一份文本发给所有档位的模型**，
所以它的成本不只是 token：写得含糊，弱模型就照着含糊去调。

细则见 `CLAUDE.md` 的「Tool Docstring Hygiene」。一条实测出来的反例：
不要维护二进制黑名单（"not cat"、"MUST avoid find"）——那把强模型推向了更差的绕路。

---

## 落地清单

决定要内置之后，以下四处必须同步（漏一处都是静默失效）：

1. **元数据四处同步**：`is_read_only` / `is_destructive` / `concurrency_safe` 要在
   `Function` 字段、`@tool` 装饰器参数、`Tool.register()` 参数、
   `Function.from_callable()` 提取处都在——细则见 `CLAUDE.md`「Tool Metadata Sync」。
   仅执行器用的旋钮（`max_result_size_chars` / `manages_own_timeout` /
   `parallel_arg`）是例外，只挂在 `Function` 上、注册后赋值。
2. **展示清单**：`BUILTIN_TOOLS`（`agentica/cli/runtime.py:77`）。
   `tests/cli/test_builtin_tools_listing.py` 会在工厂长出一个它没列的函数时失败——
   这张清单曾经漂移过（`write_todos` 缺失，而注释还写着 single source of truth）。
3. **图标**（可选）：`TOOL_ICONS`（同文件），缺省回落 `🔧`。
4. **文档**：[Tools](tools.md) 的内置工具表。

## 为什么没有「总开关」

`builtin_tools=False` 作为一个总开关被提过，并且被否决了
（`CLAUDE.md:284`）。因为 `DeepAgent` 即使关掉它，仍会 `auto_load_mcp`
并注入 `BuiltinMemoryTool`——**这个名字会承诺一个它保证不了的保证。**

需要白名单的调用方要的不是这个预设，而是 plain `Agent` 加上真正需要的那几个
预设，外加一条构造期对最终工具表的断言。
只有后者才拦得住「新加了一个默认开启的内置工具」这种情况。

---

## 判据的实际后果：两种形态的工具面

以下是实测结果（实际 `create_agent()` 两次后遍历 `agent.tools`，非数代码）。
它是上面几条判据叠加后的产物，也是核对「我新加的工具挂在了该挂的地方」的基线。

**交互式 CLI = 24 个**

```
analyze_image, apply_patch, ask_user_question, cronjob, delegate, execute,
fetch_url, get_skill_info, glob, grep, list_agents, list_skills, read_file,
save_memory, search_memory, search_session, self_manage, send_message, task,
wait, web_search, worktree, write_file, write_todos
```

**`--query` / `--print` 一次性运行 = 18 个**（反向差集为空）

少的 6 个，每一个都对应上面的一条判据：

| 工具 | 不挂载的位置 | 缺的那个前提 |
|---|---|---|
| `ask_user_question` | `tools/builtin/__init__.py:157`，由 `cli/main.py:262` 传 `False` | 没有 TUI，没人能回答 |
| `list_agents` / `send_message` | `cli/runtime.py:1083` | 没有 `PeerSession` |
| `worktree` | `cli/runtime.py:1091` | 没有 `WorktreeBinder` |
| `wait` | `tools/builtin/execute_tool.py:335` | 没有 process registry，也就没有后台进程可等 |
| `delegate` | `agent/deep.py:275`，CLI 侧二次摘除 `cli/runtime.py:1154` | 没有 registry；或已达 `MAX_DEPTH` |

复现时有一个坑：`delegate` 的可见性还取决于 `AGENTICA_DELEGATE_DEPTH`
（`MAX_DEPTH = 1`，`tools/builtin/delegate_tool.py:44`）。
在一个**被 delegate 出来的** session 里实测，交互式那边也不会有 `delegate`，
差集看起来就只有 5 个。要的是用户真实终端的数，就得在 depth 0 下测。

## 下一步

- [Tools](tools.md) —— 怎么写工具、JSON Schema 生成、并发与安全
- [Choosing](../multi-agent/choosing.md) —— `task` / `delegate` / peer 怎么选
