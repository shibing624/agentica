# Tools

工具赋予 Agent 与外部世界交互的能力。Agentica 提供内置工具，支持自定义工具、`@tool` 装饰器、并发安全标注和 MCP 协议集成。

## 工具架构

```
Tool (容器 — 一组相关功能)
  +-- Function (Schema + 入口点)
  |     +-- name, description
  |     +-- parameters (JSON Schema，自动从类型注解生成)
  |     +-- entrypoint (实际调用的函数)
  |     +-- concurrency_safe, is_read_only, is_destructive
  +-- FunctionCall (单次调用实例)
        +-- args, result, error, timing
```

**关键设计**：工具的 JSON Schema（`Function.parameters`）在首次传给 LLM 前由 `process_entrypoint()` 自动从 Python 类型注解生成。支持 `str`、`int`、`float`、`bool`、`list`、`dict`、`Literal`、`Optional`、`Union` 等类型，`Literal` 类型生成 `enum` 约束，有效引导 LLM 传入合法枚举值。

## 创建自定义工具

### 方式一：普通函数（最简方式）

任何带**类型注解**和 **docstring** 的 Python 函数都可以作为工具：

```python
from agentica import Agent, ZhipuAI

def get_weather(city: str, unit: str = "celsius") -> str:
    """获取指定城市的实时天气

    Args:
        city: 城市名称，如 "北京"、"上海"、"New York"
        unit: 温度单位，"celsius"（摄氏）或 "fahrenheit"（华氏）
    """
    # 实际场景中调用天气 API
    return f"{city}: 晴天，22°C"

agent = Agent(model=ZhipuAI(), tools=[get_weather])
result = agent.run_sync("上海今天天气怎么样？")
print(result.content)
```

**工具命名规则**：
- 工具名 = 函数名（自动使用）
- 描述 = docstring 第一段
- 参数说明 = `Args:` 部分（提升 LLM 理解准确率）
- 参数类型 = Python 类型注解 → 自动生成 JSON Schema

### 方式二：异步函数（I/O 密集型推荐）

```python
import aiohttp
from agentica import Agent, ZhipuAI

async def fetch_stock_price(symbol: str) -> str:
    """获取股票实时价格

    Args:
        symbol: 股票代码，如 "AAPL"、"000001.SZ"
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/stock/{symbol}") as resp:
            data = await resp.json()
            return f"{symbol}: {data['price']} USD"

agent = Agent(model=ZhipuAI(), tools=[fetch_stock_price])
```

Agentica 自动检测 sync/async：同步函数在 async 上下文中通过 `run_in_executor()` 在线程池中执行，不阻塞事件循环。

### 方式三：`@tool` 装饰器（精细控制）

`@tool` 装饰器让你显式控制工具的元数据和执行行为：

```python
from agentica.tools.decorators import tool
from agentica import Agent, ZhipuAI

@tool(
    name="search_code",
    description="在代码库中搜索指定模式",
    concurrency_safe=True,    # 可与其他只读工具并发执行
    is_read_only=True,        # 声明为只读，不修改状态
)
def search_in_codebase(pattern: str, path: str = ".") -> str:
    """在代码库中搜索指定模式

    Args:
        pattern: 搜索的正则表达式或字符串
        path: 搜索的目录路径
    """
    import subprocess
    result = subprocess.run(["rg", pattern, path], capture_output=True, text=True)
    return result.stdout or "No matches found"

@tool(
    name="delete_file",
    is_destructive=True,      # 声明为破坏性操作，提示 Agent 谨慎使用
    stop_after_tool_call=True # 执行后立即停止 Agent（用于危险操作）
)
def delete_file(path: str) -> str:
    """删除指定文件（不可恢复）

    Args:
        path: 要删除的文件路径
    """
    import os
    os.remove(path)
    return f"Deleted: {path}"

agent = Agent(model=ZhipuAI(), tools=[search_in_codebase, delete_file])
```

**`@tool` 装饰器参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | 函数名 | 覆盖工具名 |
| `description` | str | docstring | 覆盖工具描述 |
| `concurrency_safe` | bool | `False` | True=可与其他工具并发执行（适合只读操作） |
| `is_read_only` | bool | `False` | True=只读工具（不修改状态） |
| `is_destructive` | bool | `False` | True=破坏性操作（删除、覆写、发送） |
| `stop_after_tool_call` | bool | `False` | True=执行后立即终止 Agent |
| `show_result` | bool | `False` | True=工具结果同时展示给用户 |

### 方式四：类工具（封装一组相关功能）

继承 `Tool` 基类，适合有内部状态或需要依赖注入的场景：

```python
from agentica import Tool, Agent, ZhipuAI

class DatabaseTool(Tool):
    """数据库查询工具，支持 SELECT 和 INSERT"""

    def __init__(self, connection_string: str):
        super().__init__(name="database")
        self.conn_str = connection_string
        # 注册工具函数，并声明元数据
        self.register(self.query, is_read_only=True, concurrency_safe=True)
        self.register(self.insert, is_destructive=True)
        self.register(self.list_tables, is_read_only=True, concurrency_safe=True)

    def query(self, sql: str, limit: int = 100) -> str:
        """执行 SQL 查询并返回结果

        Args:
            sql: SELECT 语句
            limit: 最多返回的行数
        """
        # 实际场景中执行数据库查询
        return f"Query results for: {sql}"

    def insert(self, table: str, data: dict) -> str:
        """向表中插入一行数据

        Args:
            table: 表名
            data: 要插入的数据字典
        """
        return f"Inserted into {table}: {data}"

    def list_tables(self) -> str:
        """列出数据库中所有表"""
        return "users, orders, products, inventory"

    def get_system_prompt(self) -> str:
        """可选：注入工具使用指南到 System Prompt"""
        return """## Database Tool Guidelines
- Always use SELECT before INSERT to check existing data
- Limit SELECT queries to 100 rows unless user needs more
- Never execute DELETE or DROP statements
"""

agent = Agent(
    model=ZhipuAI(),
    tools=[DatabaseTool("sqlite:///app.db")]
)
```

`get_system_prompt()` 返回的内容会被自动装配进 Agent 的 system prompt：
- 工具策略类内容进入静态的 tool policy 区域
- 技能类内容进入动态的 session guidance 区域

因此它们仍然会生效，但不再一律直接拼进 `instructions`。

## 并发工具执行

当 LLM 一次返回多个工具调用时，Agentica 自动并行执行：

```
LLM 返回: [read_file("a.py"), read_file("b.py"), grep("TODO")]
          ↓
          asyncio.gather() 并发执行（3 个工具同时跑）
          ↓
结果全部就绪后，一起追加到消息历史，继续 LLM 调用
```

**前提**：工具必须标注 `concurrency_safe=True`（内置的 `read_file`、`glob`、`grep` 等只读工具默认为 `True`）。写操作工具（`write_file` 等）保持串行，并按模型发出的顺序执行。

```python
# 这两个工具可以并发：
self.register(self.read_config, concurrency_safe=True, is_read_only=True)
self.register(self.read_schema, concurrency_safe=True, is_read_only=True)

# 这个工具必须串行（写操作）：
self.register(self.write_output)  # concurrency_safe=False (默认)
```

### 由调用方逐次决定：`execute`

`execute` 在注册时无法回答「能不能并行」——同一个工具既跑 `pytest`，也跑 `git commit`。
固定成 `False` 会把互相独立的命令白白串起来，固定成 `True` 会让 `git add` 和 `git commit`
互相竞争。所以这个判断下放到每一次调用：

```python
# 工具侧：声明由哪个参数决定
self.functions["execute"].parallel_arg = "parallel_safe"
```

模型在同一条消息里发多个 `execute(command=..., parallel_safe=True)` 才会真正并发；
不带这个参数时整批串行、保持顺序，并且其中一条报错会取消后面的（sibling-abort）。
只有当批次内每条命令读写的东西互不相交时才应该带上它——有依赖关系的命令应该用
`&&` 写在同一条 `command` 里。

调度器读的是 `FunctionCall.is_concurrency_safe()`：未设置 `parallel_arg` 的工具沿用
函数级的 `concurrency_safe`，设置了的则以本次调用传入的值为准（缺省仍回落到函数级）。

> `background=True` 是**生命周期**开关（命令能否活过这一轮），不是并行入口。需要同时跑多条
> 且这轮就要结果，用 `parallel_safe`；`wait` 只服务于 `background`。

## 流控异常

工具可以通过特殊异常控制 Agent 的执行流程：

```python
from agentica.tools.base import StopAgentRun, RetryAgentRun

def validate_input(query: str) -> str:
    """验证并处理用户输入

    Args:
        query: 用户查询内容
    """
    if not query.strip():
        # 告诉 Agent 重新考虑，用 message 作为反馈
        raise RetryAgentRun("query 不能为空，请重新提供有意义的查询内容")

    if any(word in query.lower() for word in ["delete", "drop", "truncate"]):
        # 立即终止整个 Agent 执行
        raise StopAgentRun("检测到高危 SQL 关键字，拒绝执行")

    return f"处理结果: {query}"
```

| 异常 | 效果 |
|------|------|
| `RetryAgentRun(message)` | Agent 收到 `message` 作为工具返回值，重新思考并重试 |
| `StopAgentRun(message)` | 立即终止整个 Agent 运行，返回 `message` 作为最终响应 |

## JSON Schema 生成机制

Agentica 自动从 Python 类型注解生成工具的 JSON Schema，传给 LLM 约束参数格式：

```python
from typing import Literal, Optional, List

def process_data(
    input_text: str,           # → {"type": "string"}
    max_items: int = 10,       # → {"type": "number"}
    include_meta: bool = False,# → {"type": "boolean"}
    format: Literal["json", "csv", "txt"] = "json",  # → {"type": "string", "enum": ["json","csv","txt"]}
    tags: Optional[List[str]] = None,  # → {"type": "array", "items": {"type": "string"}}
) -> str:
    """处理数据并返回结果"""
    ...
```

生成的 Schema（LLM 收到的）：

```json
{
  "name": "process_data",
  "description": "处理数据并返回结果",
  "parameters": {
    "type": "object",
    "properties": {
      "input_text": {"type": "string"},
      "max_items": {"type": "number"},
      "include_meta": {"type": "boolean"},
      "format": {"type": "string", "enum": ["json", "csv", "txt"]},
      "tags": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["input_text"]
  }
}
```

**`Literal` 类型**尤其重要：它生成 `enum` 约束，有效防止 LLM 传入不合法的枚举值（否则 LLM 可能传入 `{"format": {"type": "csv"}}` 这种错误格式）。

## 内置工具

### DeepAgent 内置工具（`agentica.tools.builtin`）

这些工具由 `DeepAgent` 和 CLI 自动装载，无需手动配置：

```python
from agentica.tools.builtin import get_builtin_tools

# 获取全套内置工具
tools = get_builtin_tools(work_dir="./")
```

| 工具 | 模块 | 功能 |
|------|------|------|
| `read_file` | `BuiltinFileTool` | 读文件（offset/limit 分页，`tail` 取尾、负 `offset` 从末尾开窗；大文件守卫只拦从头分页，从尾扫描 20s 超时） |
| `write_file` | `BuiltinFileTool` | 创建/覆写文件 |
| `write_html` | `BuiltinFileTool` | 写单文件 HTML 报告（inline CSS），返回 `file://` URL。工作目录内目标免审批；用户指定的目录外目标走与 `write_file` 相同的审批。**DeepAgent / CLI / Web 默认开**；`get_builtin_tools()` 默认关（`include_html_report=True` 打开） |
| `apply_patch` | `BuiltinFileTool` | 一次补丁新增、更新或删除多个文件 |
| `glob` | `BuiltinFileTool` | 文件模式匹配（`**/*.py`） |
| `grep` | `BuiltinFileTool` | 内容搜索（基于 ripgrep；`limit` 是全局条数上限，读满即停 rg） |
| `execute` | `BuiltinExecuteTool` | Shell 命令执行（git/pytest/pip 等）；`parallel_safe=True` 同轮并发，`background=True` 进后台 |
| `wait` | `BuiltinExecuteTool` | 等待后台命令 / `delegate` 结束并取回结果 |
| `web_search` | `BuiltinWebSearchTool` | 网页搜索（引擎可替换，见下节） |
| `fetch_url` | `BuiltinFetchUrlTool` | 抓取网页内容 |
| `write_todos` | `BuiltinTodoTool` | 任务清单管理 |
| `task` | `BuiltinTaskTool` | 同进程启动子 Agent（subagent，默认 auxiliary、偏只读） |
| `delegate` | `BuiltinDelegateTool` | **交互 CLI 专用**：另起完整 `agentica --query --print` 进程；不阻塞，经 `/ps`/`wait`/`/stop` 托管 |
| `list_agents` / `send_message` | `PeerMessagingTool` | **交互 CLI 专用**：跨终端对等会话发现与纯文本消息 |

#### `task` vs `delegate` vs peer

| | `task` | `delegate` | peer |
|---|---|---|---|
| 场景 | SDK / CLI 都可用 | 仅交互 CLI（需要 `BackgroundProcessRegistry`） | 仅交互 CLI（需要 `PeerSession`） |
| 进程 | 同进程 | 新 OS 进程 | 两个用户开的终端 |
| 成本 | 低（aux 模型） | 高（完整 agentica run） | 几乎只是写 mailbox |
| 选用 | 搜代码、查资料 | 独立 context / 换目录的大活 | 通知另一会话，不是雇工 |

选型见 [Choosing](../multi-agent/choosing.md)；CLI 行为见 [终端文档](../getting-started/terminal.md)。

#### 替换 web 搜索引擎

`web_search` 是一个薄分发器：模型看到的工具名、参数、docstring 始终不变，背后的搜索引擎可换。所以换引擎**不会**影响 prompt、`RunConfig(enabled_tools=["web_search"])` 或权限规则。

引擎选择优先级：`provider=` 参数 > `AGENTICA_WEB_SEARCH` 环境变量 > 默认 `exa`。

**引擎不会根据 API key 自动推断** —— 你为别处设的 key 不应该悄悄改变 agent 的搜索行为。要换引擎就显式说。

| provider | 引擎 | API key | 说明 |
|----------|------|---------|------|
| `exa` | Exa（MCP） | 可选 `EXA_API_KEY` | 默认。不设 key 走共享免费池（有限流）；设了走自己的额度 |
| `baidu` | 百度 | 不需要 | 中文可用 |
| `duckduckgo` | DuckDuckGo | 不需要 | 直连 HTML 端点，无额外依赖 |
| `bocha` | 博查 | `BOCHA_API_KEY` | 中文效果好 |
| `serper` | Google/Serper | `SERPER_API_KEY` | |
| `zhipu` | 智谱 Web Search | `ZAI_API_KEY` | 4 档引擎，见下 |
| `mcp` | 任意 MCP 搜索服务 | 可选 | 见「自定义引擎」 |

指定了需要 key 的引擎却没提供 key 时会**直接报错**，而不是静默退回默认引擎 —— 否则你会以为在用 Bocha，实际在用 Exa。环境变量配错 / 缺 key 才会降级到 `exa` 并 warn。

```python
# SDK：显式指定
agent = Agent(tools=get_builtin_tools(web_search_provider="bocha"))

# SDK：直接构造
agent = Agent(tools=[BuiltinWebSearchTool(provider="exa"), ...])
```

```bash
# CLI + SDK 通用：环境变量
export AGENTICA_WEB_SEARCH=bocha
export BOCHA_API_KEY=sk-xxx
```

CLI 用户也可以写进 `~/.agentica/config.yaml` 的 `env:` 块，一次配置对所有会话生效：

```yaml
env:
  AGENTICA_WEB_SEARCH: bocha
  BOCHA_API_KEY: "sk-..."
```

#### 智谱的 4 档引擎

`zhipu` 是按次计费的，不同档位质量和价格都不同，用 `AGENTICA_ZHIPU_SEARCH_ENGINE` 选（默认 `search_pro`）：

| 引擎编码 | 特性 | 价格 | 结果带 link |
|----------|------|------|-------------|
| `search_std` | 基础版（智谱自研），满足日常查询，性价比最高 | 0.01 元/次 | 约 57% |
| `search_pro` | 高级版（智谱自研），多引擎协作，空结果率低、召回与准确率更高 | 0.03 元/次 | 约 60% |
| `search_pro_sogou` | 搜狗，覆盖腾讯生态（新闻/企鹅号）与知乎，百科、医疗等垂域权威性强 | 0.05 元/次 | 100% |
| `search_pro_quark` | 夸克，精准触达垂直内容 | 0.05 元/次 | 100% |

**留意 link 缺失**：实测智谱自研的两档（`search_std` / `search_pro`）约 40% 的查询会整批返回空 `link`（同一次查询要么全有要么全无），此时模型拿到标题和正文摘要但没有来源 URL —— 既没法引用，也没法再 `fetch_url` 深读。如果你的场景要求每条结果都可溯源，用 `search_pro_sogou` 或 `search_pro_quark`。

```bash
export AGENTICA_WEB_SEARCH=zhipu
export ZAI_API_KEY=sk-xxx
export AGENTICA_ZHIPU_SEARCH_ENGINE=search_pro_sogou   # 需要每条都有来源 URL 时
```

单独用这个工具时还能加域名白名单、时间范围和摘要长度：

```python
from agentica.tools.zhipu_web_search_tool import ZhipuWebSearchTool

tool = ZhipuWebSearchTool(
    search_engine="search_pro",
    search_recency_filter="oneWeek",   # oneDay / oneWeek / oneMonth / oneYear / noLimit
    search_domain_filter="www.sohu.com",
    content_size="high",              # medium（默认）/ high，high 摘要长度约翻倍
    user_id="tenant-000042",          # 6-128 字符，一个 key 服务多个终端用户时传
)
```

`max_results` 一定会被遵守：智谱的 `count` 只是建议值（`search_pro_sogou` 会向上取整到 10/20/30/40/50，其他档位在部分查询上也会超发），所以多出来的结果会在客户端截掉 —— 单条正文约 1000 字，要 3 条却收到 10 条会白烧几千 token 的 context。

#### 自定义引擎

三种方式，按「是否要写代码」和「CLI 能否用」区分。

**1. 自己的 MCP 搜索服务（零代码，CLI 可用）** —— 任何暴露搜索工具的 MCP 服务都能直接接入：

```bash
export AGENTICA_WEB_SEARCH=mcp
export AGENTICA_WEB_SEARCH_MCP_URL=https://my-bing-mcp.example.com/mcp
export AGENTICA_WEB_SEARCH_MCP_TOOL=bing_search
export AGENTICA_WEB_SEARCH_API_KEY=sk-xxx          # 可选，作为 Bearer header 发送
export AGENTICA_WEB_SEARCH_MCP_COUNT_ARG=numResults # 可选，条数参数名
```

查询参数名固定为 `query`。服务端若用别的名字，改用下面两种方式。

**2. 注册命名引擎（写代码，注册后 CLI/env 也能选）**：

```python
from agentica.tools.builtin import register_web_search_backend

register_web_search_backend(
    "bing",
    lambda api_key: MyBingTool(api_key=api_key),
    "bing_search",                # async (queries, max_results) -> str
    key_env="BING_API_KEY",
    key_required=True,
)
# 之后 provider="bing" 或 AGENTICA_WEB_SEARCH=bing 都可用
```

**3. 直接传函数（最灵活，仅 SDK）** —— 适合包装已有的 MCP client：

```python
async def my_search(queries, max_results=5) -> str:
    return await my_mcp_client.call_tool("bing_search", {"q": queries, "n": max_results})

agent = Agent(tools=[BuiltinWebSearchTool(search_fn=my_search), ...])
```

自定义引擎只需满足一个契约：**async `(queries: str | list[str], max_results: int) -> str`**。所有内置引擎都已统一到这个签名。

```python
from agentica import DeepAgent, OpenAIChat

# DeepAgent 自动包含以上所有工具
agent = DeepAgent(
    model=OpenAIChat(id="gpt-4o"),
    work_dir="./my-project",
)
```

### 搜索工具

```python
from agentica.tools.baidu_search_tool import BaiduSearchTool
from agentica.tools.duckduckgo_tool import DuckDuckGoTool
from agentica.tools.search_serper_tool import SearchSerperTool  # SERPER_API_KEY
from agentica.tools.search_exa_tool import SearchExaTool        # EXA_API_KEY

agent = Agent(tools=[DuckDuckGoTool()])
```

### 网页工具

```python
from agentica.tools.url_crawler_tool import UrlCrawlerTool
from agentica.tools.jina_tool import JinaTool          # Jina Reader API
from agentica.tools.browser_tool import BrowserTool    # playwright 浏览器自动化

agent = Agent(tools=[UrlCrawlerTool()])
```

### 代码与执行工具

```python
from agentica.tools.shell_tool import ShellTool        # Shell 命令执行
from agentica.tools.code_tool import CodeTool          # Python 代码沙箱执行

agent = Agent(tools=[ShellTool(timeout=30)])
```

### 知识与数据工具

```python
from agentica.tools.arxiv_tool import ArxivTool        # Arxiv 论文搜索
from agentica.tools.wikipedia_tool import WikipediaTool
from agentica.tools.yfinance_tool import YFinanceTool  # 金融数据
from agentica.tools.weather_tool import WeatherTool
from agentica.tools.sql_tool import SqlTool            # 数据库查询

agent = Agent(tools=[
    ArxivTool(max_results=5),
    YFinanceTool(),
])
```

### 多媒体工具

```python
from agentica.tools.dalle_tool import DalleTool              # DALL-E 图像生成
from agentica.tools.cogview_tool import CogViewTool          # 智谱 CogView
from agentica.tools.image_analysis_tool import ImageAnalysisTool
from agentica.tools.ocr_tool import OcrTool                  # 文字识别

agent = Agent(tools=[DalleTool(), ImageAnalysisTool()])
```

### Human-in-the-loop：`AskUserQuestionTool`

让 Agent 在执行过程中向用户提问：

```python
from agentica.tools.ask_user_question_tool import AskUserQuestionTool

def my_input_handler(prompt: str, options=None) -> str:
    """自定义输入处理（如 Web UI 弹窗）"""
    print(f"Agent asks: {prompt}")
    return input("> ")

agent = Agent(
    model=ZhipuAI(),
    tools=[AskUserQuestionTool(input_callback=my_input_handler)],
)
# Agent 需要确认时会调用 ask_user_question 工具：
# "确定要删除 production 数据库吗？(yes/no)"
```

### MCP 工具

通过 MCP 协议集成任意 MCP Server 的工具：

```python
from agentica.tools.mcp_tool import McpTool

# stdio 模式（本地进程）
async with McpTool("python mcp_server.py") as mcp:
    agent = Agent(model=ZhipuAI(), tools=[mcp])
    result = await agent.run("用 MCP 工具做计算")

# SSE 模式（远程服务）
async with McpTool(url="http://localhost:8080/sse") as mcp:
    agent = Agent(model=ZhipuAI(), tools=[mcp])
```

详见 [MCP 集成](../advanced/mcp.md)。

## 工具安全

### `ToolConfig` 配置工具行为

```python
from agentica.agent.config import ToolConfig

agent = Agent(
    model=ZhipuAI(),
    tools=[...],
    tool_config=ToolConfig(
        tool_call_limit=20,            # 单次 run 最多调用 20 次工具
    ),
)
```

### Guardrails（工具级守卫）

对工具的输入/输出添加安全检查：

```python
from agentica.guardrails.tool import ToolInputGuardrail, ToolOutputGuardrail

def check_no_rm_rf(tool_name: str, args: dict) -> bool:
    """阻止危险的 rm -rf 命令"""
    if tool_name == "execute" and "rm -rf" in args.get("command", ""):
        return False  # 拦截
    return True

agent = Agent(
    model=ZhipuAI(),
    tools=[ShellTool()],
    tool_input_guardrails=[ToolInputGuardrail(check_no_rm_rf)],
)
```

详见 [Guardrails](../advanced/guardrails.md)。

## 工具使用最佳实践

### 1. 清晰的 docstring 决定工具命中率

LLM 通过工具名称和描述决定何时调用。描述越精确，调用越准确：

```python
# 差：描述模糊
def process(data: str) -> str:
    """处理数据"""
    ...

# 好：描述精确，包含适用场景
def parse_csv_to_json(csv_content: str, delimiter: str = ",") -> str:
    """将 CSV 格式文本转换为 JSON 数组

    适用于：数据格式转换、从 CSV 文件提取结构化数据
    Args:
        csv_content: CSV 格式的文本内容（含表头行）
        delimiter: 分隔符，默认为逗号，Excel 导出常用分号 ";"
    """
    ...
```

### 2. 使用 `Literal` 类型约束枚举参数

```python
from typing import Literal

def export_report(
    format: Literal["pdf", "html", "markdown"],
    include_charts: bool = True,
) -> str:
    """导出分析报告"""
    ...
# LLM 收到 enum: ["pdf", "html", "markdown"]，不会乱传
```

### 3. 返回结构化错误信息

```python
def query_database(sql: str) -> str:
    """执行 SQL 查询"""
    try:
        result = db.execute(sql)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        # 返回描述性错误，让 LLM 能理解并修正
        return f"Error: {type(e).__name__}: {e}. Hint: Check column names with list_tables() first."
```

### 4. 控制工具数量

```python
# 推荐：3-7 个相关工具（LLM 上下文有限）
agent = Agent(tools=[search_tool, crawl_tool, analyze_tool])

# 避免：超过 10 个工具（LLM 选择准确率下降）
```

## 下一步

- [Agent 概念](agent.md) -- Agent 如何调度工具
- [DeepAgent](agent.md#deepagent) -- 内置全套工具的高级 Agent
- [Guardrails](../advanced/guardrails.md) -- 工具级安全守卫
- [MCP 集成](../advanced/mcp.md) -- MCP 协议工具集成
- [Hooks](../advanced/hooks.md) -- 工具调用生命周期钩子
