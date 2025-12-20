# Agentica 工具使用指南

> 40+ 内置工具的详细使用说明和示例

## 目录

- [工具基础](#工具基础)
- [搜索工具](#搜索工具)
- [代码执行工具](#代码执行工具)
- [文件操作工具](#文件操作工具)
- [网页工具](#网页工具)
- [知识工具](#知识工具)
- [多媒体工具](#多媒体工具)
- [其他工具](#其他工具)
- [自定义工具](#自定义工具)

---

## 工具基础

### 添加工具到 Agent

```python
from agentica import Agent, OpenAIChat
from agentica.tools import DuckDuckGoTool, CalculatorTool

agent = Agent(
    model=OpenAIChat(),
    tools=[
        DuckDuckGoTool(),
        CalculatorTool(),
    ],
)
```

### 使用函数作为工具

```python
def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

agent = Agent(tools=[get_current_time])
```

---

## 搜索工具

### DuckDuckGoTool

免费的网页搜索工具。

```python
from agentica.tools import DuckDuckGoTool

tool = DuckDuckGoTool(
    search=True,      # 启用搜索
    news=True,        # 启用新闻搜索
    max_results=5,    # 最大结果数
)

agent = Agent(tools=[tool])
agent.print_response("搜索最新的 AI 新闻")
```

### BaiduSearchTool

百度搜索工具。

```python
from agentica.tools import BaiduSearchTool

tool = BaiduSearchTool(max_results=10)
agent = Agent(tools=[tool])
```

### SerperTool

Google 搜索 API（需要 API Key）。

```python
from agentica.tools import SerperTool

tool = SerperTool(
    api_key="xxx",  # 或设置 SERPER_API_KEY 环境变量
    search_type="search",  # search, news, images
)
```

### ExaTool

Exa AI 搜索（语义搜索）。

```python
from agentica.tools import ExaTool

tool = ExaTool(
    api_key="xxx",  # 或设置 EXA_API_KEY 环境变量
    num_results=5,
)
```

### BochaSearchTool

博查搜索工具。

```python
from agentica.tools import BochaSearchTool

tool = BochaSearchTool(api_key="xxx")
```

---

## 代码执行工具

### RunPythonCodeTool

执行 Python 代码。

```python
from agentica.tools import RunPythonCodeTool

tool = RunPythonCodeTool(
    save_and_run=True,     # 保存并运行
    pip_install=True,      # 允许 pip 安装
    run_code=True,         # 允许运行代码
    list_files=True,       # 允许列出文件
    read_files=True,       # 允许读取文件
    safe_globals=None,     # 自定义全局变量
    base_dir="./work",     # 工作目录
)

agent = Agent(
    tools=[tool],
    instructions=["你是 Python 专家，可以编写和执行代码"],
)
agent.print_response("计算斐波那契数列前 10 项")
```

### ShellTool

执行 Shell 命令。

```python
from agentica.tools import ShellTool

tool = ShellTool(
    allowed_commands=["ls", "cat", "grep"],  # 允许的命令
    timeout=30,  # 超时（秒）
)

agent = Agent(tools=[tool])
agent.print_response("列出当前目录的文件")
```

### RunNbCodeTool

执行 Jupyter Notebook 代码。

```python
from agentica.tools import RunNbCodeTool

tool = RunNbCodeTool(
    kernel_name="python3",
    timeout=60,
)
```

---

## 文件操作工具

### FileTool

文件读写操作。

```python
from agentica.tools import FileTool

tool = FileTool(
    base_dir="./workspace",
    read_files=True,
    save_files=True,
    list_files=True,
)

agent = Agent(tools=[tool])
agent.print_response("读取 config.json 文件内容")
```

### EditTool

文件编辑工具（支持查看、创建、替换）。

```python
from agentica.tools import EditTool

tool = EditTool(base_dir="./project")

agent = Agent(tools=[tool])
agent.print_response("在 main.py 中添加一个 hello 函数")
```

**支持的操作:**
- `view`: 查看文件内容
- `create`: 创建新文件
- `str_replace`: 字符串替换
- `insert`: 插入内容

### WorkspaceTool

工作区管理工具。

```python
from agentica.tools import WorkspaceTool

tool = WorkspaceTool(
    workspace_dir="./workspace",
    read_files=True,
    write_files=True,
    list_files=True,
    search_files=True,
)
```

---

## 网页工具

### UrlCrawlerTool

URL 内容抓取。

```python
from agentica.tools import UrlCrawlerTool

tool = UrlCrawlerTool(
    max_length=5000,  # 最大内容长度
    timeout=10,       # 超时（秒）
)

agent = Agent(tools=[tool])
agent.print_response("获取 https://example.com 的内容")
```

### JinaTool

使用 Jina AI Reader 抓取网页。

```python
from agentica.tools import JinaTool

tool = JinaTool(
    api_key="xxx",  # 可选
    max_content_length=10000,
)

agent = Agent(tools=[tool])
agent.print_response("读取这篇文章: https://example.com/article")
```

### BrowserTool

浏览器自动化工具（基于 Playwright）。

```python
from agentica.tools import BrowserTool

tool = BrowserTool(
    headless=True,
    timeout=30000,
)

agent = Agent(tools=[tool])
agent.print_response("打开百度搜索 Python 教程")
```

**支持的操作:**
- 打开网页
- 点击元素
- 输入文本
- 截图
- 提取内容

### NewspaperTool

新闻文章提取。

```python
from agentica.tools import NewspaperTool

tool = NewspaperTool()
agent = Agent(tools=[tool])
agent.print_response("提取这篇新闻的内容: https://news.example.com/article")
```

---

## 知识工具

### ArxivTool

arXiv 论文搜索。

```python
from agentica.tools import ArxivTool

tool = ArxivTool(max_results=5)

agent = Agent(tools=[tool])
agent.print_response("搜索关于 Transformer 的最新论文")
```

### WikipediaTool

Wikipedia 搜索。

```python
from agentica.tools import WikipediaTool

tool = WikipediaTool(language="zh")  # 中文

agent = Agent(tools=[tool])
agent.print_response("查询爱因斯坦的生平")
```

### DblpTool

DBLP 计算机科学论文搜索。

```python
from agentica.tools import DblpTool

tool = DblpTool(max_results=10)
agent = Agent(tools=[tool])
```

### HackerNewsTool

Hacker News 内容获取。

```python
from agentica.tools import HackerNewsTool

tool = HackerNewsTool(
    get_top_stories=True,
    get_user_details=True,
)
```

---

## 多媒体工具

### DalleTool

DALL-E 图像生成。

```python
from agentica.tools import DalleTool

tool = DalleTool(
    model="dall-e-3",
    size="1024x1024",
    quality="standard",  # standard, hd
)

agent = Agent(tools=[tool])
agent.print_response("生成一张日落海滩的图片")
```

### CogViewTool

智谱 CogView 图像生成。

```python
from agentica.tools import CogViewTool

tool = CogViewTool(
    api_key="xxx",  # 或设置 ZHIPUAI_API_KEY
    model="cogview-3",
)
```

### CogVideoTool

智谱 CogVideo 视频生成。

```python
from agentica.tools import CogVideoTool

tool = CogVideoTool(
    api_key="xxx",
    model="cogvideox",
)
```

### ImageAnalysisTool

图像分析工具。

```python
from agentica.tools import ImageAnalysisTool

tool = ImageAnalysisTool(model=OpenAIChat(model="gpt-4o"))

agent = Agent(tools=[tool])
agent.print_response("分析这张图片: ./image.jpg")
```

### VideoAnalysisTool

视频分析工具。

```python
from agentica.tools import VideoAnalysisTool

tool = VideoAnalysisTool(
    model=OpenAIChat(model="gpt-4o"),
    extract_frames=True,
    frames_per_second=1,
)
```

### VideoDownloadTool

视频下载工具（基于 yt-dlp）。

```python
from agentica.tools import VideoDownloadTool

tool = VideoDownloadTool(
    output_dir="./videos",
    format="mp4",
)
```

### OcrTool

OCR 文字识别。

```python
from agentica.tools import OcrTool

tool = OcrTool()
agent = Agent(tools=[tool])
agent.print_response("识别这张图片中的文字: ./document.png")
```

---

## 其他工具

### CalculatorTool

数学计算。

```python
from agentica.tools import CalculatorTool

tool = CalculatorTool()
agent = Agent(tools=[tool])
agent.print_response("计算 (123 + 456) * 789")
```

### WeatherTool

天气查询。

```python
from agentica.tools import WeatherTool

tool = WeatherTool(api_key="xxx")  # OpenWeatherMap API
agent = Agent(tools=[tool])
agent.print_response("北京今天天气怎么样？")
```

### YFinanceTool

金融数据查询。

```python
from agentica.tools import YFinanceTool

tool = YFinanceTool(
    stock_price=True,
    stock_fundamentals=True,
    analyst_recommendations=True,
)

agent = Agent(tools=[tool])
agent.print_response("查询苹果公司的股票信息")
```

### SqlTool

SQL 数据库查询。

```python
from agentica.tools import SqlTool

tool = SqlTool(
    db_url="sqlite:///data.db",
    tables=["users", "orders"],
)

agent = Agent(tools=[tool])
agent.print_response("查询所有用户的订单数量")
```

### ResendTool

邮件发送（Resend API）。

```python
from agentica.tools import ResendTool

tool = ResendTool(
    api_key="xxx",
    from_email="noreply@example.com",
)
```

### AirflowTool

Airflow 工作流集成。

```python
from agentica.tools import AirflowTool

tool = AirflowTool(
    airflow_url="http://localhost:8080",
    username="admin",
    password="admin",
)
```

### VolcTtsTool

火山引擎 TTS 语音合成。

```python
from agentica.tools import VolcTtsTool

tool = VolcTtsTool(
    app_id="xxx",
    access_token="xxx",
)
```

### ApifyTool

Apify 网页爬虫。

```python
from agentica.tools import ApifyTool

tool = ApifyTool(api_key="xxx")
```

---

## 自定义工具

### 方式 1: 使用函数

最简单的方式，直接定义函数。

```python
def search_database(query: str, limit: int = 10) -> str:
    """搜索数据库
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量限制
    
    Returns:
        搜索结果
    """
    # 实现搜索逻辑
    results = db.search(query, limit=limit)
    return str(results)

agent = Agent(tools=[search_database])
```

### 方式 2: 继承 Tool 类

更灵活的方式，支持多个函数。

```python
from agentica import Tool

class MyDatabaseTool(Tool):
    def __init__(self, db_connection):
        super().__init__(name="database")
        self.db = db_connection
        
        # 注册函数
        self.register(self.search)
        self.register(self.insert)
        self.register(self.delete)
    
    def search(self, query: str, limit: int = 10) -> str:
        """搜索数据库"""
        return str(self.db.search(query, limit))
    
    def insert(self, data: dict) -> str:
        """插入数据"""
        self.db.insert(data)
        return "插入成功"
    
    def delete(self, id: str) -> str:
        """删除数据"""
        self.db.delete(id)
        return "删除成功"

tool = MyDatabaseTool(db_connection)
agent = Agent(tools=[tool])
```

### 方式 3: 使用 Function 类

精细控制函数定义。

```python
from agentica import Function

def my_function(x: int, y: int) -> int:
    return x + y

func = Function(
    name="add_numbers",
    description="将两个数字相加",
    parameters={
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "第一个数"},
            "y": {"type": "integer", "description": "第二个数"},
        },
        "required": ["x", "y"],
    },
    entrypoint=my_function,
    strict=True,
    show_result=True,
)

agent = Agent(tools=[func])
```

### 工具钩子

添加前置和后置钩子。

```python
def pre_hook(function_call):
    print(f"即将调用: {function_call.function.name}")
    print(f"参数: {function_call.arguments}")

def post_hook(function_call):
    print(f"调用完成: {function_call.function.name}")
    print(f"结果: {function_call.result}")

func = Function.from_callable(
    my_function,
    pre_hook=pre_hook,
    post_hook=post_hook,
)
```

### 停止执行

工具可以控制 Agent 停止执行。

```python
from agentica import Function

def confirm_action(action: str) -> str:
    """确认操作"""
    return f"请确认是否执行: {action}"

func = Function.from_callable(
    confirm_action,
    stop_after_tool_call=True,  # 调用后停止
)
```

### 工具异常

抛出异常控制执行流程。

```python
from agentica.tools import ToolCallException, StopAgentRun

def risky_operation(data: str) -> str:
    if not validate(data):
        raise ToolCallException(
            user_message="数据验证失败，请检查输入",
            stop_execution=True,
        )
    return process(data)

def emergency_stop() -> str:
    raise StopAgentRun(
        user_message="紧急停止",
        agent_message="操作已取消",
    )
```

---

## 工具组合示例

### 研究助手

```python
from agentica import Agent, OpenAIChat
from agentica.tools import (
    DuckDuckGoTool,
    ArxivTool,
    WikipediaTool,
    UrlCrawlerTool,
)

agent = Agent(
    name="Research Assistant",
    model=OpenAIChat(model="gpt-4o"),
    tools=[
        DuckDuckGoTool(),
        ArxivTool(),
        WikipediaTool(),
        UrlCrawlerTool(),
    ],
    instructions=[
        "你是一个研究助手",
        "使用多种工具收集信息",
        "综合分析后给出结论",
    ],
)

agent.print_response("研究量子计算的最新进展")
```

### 数据分析师

```python
from agentica import Agent, OpenAIChat
from agentica.tools import (
    RunPythonCodeTool,
    FileTool,
    CalculatorTool,
)

agent = Agent(
    name="Data Analyst",
    model=OpenAIChat(model="gpt-4o"),
    tools=[
        RunPythonCodeTool(pip_install=True),
        FileTool(base_dir="./data"),
        CalculatorTool(),
    ],
    instructions=[
        "你是数据分析专家",
        "可以读取数据文件并用 Python 分析",
        "生成可视化图表",
    ],
)

agent.print_response("分析 sales.csv 中的销售趋势")
```

### 内容创作者

```python
from agentica import Agent, OpenAIChat
from agentica.tools import (
    DuckDuckGoTool,
    DalleTool,
    FileTool,
)

agent = Agent(
    name="Content Creator",
    model=OpenAIChat(model="gpt-4o"),
    tools=[
        DuckDuckGoTool(),
        DalleTool(),
        FileTool(base_dir="./content"),
    ],
    instructions=[
        "你是内容创作专家",
        "可以搜索资料、生成图片、保存文件",
    ],
)

agent.print_response("创作一篇关于 AI 的文章，配上插图")
```

---

*文档最后更新: 2025-12-20*
