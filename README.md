[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# Agentica: Build AI Agents
[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![MseeP.ai](https://img.shields.io/badge/mseep.ai-agentica-blue)](https://mseep.ai/app/shibing624-agentica)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#社区与支持)

## 目录

- [简介](#简介)
- [特性](#特性)
- [系统架构](#系统架构)
- [安装](#安装)
- [快速开始](#快速开始)
- [功能示例](#功能示例)
- [应用场景](#应用场景)
- [与同类工具对比](#与同类工具对比)
- [常见问题](#常见问题)
- [社区与支持](#社区与支持)
- [引用](#引用)
- [许可证](#许可证)

## 简介

**Agentica** 是一个轻量级、模块化的 AI Agent 框架，专注于构建智能、具备反思能力、可协作的多模态 AI Agent。它提供了丰富的工具集成、多模型支持和灵活的工作流编排能力，让开发者能够轻松构建复杂的 AI 应用。

无论是构建个人助手、知识管理系统、自动化工作流，还是多 Agent 协作系统，Agentica 都能提供强大而灵活的支持。通过简洁的 API 和丰富的示例，即使是 AI 开发新手也能快速上手，构建出功能强大的智能应用。

**为什么选择 Agentica?**

- **简单易用**：简洁直观的 API 设计，降低学习门槛
- **功能全面**：从单 Agent 到多 Agent 协作，从简单对话到复杂工作流，一站式解决
- **中文优化**：为中文用户打造，文档和示例丰富
- **多模型支持**：支持国内外多种大模型，灵活切换
- **开箱即用**：丰富的内置工具和示例，快速实现各类应用场景

## 版本更新

- **[2025/06/19] v1.0.10 版本**：支持思考过程流式输出，适配所有推理模型，详见 [Release-v1.0.10](https://github.com/shibing624/agentica/releases/tag/1.0.10)
- **[2025/05/19] v1.0.6 版本**：新增了 MCP 的 StreamableHttp 支持，兼容 StreamableHttp/SSE/Stdio 三种 MCP Server，详见 [Release-v1.0.6](https://github.com/shibing624/agentica/releases/tag/1.0.6)
- **[2025/04/21] v1.0.0 版本**：支持了 MCP 的工具调用，兼容 SSE/Stdio 的 MCP Server，详见 [Release-v1.0.0](https://github.com/shibing624/agentica/releases/tag/1.0.0)
- **[2024/12/29] v0.2.3 版本**：支持了 ZhipuAI 的 api 调用，包括免费模型和工具使用，详见 [Release-v0.2.3](https://github.com/shibing624/agentica/releases/tag/0.2.3)
- **[2024/12/25] v0.2.0 版本**：支持了多模态模型，输入可以是文本、图片、音频、视频，并升级 Assistant 为 Agent，Workflow 支持拆解并实现复杂任务，详见 [Release-v0.2.0](https://github.com/shibing624/agentica/releases/tag/0.2.0)
- **[2024/07/02] v0.1.0 版本**：实现了基于 LLM 的 Assistant，可以快速用 function call 搭建大语言模型助手，详见 [Release-v0.1.0](https://github.com/shibing624/agentica/releases/tag/0.1.0)

[查看所有版本](https://github.com/shibing624/agentica/releases)

## 特性
`Agentica`是一个用于构建Agent的工具，具有以下功能：

- **Agent编排**：通过简单代码快速编排Agent，支持 Reflection(反思)、Plan and Solve(计划并执行)、RAG 等功能
- **工具调用**：支持自定义工具OpenAI的function call，支持MCP Server 的工具调用
- **LLM 集成**：支持OpenAI、Azure、Deepseek、Moonshot、Anthropic、ZhipuAI、Ollama、Together等多方大模型厂商的API
- **记忆功能**：支持短期记忆和长期记忆功能
- **Multi-Agent 协作**：支持多Agent和任务委托(Team)的团队协作
- **Workflow 工作流**：拆解复杂任务为多个子工具任务，基于工作流自动化串行逐步完成任务
- **自我进化 Agent**：具有反思和增强记忆能力的自我进化 Agent
- **Web UI**：兼容 ChatPilot，可以基于 Web 页面交互，支持主流的 open-webui、streamlit、gradio 等前端交互框架

## 系统架构

<div align="center">
    <img src="https://github.com/shibing624/agentica/blob/main/docs/agentica_architecture.png" alt="Agentica Architecture" width="800"/>
</div>

Agentica 采用模块化设计，主要包括以下核心组件：

1. **Agent Core**：核心控制模块，负责 Agent 的创建和管理
   - `agent.py`：Agent 的核心实现
   - `agent_session.py`：Agent 会话管理
   - `memory.py`：记忆管理实现

2. **Model Integration**：模型接入层，支持多种 LLM 模型接口
   - 支持 OpenAI、Azure、Moonshot、ZhipuAI 等多种模型
   - 统一的模型调用接口，便于切换和比较

3. **Tools System**：工具调用系统，提供丰富的工具调用能力
   - Web 搜索、OCR、图像生成、Shell 命令等内置工具
   - 自定义工具支持，轻松扩展功能

4. **Memory Management**：记忆管理，实现短期和长期记忆功能
   - 短期记忆：保存对话历史
   - 长期记忆：持久化存储重要信息

5. **Knowledge & RAG**：知识库与检索增强生成
   - 文档解析和向量化
   - 混合检索策略
   - 结果重排序

6. **Multi-Agent Collaboration**：多 Agent 协作，实现团队协作和任务委托
   - Team 模式：多角色协作
   - 任务委托：分配和监督子任务

7. **Workflow Orchestration**：工作流编排，支持复杂任务的拆解和执行
   - 任务分解：将复杂任务拆分为子任务
   - 自动化执行：按顺序或并行执行子任务
   - 结果汇总：整合子任务结果

8. **Storage**：存储系统，支持 SQL 和向量数据库
   - SQL：结构化数据存储
   - 向量数据库：高效相似度搜索

9. **MCP**：Model Context Protocol 支持
   - 支持 StreamableHttp/SSE/Stdio 三种 MCP Server
   - 标准化模型交互协议
   

## 安装

### 使用 pip 安装

```bash
pip install -U agentica
```

### 从源码安装
```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install .
```

## 快速开始

### 环境准备
1. 复制[.env.example](https://github.com/shibing624/agentica/blob/main/.env.example)文件为`~/.agentica/.env`，并填写LLM api key(选填OPENAI_API_KEY、ZHIPUAI_API_KEY、MOONSHOT_API_KEY 等任一个)。或者使用`export`命令设置环境变量：
    
    ```shell
    export MOONSHOT_API_KEY=your_api_key
    export SERPER_API_KEY=your_serper_api_key
    ```

2. 使用 agentica 构建 Agent 并执行：

### 基础示例：天气查询

```python
from agentica import Agent, Moonshot, WeatherTool

m = Agent(model=Moonshot(), tools=[WeatherTool()], add_datetime_to_instructions=True)
m.print_response("明天北京天气咋样")
```

output:
```markdown
明天北京的天气预报如下：

- 早晨：晴朗，气温约18°C，风速较小，约为3 km/h。
- 中午：晴朗，气温升至23°C，风速6-7 km/h。
- 傍晚：晴朗，气温略降至21°C，风速较大，为35-44 km/h。
- 夜晚：晴朗转晴，气温下降至15°C，风速32-39 km/h。

全天无降水，能见度良好。请注意傍晚时分的风速较大，外出时需注意安全。
```

### 高级示例：自定义工具

创建和使用自定义工具：

```python
from agentica import Agent, OpenAIChat, Tool


class CalculatorTool(Tool):
   def __init__(self):
      super().__init__(name="calculator")
      self.register(self.calc)

   def calc(self, expression: str):
      try:
         result = eval(expression)
         return str(result)
      except Exception as e:
         return str(e)


agent = Agent(
   model=OpenAIChat(id="gpt-4o"),
   tools=[CalculatorTool()],
)

agent.print_response("计算 (123 + 456) * 789 的结果")
```

## 功能示例

Agentica 提供了丰富的功能示例，帮助您快速上手：

| 示例                                                                                                                                                    | 描述                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [examples/01_llm_demo.py](https://github.com/shibing624/agentica/blob/main/examples/01_llm_demo.py)                                                   | LLM问答Demo                                                                                                                         |
| [examples/02_user_prompt_demo.py](https://github.com/shibing624/agentica/blob/main/examples/02_user_prompt_demo.py)                                   | 自定义用户prompt的Demo                                                                                                                  |
| [examples/03_user_messages_demo.py](https://github.com/shibing624/agentica/blob/main/examples/03_user_messages_demo.py)                               | 自定义输入用户消息的Demo                                                                                                                    |
| [examples/04_memory_demo.py](https://github.com/shibing624/agentica/blob/main/examples/04_memory_demo.py)                                             | Agent的记忆Demo                                                                                                                      |
| [examples/05_response_model_demo.py](https://github.com/shibing624/agentica/blob/main/examples/05_response_model_demo.py)                             | 按指定格式（pydantic的BaseModel）回复的Demo                                                                                                  |
| [examples/06_calc_with_csv_file_demo.py](https://github.com/shibing624/agentica/blob/main/examples/06_calc_with_csv_file_demo.py)                     | LLM加载CSV文件，并执行计算来回答的Demo                                                                                                          |
| [examples/07_create_image_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/07_create_image_tool_demo.py)                       | 实现了创建图像工具的Demo                                                                                                                    |
| [examples/08_ocr_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/08_ocr_tool_demo.py)                                         | 实现了OCR工具的Demo                                                                                                                     |
| [examples/09_remove_image_background_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/09_remove_image_background_tool_demo.py) | 实现了自动去除图片背景功能，包括自动通过pip安装库，调用库实现去除图片背景                                                                                            |
| [examples/10_vision_demo.py](https://github.com/shibing624/agentica/blob/main/examples/10_vision_demo.py)                                             | 视觉理解Demo                                                                                                                          |
| [examples/11_web_search_openai_demo.py](https://github.com/shibing624/agentica/blob/main/examples/11_web_search_openai_demo.py)                       | 基于OpenAI的function call做网页搜索Demo                                                                                                   |
| [examples/20_advanced_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/20_advanced_rag_demo.py)                                 | 实现了高级版RAG，基于PDF文档回答问题，新增功能：pdf文件解析、query改写，字面+语义多路混合召回，召回排序（rerank）                                                               |
| [examples/31_team_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/31_team_news_article_demo.py)                       | Team实现：写新闻稿的team协作，multi-role实现，委托不用角色完成各自任务：研究员检索分析文章，撰写员根据排版写文章，汇总多角色成果输出结果                                                     |
| [examples/33_self_evolving_agent_demo.py](https://github.com/shibing624/agentica/blob/main/examples/33_self_evolving_agent_demo.py)                   | 实现了自我进化Agent的Demo                                                                                                                 |
| [examples/34_llm_os_demo.py](https://github.com/shibing624/agentica/blob/main/examples/34_llm_os_demo.py)                                             | 实现了LLM OS的初步设计，基于LLM设计操作系统，可以通过LLM调用RAG、代码执行器、Shell等工具，并协同代码解释器、研究助手、投资助手等来解决问题。                                                  |
| [examples/43_minimax_mcp_demo.py](https://github.com/shibing624/agentica/blob/main/examples/43_minimax_mcp_demo.py)                                   | Minimax语音生成调用的Demo                                                                                                                |

[查看更多示例](https://github.com/shibing624/ChatPilot/blob/main/examples/)


## 命令行模式（CLI）

支持终端命令行快速搭建 Agentica Agent，使用简单的命令行参数即可完成。

code: [cli.py](https://github.com/shibing624/agentica/blob/main/agentica/cli.py)

```
> agentica -h                                    
usage: cli.py [-h] [--query QUERY]
              [--model_provider {openai,azure,moonshot,zhipuai,deepseek,yi}]
              [--model_name MODEL_NAME] [--api_base API_BASE]
              [--api_key API_KEY] [--max_tokens MAX_TOKENS]
              [--temperature TEMPERATURE] [--verbose VERBOSE]
              [--tools [{search_serper,file_tool,shell_tool,yfinance_tool,web_search_pro,cogview,cogvideo,jina,wikipedia} ...]]

CLI for agentica

options:
  -h, --help            show this help message and exit
  --query QUERY         Question to ask the LLM
  --model_provider {openai,azure,moonshot,zhipuai,deepseek,yi}
                        LLM model provider
  --model_name MODEL_NAME
                        LLM model name to use, can be
                        gpt-4o/glm-4-flash/deepseek-chat/yi-lightning/...
  --api_base API_BASE   API base URL for the LLM
  --api_key API_KEY     API key for the LLM
  --max_tokens MAX_TOKENS
                        Maximum number of tokens for the LLM
  --temperature TEMPERATURE
                        Temperature for the LLM
  --verbose VERBOSE     enable verbose mode
  --tools [{search_serper,file_tool,shell_tool,yfinance_tool,web_search_pro,cogview,cogvideo,jina,wikipedia} ...]
                        Tools to enable
```

run：

```shell
pip install agentica -U
# 单次调用，填入`--query`参数
agentica --query "下一届奥运会在哪里举办" --model_provider zhipuai --model_name glm-4-flash --tools web_search_pro
# 多次调用，多轮对话，不填`--query`参数
agentica --model_provider zhipuai --model_name glm-4-flash --tools web_search_pro cogview --verbose 1
```

output:
```shell
2024-12-30 21:59:15,000 - agentica - INFO - Agentica CLI
>>> 帮我画个大象在月球上的图

> 我帮你画了一张大象在月球上的图，它看起来既滑稽又可爱。大象穿着宇航服，站在月球表面，背景是广阔的星空和地球。这张图色彩明亮，细节丰富，具有卡通风格。你可以点击下面的链接查看和下载这张图片：

![大象在月球上的图](https://aigc-files.bigmodel.cn/api/cogview/20241230215915cfa22f46425e4cb0_0.png)
```

## Web UI

[shibing624/ChatPilot](https://github.com/shibing624/ChatPilot) 兼容`agentica`，可以通过Web UI进行交互。

Web Demo: https://chat.mulanai.com

<div align="center">
    <img src="https://github.com/shibing624/ChatPilot/blob/main/docs/shot.png" width="800" />
</div>

### 部署 Web UI

```bash
git clone https://github.com/shibing624/ChatPilot.git
cd ChatPilot
pip install -r requirements.txt

cp .env.example .env

bash start.sh
```


## 应用场景

Agentica 适用于多种现实应用场景：

### 1. 智能助手与对话系统
- **个人助理**：构建具有记忆能力和工具使用能力的个性化助手
- **客服机器人**：创建能够查询知识库、处理复杂问题的客服系统
- **专业领域助手**：如法律助手、医疗咨询、教育辅导等专业领域的智能助手

### 2. 知识管理与检索增强
- **智能文档分析**：对 PDF、文本等文档进行深度理解和问答
- **企业知识库**：构建企业内部知识管理系统，实现智能检索和问答
- **研究辅助工具**：自动检索、汇总和分析研究资料，生成研究报告

### 3. 自动化工作流
- **内容创作**：自动化撰写文章、教程、小说等创意内容
- **数据分析报告**：自动收集数据、分析并生成可视化报告
- **投资研究**：收集股票信息、分析市场趋势、生成投资建议

### 4. 多 Agent 协作系统
- **团队协作模拟**：模拟不同角色的团队协作，如辩论、讨论、决策
- **复杂任务分解**：将大型任务分解为子任务，由不同专业 Agent 协作完成
- **业务流程自动化**：模拟企业内不同部门协作，实现业务流程自动化

### 5. 多模态交互应用
- **图像理解与生成**：结合视觉理解和图像生成能力的应用
- **语音交互系统**：支持语音输入输出的对话系统
- **多媒体内容分析**：分析视频、音频、图像等多媒体内容

## 与同类工具对比

### Agentica vs LangChain

| 特性 | Agentica | LangChain |
|------|----------|-----------|
| 核心设计 | 以 Agent 为中心，强调 Agent 的自主性和能力 | 以 Chain 为中心，强调组件的链式调用 |
| 记忆管理 | 内置短期和长期记忆机制 | 需要额外配置记忆组件 |
| 多 Agent 协作 | 原生支持 Team 和任务委托 | 需要额外配置和开发 |
| 工作流编排 | 内置工作流系统，支持任务分解和执行 | 需要使用额外的编排工具 |
| 中文支持 | 优秀的中文支持和文档 | 主要面向英文用户 |
| 学习曲线 | 相对简单，API 设计直观 | 较为复杂，组件众多 |
| 社区生态 | 较小但增长迅速 | 庞大且活跃 |

### Agentica vs AutoGen

| 特性 | Agentica | AutoGen |
|------|----------|---------|
| 核心设计 | 全面的 Agent 框架，支持多种功能 | 专注于多 Agent 对话和协作 |
| 工具集成 | 丰富的内置工具和自定义工具支持 | 较为基础的工具支持 |
| RAG 能力 | 内置知识库和 RAG 功能 | 需要额外集成 |
| 模型支持 | 支持多种国内外大模型 | 主要支持 OpenAI 等国际模型 |
| 部署便捷性 | 支持 CLI 和 Web UI 多种部署方式 | 主要面向开发者的 API |
| MCP 协议 | 支持 Model Context Protocol | 不支持 |

### Agentica vs CrewAI

| 特性 | Agentica | CrewAI |
|------|----------|--------|
| 核心设计 | 全面的 Agent 框架 | 专注于 Agent 团队协作 |
| 单 Agent 能力 | 完整的单 Agent 功能集 | 相对简单的单 Agent 功能 |
| 工作流管理 | 灵活的工作流定义和执行 | 基于任务的工作流 |
| 记忆系统 | 复杂的记忆管理机制 | 基础的记忆功能 |
| 多模态支持 | 支持文本、图像、音频等多模态 | 主要支持文本 |
| 中文支持 | 良好的中文支持 | 有限的中文支持 |


## 常见问题

### 1. 如何选择合适的 LLM 模型？

Agentica 支持多种 LLM 模型，选择取决于您的具体需求：
- 对于复杂任务和高质量输出，推荐使用 OpenAI 的 GPT-4o 或 Moonshot 的 glm-4-flash
- 对于简单任务和快速响应，可以使用 OpenAI 的 GPT-3.5-turbo 或 Moonshot 的 moonshot-v1-8k
- 对于本地部署，可以使用 Ollama 提供的开源模型

### 2. 如何自定义工具？

Agentica 提供了灵活的工具自定义机制，您可以通过继承 `Toolkit` 类来创建自己的工具：

```python
from agentica.tools.base import Tool
import json


class MyCustomTool(Tool):
   def __init(self):
      name = "my_custom_tool"
      super().__init__(name=name)
      self.register(self.run)

   def run(self, **kwargs):
      # 实现工具逻辑
      return json.dumps({"result": "工具执行结果"})
```

### 3. 如何处理大型文档？

对于大型文档处理，建议使用 Agentica 的 RAG 功能：
1. 使用 `examples/20_advanced_rag_demo.py` 作为参考
2. 将文档分割成适当大小的块
3. 使用向量数据库存储文档嵌入
4. 结合检索和生成能力回答问题

### 4. 如何实现多 Agent 协作？

Agentica 提供了两种多 Agent 协作方式：
1. **Team 模式**：使用 `examples/31_team_news_article_demo.py` 作为参考，创建不同角色的 Agent 团队
2. **Workflow 模式**：使用 `examples/36_workflow_news_article_demo.py` 作为参考，将任务分解为多个步骤，由不同 Agent 执行

### 5. 如何部署到生产环境？

Agentica 应用可以通过多种方式部署到生产环境：
1. 作为 API 服务：使用 FastAPI 或 Flask 封装 Agentica 功能
2. 作为 Web 应用：使用 ChatPilot 提供 Web 界面
3. 作为命令行工具：使用内置的 CLI 功能

## 社区与支持

- Issue(建议)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我： 加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="https://github.com/shibing624/agentica/blob/main/docs/wechat.jpeg" width="200" />

## 引用

如果你在研究中使用了 Agentica，请按如下格式引用：

APA:

```
Xu, M. agentica: A Human-Centric Framework for Large Language Model Agent Workflows (Version 0.0.2) [Computer software]. https://github.com/shibing624/agentica
```

BibTeX:

```bibtex
@misc{agentica,
  author = {Ming Xu},
  title = {Agentica: Effortlessly Build Intelligent, Reflective, and Collaborative Multimodal AI Agents},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/shibing624/agentica}},
}
```

## 许可证

授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加`agentica`的链接和授权协议。
## 贡献指南

我们欢迎各种形式的贡献，包括但不限于：

- 提交问题和功能需求
- 提交代码修复和新功能
- 改进文档和示例
- 分享使用案例和最佳实践

请查看 [贡献指南](CONTRIBUTING.md) 了解更多详情。

## 感谢

- [langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- [simonmesmith/agentflow](https://github.com/simonmesmith/agentflow)
- [phidatahq/phidata](https://github.com/phidatahq/phidata)


Thanks for their great work!
