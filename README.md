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
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)


**Agentica**: A Human-Centric Framework for Large Language Model Agent Building.

**Agentica**: 构建你自己的Agent

## Overview

#### LLM Agent
![llm_agnet](https://github.com/shibing624/agentica/blob/main/docs/llm_agentv2.png)

- **规划（Planning）**：任务拆解、生成计划、反思
- **记忆（Memory）**：短期记忆（prompt实现）、长期记忆（RAG实现）
- **工具使用（Tool use）**：function call能力，调用外部API，以获取外部信息，包括当前日期、日历、代码执行能力、对专用信息源的访问等

#### Agentica架构
![agentica_arch](https://github.com/shibing624/agentica/blob/main/docs/agent_arch.png)

- **Planner**：负责让LLM生成一个多步计划来完成复杂任务，生成相互依赖的“链式计划”，定义每一步所依赖的上一步的输出
- **Worker**：接受“链式计划”，循环遍历计划中的每个子任务，并调用工具完成任务，可以自动反思纠错以完成任务
- **Solver**：求解器将所有这些输出整合为最终答案

## Features
`Agentica`是一个Agent构建工具，功能：

- 简单代码快速编排Agent，支持 Reflection(反思）、Plan and Solve(计划并执行)、RAG、Agent、Multi-Agent、Multi-Role、Workflow等功能
- Agent支持prompt自定义，支持多种工具调用（tool_calls）
- 支持OpenAI/Azure/Deepseek/Moonshot/Claude/Ollama/Together API调用

## Installation

```bash
pip install -U agentica
```

or

```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install .
```

## Getting Started

#### 1. Install requirements

```shell
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install -r requirements.txt
```

#### 2. Run the example
```shell
# Copying required .env file, and fill in the LLM api key
cp .env.example ~/.agentica/.env

cd examples
python web_search_deepseek_demo.py
```

1. 复制[.env.example](https://github.com/shibing624/agentica/blob/main/.env.example)文件为`~/.agentica/.env`，并填写LLM api key(选填DEEPSEEK_API_KEY、MOONSHOT_API_KEY、OPENAI_API_KEY等任一个即可)。

2. 使用`agentica`构建Agent并执行：

自动调用google搜索工具，示例[examples/web_search_deepseek_demo.py](https://github.com/shibing624/agentica/blob/main/examples/web_search_deepseek_demo.py)

```python
from agentica import Assistant, Deepseek
from agentica.tools.search_serper import SearchSerperTool

m = Assistant(
  llm=Deepseek(),
  description="You are a helpful ai assistant.",
  show_tool_calls=True,
  # Enable the assistant to search the knowledge base
  search_knowledge=False,
  tools=[SearchSerperTool()],
  # Enable the assistant to read the chat history
  read_chat_history=True,
  debug_mode=True,
)

r = m.run("一句话介绍林黛玉")
print(r, "".join(r))
r = m.run("北京最近的新闻top3", stream=True, print_output=True)
print(r, "".join(r))
r = m.run("总结前面的问答", stream=False, print_output=False)
print(r)
```


## Examples

| 示例                                                                                                                                    | 描述                                                                                                                              |
|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| [examples/naive_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/naive_rag_demo.py)                             | 实现了基础版RAG，基于Txt文档回答问题                                                                                                           |
| [examples/advanced_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/advanced_rag_demo.py)                       | 实现了高级版RAG，基于PDF文档回答问题，新增功能：pdf文件解析、query改写，字面+语义多路召回，召回排序（rerank）                                                               |
| [examples/python_assistant_demo.py](https://github.com/shibing624/agentica/blob/main/examples/python_assistant_demo.py)               | 实现了Code Interpreter功能，自动生成python代码，并执行                                                                                          |
| [examples/research_demo.py](https://github.com/shibing624/agentica/blob/main/examples/research_demo.py)                               | 实现了Research功能，自动调用搜索工具，汇总信息后撰写科技报告                                                                                              |
| [examples/team_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/team_news_article_demo.py)             | 实现了写新闻稿的team协作，multi-role实现，委托不用角色完成各自任务：研究员检索分析文章，撰写员根据排版写文章，汇总多角色成果输出结果                                                       |
| [examples/workflow_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_news_article_demo.py)     | 实现了写新闻稿的工作流，multi-agent的实现，定义了多个Assistant和Task，多次调用搜索工具，并生成高级排版的新闻文章                                                            |
| [examples/workflow_investment_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_investment_demo.py)         | 实现了投资研究的工作流：股票信息收集 - 股票分析 - 撰写分析报告 - 复查报告等多个Task                                                                                |
| [examples/crawl_webpage_demo.py](https://github.com/shibing624/agentica/blob/main/examples/crawl_webpage_demo.py)                          | 实现了网页分析工作流：从Url爬取融资快讯 - 分析网页内容和格式 - 提取核心信息 - 汇总保存为md文件                                                                          |
| [examples/find_paper_from_arxiv_demo.py](https://github.com/shibing624/agentica/blob/main/examples/find_paper_from_arxiv_demo.py)     | 实现了论文推荐工作流：自动从arxiv搜索多组论文 - 相似论文去重 - 提取核心论文信息 - 保存为csv文件                                                                        |
| [examples/remove_image_background_demo.py](https://github.com/shibing624/agentica/blob/main/examples/remove_image_background_demo.py) | 实现了自动去除图片背景功能，包括自动通过pip安装库，调用库实现去除图片背景                                                                                          |
| [examples/text_classification_demo.py](https://github.com/shibing624/agentica/blob/main/examples/text_classification_demo.py)         | 实现了自动训练分类模型的工作流：读取训练集文件并理解格式 - 谷歌搜索pytextclassifier库 - 爬取github页面了解pytextclassifier的调用方法 - 写代码并执行fasttext模型训练 - check训练好的模型预测结果 |
| [examples/llm_os_demo.py](https://github.com/shibing624/agentica/blob/main/examples/llm_os_demo.py)                                   | 实现了LLM OS的初步设计，基于LLM设计操作系统，可以通过LLM调用RAG、代码执行器、Shell等工具，并协同代码解释器、研究助手、投资助手等来解决问题。                                                |
| [examples/workflow_write_novel_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_write_novel_demo.py)        | 实现了写小说的工作流：定小说提纲 - 搜索谷歌反思提纲 - 撰写小说内容 - 保存为md文件                                                                                  |
| [examples/workflow_write_tutorial_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_write_tutorial_demo.py)  | 实现了写技术教程的工作流：定教程目录 - 反思目录内容 - 撰写教程内容 - 保存为md文件                                                                                  |


### LLM OS
The LLM OS design:

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/llmos.png" width="600" />

#### Run the LLM OS App

```shell
cd examples
streamlit run llm_os_demo.py
```

![llm_os](https://github.com/shibing624/agentica/blob/main/docs/llm_os_snap.png)

## Web UI

[shibing624/ChatPilot](https://github.com/shibing624/ChatPilot) 兼容`agentica`，可以通过Web UI进行交互。

Web Demo: https://chat.mulanai.com

![](https://github.com/shibing624/ChatPilot/blob/main/docs/shot.png)

```shell
git clone https://github.com/shibing624/ChatPilot.git
cd ChatPilot
pip install -r requirements.txt

cp .env.example .env

bash start.sh
```


## Contact

- Issue(建议)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我： 加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="https://github.com/shibing624/agentica/blob/main/docs/wechat.jpeg" width="200" />

## Citation

如果你在研究中使用了`agentica`，请按如下格式引用：

APA:

```
Xu, M. agentica: A Human-Centric Framework for Large Language Model Agent Workflows (Version 0.0.2) [Computer software]. https://github.com/shibing624/agentica
```

BibTeX:

```
@misc{Xu_agentica,
  title={agentica: A Human-Centric Framework for Large Language Model Agent Workflows},
  author={Xu Ming},
  year={2024},
  howpublished={\url{https://github.com/shibing624/agentica}},
}
```

## License

授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加`agentica`的链接和授权协议。
## Contribute

项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

- 在`tests`添加相应的单元测试
- 使用`python -m pytest`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

## Acknowledgements 

- [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- [https://github.com/simonmesmith/agentflow](https://github.com/simonmesmith/agentflow)
- [https://github.com/phidatahq/phidata](https://github.com/phidatahq/phidata)


Thanks for their great work!
