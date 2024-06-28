[**🇨🇳中文**](https://github.com/shibing624/actionflow/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/actionflow/blob/main/README_EN.md)

<div align="center">
  <a href="https://github.com/shibing624/actionflow">
    <img src="https://raw.githubusercontent.com/shibing624/actionflow/main/docs/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# Actionflow: Agent Workflows with Prompts and Tools
[![PyPI version](https://badge.fury.io/py/actionflow.svg)](https://badge.fury.io/py/actionflow)
[![Downloads](https://static.pepy.tech/badge/actionflow)](https://pepy.tech/project/actionflow)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/actionflow.svg)](https://github.com/shibing624/actionflow/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)


**actionflow**: A Human-Centric Framework for Large Language Model Agent Workflows, build your agent workflows quickly

**actionflow**: 快速构建你自己的Agent工作流

## Overview
![llm_agnet](https://github.com/shibing624/actionflow/blob/main/docs/llm_agent.png)

- **规划（Planning）**：任务拆解、生成计划、反思
- **记忆（Memory）**：短期记忆（prompt实现）、长期记忆（RAG实现）
- **工具使用（Tool use）**：function call能力，调用外部API，以获取外部信息，包括当前日期、日历、代码执行能力、对专用信息源的访问等


![actionflow_arch](https://github.com/shibing624/actionflow/blob/main/docs/actionflow_arch.png)

- **Planner**：负责让LLM生成一个多步计划来完成复杂任务，生成相互依赖的“链式计划”，定义每一步所依赖的上一步的输出
- **Worker**：接受“链式计划”，循环遍历计划中的每个子任务，并调用工具完成任务，可以自动反思纠错以完成任务
- **Solver**：求解器将所有这些输出整合为最终答案

## Features
`Actionflow`是一个Agent工作流构建工具，功能：

- 简单代码快速编排复杂工作流
- 工作流的编排不仅支持prompt命令，还支持工具调用（tool_calls）
- 支持OpenAI API和Moonshot API(kimi)调用

## Installation

```bash
pip install -U actionflow
```

or

```bash
git clone https://github.com/shibing624/actionflow.git
cd actionflow
pip install .
```

## Getting Started

1. 复制[example.env](https://github.com/shibing624/actionflow/blob/main/example.env)文件为`.env`，并粘贴OpenAI API key或者Moonshoot API key。

2. 运行Agent示例，自动调用google搜索工具：

```python
from actionflow import Assistant, OpenAILLM, AzureOpenAILLM
from actionflow.tools.search_serper import SearchSerperTool

m = Assistant(
    llm=AzureOpenAILLM(),
    description="You are a helpful ai assistant.",
    show_tool_calls=True,
    # Enable the assistant to search the knowledge base
    search_knowledge=False,
    tools=[SearchSerperTool()],
    # Enable the assistant to read the chat history
    read_chat_history=True,
    debug_mode=True,
)
print("LLM:", m.llm)
print(m.run("介绍林黛玉", stream=False))
print(m.run("北京最近的新闻", stream=False))
print(m.run("我前面问了啥", stream=False))
```

## Examples
运行工作流（Workflow）示例：

- [examples/rag_assistant_demo.py](https://github.com/shibing624/actionflow/blob/main/examples/rag_assistant_demo.py) 实现了RAG功能，基于PDF文档回答问题
- [examples/python_assistant_demo.py](https://github.com/shibing624/actionflow/blob/main/examples/python_assistant_demo.py) 实现了Code Interpreter功能，自动生成python代码，并执行
- [examples/research_demo.py](https://github.com/shibing624/actionflow/blob/main/examples/research_demo.py) 实现了Research功能，自动调用搜索工具，汇总信息后撰写科技报告
- [examples/run_flow_news_article_demo.py](https://github.com/shibing624/actionflow/blob/main/examples/run_flow_news_article_demo.py) 实现了写新闻稿的工作流，multi-agent的实现，定义了多个Assistant和Task，多次调用搜索工具，并生成高级排版的新闻文章
- [examples/run_flow_investment_demo.py](https://github.com/shibing624/actionflow/blob/main/examples/run_flow_investment_demo.py) 实现了投资研究的工作流，依次执行股票信息收集、股票分析、撰写分析报告，复查报告等多个Task

## Contact

- Issue(建议)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/actionflow.svg)](https://github.com/shibing624/actionflow/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我： 加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="https://github.com/shibing624/actionflow/blob/main/docs/wechat.jpeg" width="200" />

## Citation

如果你在研究中使用了`actionflow`，请按如下格式引用：

APA:

```
Xu, M. actionflow: A Human-Centric Framework for Large Language Model Agent Workflows (Version 0.0.2) [Computer software]. https://github.com/shibing624/actionflow
```

BibTeX:

```
@misc{Xu_actionflow,
  title={actionflow: A Human-Centric Framework for Large Language Model Agent Workflows},
  author={Xu Ming},
  year={2024},
  howpublished={\url{https://github.com/shibing624/actionflow}},
}
```

## License

授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加`actionflow`的链接和授权协议。
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
