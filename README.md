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


**Agentica**: 轻松构建智能、具备反思能力、可协作的多模态AI Agent。


## 📖 Introduction

**Agentica** 可以构建AI Agent，包括规划、记忆和工具使用、执行等组件。

#### Agent Components
<img src="https://github.com/shibing624/agentica/blob/main/docs/llm_agentv2.png" width="800" />

- **规划（Planning）**：任务拆解、生成计划、反思
- **记忆（Memory）**：短期记忆（prompt实现）、长期记忆（RAG实现）
- **工具使用（Tool use）**：function call能力，调用外部API，以获取外部信息，包括当前日期、日历、代码执行能力、对专用信息源的访问等

#### Agentica Workflow

**Agentica** can also build multi-agent systems and workflows.

**Agentica** 还可以构建多Agent系统和工作流。

<img src="https://github.com/shibing624/agentica/blob/main/docs/agent_arch.png" width="800" />

- **Planner**：负责让LLM生成一个多步计划来完成复杂任务，生成相互依赖的“链式计划”，定义每一步所依赖的上一步的输出
- **Worker**：接受“链式计划”，循环遍历计划中的每个子任务，并调用工具完成任务，可以自动反思纠错以完成任务
- **Solver**：求解器将所有这些输出整合为最终答案

## 🔥 News
[2024/12/29] v0.2.3版本: 支持了`ZhipuAI`的api调用，包括免费模型和工具使用，详见[Release-v0.2.3](https://github.com/shibing624/agentica/releases/tag/0.2.3)

[2024/12/25] v0.2.0版本: 支持了多模态模型，输入可以是文本、图片、音频、视频，升级Assistant为Agent，Workflow支持拆解并实现复杂任务，详见[Release-v0.2.0](https://github.com/shibing624/agentica/releases/tag/0.2.0)

[2024/07/02] v0.1.0版本：实现了基于LLM的Assistant，可以快速用function call搭建大语言模型助手，详见[Release-v0.1.0](https://github.com/shibing624/agentica/releases/tag/0.1.0)


## 😊 Features
`Agentica`是一个用于构建Agent的工具，具有以下功能：

- **Agent编排**：通过简单代码快速编排Agent，支持 Reflection(反思）、Plan and Solve(计划并执行)、RAG、Agent、Multi-Agent、Team、Workflow等功能
- **自定义prompt**：Agent支持自定义prompt和多种工具调用（tool_calls）
- **LLM集成**：支持OpenAI、Azure、Deepseek、Moonshot、Claude、Ollama、Together等多方大模型厂商的API
- **记忆功能**：包括短期记忆和长期记忆功能
- **Multi-Agent协作**：支持多Agent和任务委托（Team）的团队协作。
- **Workflow工作流**：拆解复杂任务为多个Agent，基于工作流自动化串行逐步完成任务，如投资研究、新闻文章撰写和技术教程创建
- **自我进化Agent**：具有反思和增强记忆能力的自我进化Agent
- **Web UI**：兼容ChatPilot，可以基于Web页面交互，支持主流的open-webui、streamlit、gradio等前端交互框架

## 💾 Install

```bash
pip install -U agentica
```

or

```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install .
```

## 🚀 Getting Started

#### Run the example
```shell
# Copying required .env file, and fill in the LLM api key
cp .env.example ~/.agentica/.env

cd examples
python web_search_moonshot_demo.py
```

1. 复制[.env.example](https://github.com/shibing624/agentica/blob/main/.env.example)文件为`~/.agentica/.env`，并填写LLM api key(选填DEEPSEEK_API_KEY、MOONSHOT_API_KEY、OPENAI_API_KEY、ZHIPUAI_API_KEY等任一个即可)。或者使用`export`命令设置环境变量：
    
    ```shell
    export MOONSHOT_API_KEY=your_moonshot_api_key
    export SERPER_API_KEY=your_serper_api_key
    ```

2. 使用`agentica`构建Agent并执行：

自动调用google搜索工具，示例[examples/12_web_search_moonshot_demo.py](https://github.com/shibing624/agentica/blob/main/examples/12_web_search_moonshot_demo.py)

```python
from agentica import Agent, MoonshotChat, SearchSerperTool

m = Agent(model=MoonshotChat(), tools=[SearchSerperTool()], add_datetime_to_instructions=True)
r = m.run("下一届奥运会在哪里举办")
print(r)
```


## ▶️ Web UI

[shibing624/ChatPilot](https://github.com/shibing624/ChatPilot) 兼容`agentica`，可以通过Web UI进行交互。

Web Demo: https://chat.mulanai.com

<img src="https://github.com/shibing624/ChatPilot/blob/main/docs/shot.png" width="800" />

```shell
git clone https://github.com/shibing624/ChatPilot.git
cd ChatPilot
pip install -r requirements.txt

cp .env.example .env

bash start.sh
```


## 😀 Examples


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
| [examples/12_web_search_moonshot_demo.py](https://github.com/shibing624/agentica/blob/main/examples/12_web_search_moonshot_demo.py)                   | 基于Moonshot的function call做网页搜索Demo                                                                                                 |
| [examples/13_storage_demo.py](https://github.com/shibing624/agentica/blob/main/examples/13_storage_demo.py)                                           | Agent的存储Demo                                                                                                                      |
| [examples/14_custom_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/14_custom_tool_demo.py)                                   | 自定义工具，并用大模型自主选择调用的Demo                                                                                                            |
| [examples/15_crawl_webpage_demo.py](https://github.com/shibing624/agentica/blob/main/examples/15_crawl_webpage_demo.py)                               | 实现了网页分析工作流：从Url爬取融资快讯 - 分析网页内容和格式 - 提取核心信息 - 汇总存为md文件                                                                             |
| [examples/16_get_top_papers_demo.py](https://github.com/shibing624/agentica/blob/main/examples/16_get_top_papers_demo.py)                             | 解析每日论文，并保存为json格式的Demo                                                                                                            |
| [examples/17_find_paper_from_arxiv_demo.py](https://github.com/shibing624/agentica/blob/main/examples/17_find_paper_from_arxiv_demo.py)               | 实现了论文推荐的Demo：自动从arxiv搜索多组论文 - 相似论文去重 - 提取核心论文信息 - 保存为csv文件                                                                        |
| [examples/18_agent_input_is_list.py](https://github.com/shibing624/agentica/blob/main/examples/18_agent_input_is_list.py)                             | 展示Agent的message可以是列表的Demo                                                                                                         |
| [examples/19_naive_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/19_naive_rag_demo.py)                                       | 实现了基础版RAG，基于Txt文档回答问题                                                                                                             |
| [examples/20_advanced_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/20_advanced_rag_demo.py)                                 | 实现了高级版RAG，基于PDF文档回答问题，新增功能：pdf文件解析、query改写，字面+语义多路混合召回，召回排序（rerank）                                                               |
| [examples/21_memorydb_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/21_reference_in_prompt_rag_demo.py)                      | 把参考资料放到prompt的传统RAG做法的Demo                                                                                                        |
| [examples/22_chat_pdf_app_demo.py](https://github.com/shibing624/agentica/blob/main/examples/22_chat_pdf_app_demo.py)                                 | 对PDF文档做深入对话的Demo                                                                                                                  |
| [examples/23_python_agent_memory_demo.py](https://github.com/shibing624/agentica/blob/main/examples/23_python_agent_memory_demo.py)                   | 实现了带记忆的Code Interpreter功能，自动生成python代码并执行，下次执行时从记忆获取结果                                                                            |
| [examples/24_context_demo.py](https://github.com/shibing624/agentica/blob/main/examples/24_context_demo.py)                                           | 实现了传入上下文进行对话的Demo                                                                                                                 |
| [examples/25_tools_with_context_demo.py](https://github.com/shibing624/agentica/blob/main/examples/25_tools_with_context_demo.py)                     | 工具带上下文传参的Demo                                                                                                                     |
| [examples/26_complex_translate_demo.py](https://github.com/shibing624/agentica/blob/main/examples/26_complex_translate_demo.py)                       | 实现了复杂翻译Demo                                                                                                                       |
| [examples/27_research_agent_demo.py](https://github.com/shibing624/agentica/blob/main/examples/27_research_agent_demo.py)                             | 实现了Research功能，自动调用搜索工具，汇总信息后撰写科技报告                                                                                                |
| [examples/28_rag_integrated_langchain_demo.py](https://github.com/shibing624/agentica/blob/main/examples/28_rag_integrated_langchain_demo.py)         | 集成LangChain的RAG Demo                                                                                                              |
| [examples/29_rag_integrated_llamaindex_demo.py](https://github.com/shibing624/agentica/blob/main/examples/29_rag_integrated_llamaindex_demo.py)       | 集成LlamaIndex的RAG Demo                                                                                                             |
| [examples/30_text_classification_demo.py](https://github.com/shibing624/agentica/blob/main/examples/30_text_classification_demo.py)                   | 实现了自动训练分类模型的Agent：读取训练集文件并理解格式 - 谷歌搜索pytextclassifier库 - 爬取github页面了解pytextclassifier的调用方法 - 写代码并执行fasttext模型训练 - check训练好的模型预测结果 |
| [examples/31_team_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/31_team_news_article_demo.py)                       | Team实现：写新闻稿的team协作，multi-role实现，委托不用角色完成各自任务：研究员检索分析文章，撰写员根据排版写文章，汇总多角色成果输出结果                                                     |
| [examples/32_team_debate_demo.py](https://github.com/shibing624/agentica/blob/main/examples/32_team_debate_demo.py)                                   | Team实现：基于委托做双人辩论Demo，特朗普和拜登辩论                                                                                                     |
| [examples/33_self_evolving_agent_demo.py](https://github.com/shibing624/agentica/blob/main/examples/33_self_evolving_agent_demo.py)                   | 实现了自我进化Agent的Demo                                                                                                                 |
| [examples/34_llm_os_demo.py](https://github.com/shibing624/agentica/blob/main/examples/34_llm_os_demo.py)                                             | 实现了LLM OS的初步设计，基于LLM设计操作系统，可以通过LLM调用RAG、代码执行器、Shell等工具，并协同代码解释器、研究助手、投资助手等来解决问题。                                                  |
| [examples/35_workflow_investment_demo.py](https://github.com/shibing624/agentica/blob/main/examples/35_workflow_investment_demo.py)                   | 实现了投资研究的工作流：股票信息收集 - 股票分析 - 撰写分析报告 - 复查报告等多个Task                                                                                  |
| [examples/36_workflow_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/36_workflow_news_article_demo.py)               | 实现了写新闻稿的工作流，multi-agent的实现，多次调用搜索工具，并生成高级排版的新闻文章                                                                                  |
| [examples/37_workflow_write_novel_demo.py](https://github.com/shibing624/agentica/blob/main/examples/37_workflow_write_novel_demo.py)                 | 实现了写小说的工作流：定小说提纲 - 搜索谷歌反思提纲 - 撰写小说内容 - 保存为md文件                                                                                    |
| [examples/38_workflow_write_tutorial_demo.py](https://github.com/shibing624/agentica/blob/main/examples/38_workflow_write_tutorial_demo.py)           | 实现了写技术教程的工作流：定教程目录 - 反思目录内容 - 撰写教程内容 - 保存为md文件                                                                                    |
| [examples/39_audio_multi_turn_demo.py](https://github.com/shibing624/agentica/blob/main/examples/39_audio_multi_turn_demo.py)                         | 基于openai的语音api做多轮音频对话的Demo                                                                                                        |


### Self-evolving Agent
The self-evolving agent design:

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/sage_arch.png" width="800" />

#### Feature

具有反思和增强记忆能力的自我进化智能体(self-evolving Agents with Reflective and Memory-augmented Abilities, SAGE)

实现方法:

1. 使用PythonAgent作为SAGE智能体，使用AzureOpenAIChat作为LLM, 具备code-interpreter功能，可以执行Python代码，并自动纠错。
2. 使用CsvMemoryDb作为SAGE智能体的记忆，用于存储用户的问题和答案，下次遇到相似的问题时，可以直接返回答案。

#### Run Self-evolving Agent App

```shell
cd examples
streamlit run 33_self_evolving_agent_demo.py
```

<img alt="sage_snap" src="https://github.com/shibing624/agentica/blob/main/docs/sage_snap.png" width="800" />


### LLM OS
The LLM OS design:

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/llmos.png" width="800" />

#### Run the LLM OS App

```shell
cd examples
streamlit run 34_llm_os_demo.py
```

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/llm_os_snap.png" width="800" />

## ☎️ Contact

- Issue(建议)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我： 加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="https://github.com/shibing624/agentica/blob/main/docs/wechat.jpeg" width="200" />

## 😇 Citation

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

## ⚠️ License

授权协议为 [The Apache License 2.0](/LICENSE)，可免费用做商业用途。请在产品说明中附加`agentica`的链接和授权协议。
## 😍 Contribute

项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目，在提交之前，注意以下两点：

- 在`tests`添加相应的单元测试
- 使用`python -m pytest`来运行所有单元测试，确保所有单测都是通过的

之后即可提交PR。

## 💕 Acknowledgements

- [langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- [simonmesmith/agentflow](https://github.com/simonmesmith/agentflow)
- [phidatahq/phidata](https://github.com/phidatahq/phidata)


Thanks for their great work!
