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
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)


**Agentica**: 轻松构建智能、具备反思能力、可协作的多模态AI Agent。


## 📖 Introduction

**Agentica** 可以构建AI Agent，包括规划、记忆和工具使用、执行等组件。

## 🔥 News
[2025/05/19] v1.0.6版本：新增了`MCP`的`StreamableHttp`支持，兼容 StreamableHttp/SSE/Stdio 三种MCP Server，详见[Release-v1.0.6](https://github.com/shibing624/agentica/releases/tag/1.0.6)

[2025/04/21] v1.0.0版本：支持了`MCP`的工具调用，兼容 SSE/Stdio 的 MCP Server，详见[Release-v1.0.0](https://github.com/shibing624/agentica/releases/tag/1.0.0)

[2024/12/29] v0.2.3版本: 支持了`ZhipuAI`的api调用，包括免费模型和工具使用，详见[Release-v0.2.3](https://github.com/shibing624/agentica/releases/tag/0.2.3)

[2024/12/25] v0.2.0版本: 支持了多模态模型，输入可以是文本、图片、音频、视频，升级Assistant为Agent，Workflow支持拆解并实现复杂任务，详见[Release-v0.2.0](https://github.com/shibing624/agentica/releases/tag/0.2.0)

[2024/07/02] v0.1.0版本：实现了基于LLM的Assistant，可以快速用function call搭建大语言模型助手，详见[Release-v0.1.0](https://github.com/shibing624/agentica/releases/tag/0.1.0)


## 😊 Features
`Agentica`是一个用于构建Agent的工具，具有以下功能：

- **Agent编排**：通过简单代码快速编排Agent，支持 Reflection(反思）、Plan and Solve(计划并执行)、RAG、Agent、Multi-Agent、Team、Workflow等功能
- **工具调用**：支持自定义工具OpenAI的function call，支持MCP Server的工具调用
- **LLM集成**：支持OpenAI、Azure、Deepseek、Moonshot、Anthropic、ZhipuAI、Ollama、Together等多方大模型厂商的API
- **记忆功能**：支持短期记忆和长期记忆功能
- **Multi-Agent协作**：支持多Agent和任务委托（Team）的团队协作
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
python 12_web_search_moonshot_demo.py
```

1. 复制[.env.example](https://github.com/shibing624/agentica/blob/main/.env.example)文件为`~/.agentica/.env`，并填写LLM api key(选填OPENAI_API_KEY、ZHIPUAI_API_KEY、MOONSHOT_API_KEY 等任一个)。或者使用`export`命令设置环境变量：
    
    ```shell
    export MOONSHOT_API_KEY=your_api_key
    export SERPER_API_KEY=your_serper_api_key
    ```

2. 使用`agentica`构建Agent并执行：

自动调用google搜索工具，示例[examples/12_web_search_moonshot_demo.py](https://github.com/shibing624/agentica/blob/main/examples/12_web_search_moonshot_demo.py)

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
| [examples/40_weather_zhipuai_demo.py](https://github.com/shibing624/agentica/blob/main/examples/40_web_search_zhipuai_demo.py)                        | 基于智谱AI的api做天气查询的Demo                                                                                                              |
| [examples/41_mcp_stdio_demo.py](https://github.com/shibing624/agentica/blob/main/examples/41_mcp_stdio_demo.py)                                       | Stdio的MCP Server调用的Demo                                                                                                           |
| [examples/42_mcp_sse_server.py](https://github.com/shibing624/agentica/blob/main/examples/42_mcp_sse_server.py)                                       | SSE的MCP Server调用的Demo                                                                                                             |
| [examples/42_mcp_sse_client.py](https://github.com/shibing624/agentica/blob/main/examples/42_mcp_sse_client.py)                                       | SSE的MCP Client调用的Demo                                                                                                             |
| [examples/43_minimax_mcp_demo.py](https://github.com/shibing624/agentica/blob/main/examples/43_minimax_mcp_demo.py)                                   | Minimax语音生成调用的Demo                                                                                                                |
| [examples/44_mcp_streamable_http_server.py](https://github.com/shibing624/agentica/blob/main/examples/44_mcp_streamable_http_server.py)                           | Streamable Http的MCP Server调用的Demo                                                                                                 |
| [examples/44_mcp_streamable_http_client.py](https://github.com/shibing624/agentica/blob/main/examples/44_mcp_streamable_http_client.py)                           | Streamable Http的MCP Client调用的Demo                                                                                                 |


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


### 命令行模式（CLI）

支持终端命令行快速搭建并体验Agent

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
