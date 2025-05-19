# Agentica Examples


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


## LLM OS

Let's build the LLM OS proposed by Andrej Karpathy [in this tweet](https://twitter.com/karpathy/status/1723140519554105733), [this tweet](https://twitter.com/karpathy/status/1707437820045062561) and [this video](https://youtu.be/zjkBMFhNj_g?t=2535).

### The LLM OS design:

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/llmos.png" width="600" />

- LLMs are the kernel process of an emerging operating system.
- This process (LLM) can solve problems by coordinating other resources (memory, computation tools).
- The LLM OS:
  - [x] Can read/generate text
  - [x] Has more knowledge than any single human about all subjects
  - [x] Can browse the internet
  - [x] Can use existing software infra (calculator, python, mouse/keyboard)
  - [x] Can see and generate images and video
  - [x] Can hear and speak, and generate music
  - [x] Can think for a long time using a system 2
  - [x] Can “self-improve” in domains
  - [x] Can be customized and fine-tuned for specific tasks
  - [x] Can communicate with other LLMs


## Running the LLM OS:

> Note: Fork and clone this repository if needed

### 1. Create a virtual environment

```shell
python3 -m venv ~/.venvs/aienv
source ~/.venvs/aienv/bin/activate
```

### 2. Install libraries

```shell
pip install agentica streamlit text2vec sqlalchemy lancedb pyarrow yfinance
```

### 3. Export credentials

- Our initial implementation uses GPT-4, so export your OpenAI API Key in the `../.env` file

```shell
OPENAI_API_KEY=***
EXA_API_KEY=xxx # optional
SERPER_API_KEY=xxx # optional
```

### 4. Run the LLM OS App

```shell
cd examples
streamlit run 34_llm_os_demo.py
```

![llm_os](https://github.com/shibing624/agentica/blob/main/docs/llm_os_snap.png)

- Open [localhost:8501](http://localhost:8501) to view your LLM OS.
- Add a blog URL to knowledge base: https://blog.samaltman.com/gpt-4o
- Ask: What is gpt-4o?
- `Web search`: 北京今天天气?
- Enable `shell tool` and ask: is docker running?
- `Python Assistant`: 帮我计算下 [168, 151, 171, 105, 124, 159, 153, 132, 112.2] , 计算它们的平均值。
- Enable the `Research Assistant` and ask: write a report on the ibm hashicorp acquisition
- Enable the `Investment Assistant` and ask: shall i invest in nvda?


