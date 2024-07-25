# Agentica Examples



| 示例                                                                                                                                     | 描述                                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| [examples/naive_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/naive_rag_demo.py)                              | 实现了基础版RAG，基于Txt文档回答问题                                                                                                           |
| [examples/advanced_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/advanced_rag_demo.py)                        | 实现了高级版RAG，基于PDF文档回答问题，新增功能：pdf文件解析、query改写，字面+语义多路召回，召回排序（rerank）                                                               |
| [examples/python_assistant_demo.py](https://github.com/shibing624/agentica/blob/main/examples/python_assistant_demo.py)                | 实现了Code Interpreter功能，自动生成python代码，并执行                                                                                          |
| [examples/research_demo.py](https://github.com/shibing624/agentica/blob/main/examples/research_demo.py)                                | 实现了Research功能，自动调用搜索工具，汇总信息后撰写科技报告                                                                                              |
| [examples/team_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/team_news_article_demo.py)              | 实现了写新闻稿的team协作，multi-role实现，委托不用角色完成各自任务：研究员检索分析文章，撰写员根据排版写文章，汇总多角色成果输出结果                                                       |
| [examples/workflow_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_news_article_demo.py)      | 实现了写新闻稿的工作流，multi-agent的实现，定义了多个Assistant和Task，多次调用搜索工具，并生成高级排版的新闻文章                                                            |
| [examples/workflow_investment_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_investment_demo.py)          | 实现了投资研究的工作流：股票信息收集 - 股票分析 - 撰写分析报告 - 复查报告等多个Task                                                                                |
| [examples/crawl_webpage_demo.py](https://github.com/shibing624/agentica/blob/main/examples/crawl_webpage_demo.py)                      | 实现了网页分析工作流：从Url爬取融资快讯 - 分析网页内容和格式 - 提取核心信息 - 汇总保存为md文件                                                                          |
| [examples/find_paper_from_arxiv_demo.py](https://github.com/shibing624/agentica/blob/main/examples/find_paper_from_arxiv_demo.py)      | 实现了论文推荐工作流：自动从arxiv搜索多组论文 - 相似论文去重 - 提取核心论文信息 - 保存为csv文件                                                                        |
| [examples/remove_image_background_demo.py](https://github.com/shibing624/agentica/blob/main/examples/remove_image_background_demo.py)  | 实现了自动去除图片背景功能，包括自动通过pip安装库，调用库实现去除图片背景                                                                                          |
| [examples/text_classification_demo.py](https://github.com/shibing624/agentica/blob/main/examples/text_classification_demo.py)          | 实现了自动训练分类模型的工作流：读取训练集文件并理解格式 - 谷歌搜索pytextclassifier库 - 爬取github页面了解pytextclassifier的调用方法 - 写代码并执行fasttext模型训练 - check训练好的模型预测结果 |
| [examples/llm_os_demo.py](https://github.com/shibing624/agentica/blob/main/examples/llm_os_demo.py)                                    | 实现了LLM OS的初步设计，基于LLM设计操作系统，可以通过LLM调用RAG、代码执行器、Shell等工具，并协同代码解释器、研究助手、投资助手等来解决问题。                                                |
| [examples/workflow_write_novel_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_write_novel_demo.py)        | 实现了写小说的工作流：定小说提纲 - 搜索谷歌反思提纲 - 撰写小说内容 - 保存为md文件                                                                                  |
| [examples/workflow_write_tutorial_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_write_tutorial_demo.py)  | 实现了写技术教程的工作流：定教程目录 - 反思目录内容 - 撰写教程内容 - 保存为md文件                                                                                  |


## LLM OS

Lets build the LLM OS proposed by Andrej Karpathy [in this tweet](https://twitter.com/karpathy/status/1723140519554105733), [this tweet](https://twitter.com/karpathy/status/1707437820045062561) and [this video](https://youtu.be/zjkBMFhNj_g?t=2535).

### The LLM OS design:

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/llmos.png" width="600" />

- LLMs are the kernel process of an emerging operating system.
- This process (LLM) can solve problems by coordinating other resources (memory, computation tools).
- The LLM OS:
  - [x] Can read/generate text
  - [x] Has more knowledge than any single human about all subjects
  - [x] Can browse the internet
  - [x] Can use existing software infra (calculator, python, mouse/keyboard)
  - [ ] Can see and generate images and video
  - [ ] Can hear and speak, and generate music
  - [ ] Can think for a long time using a system 2
  - [ ] Can “self-improve” in domains
  - [ ] Can be customized and fine-tuned for specific tasks
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
streamlit run llm_os_demo.py
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


