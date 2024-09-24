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


**Agentica**: A Human-Centric Framework for Large Language Model Agent Building. Quickly build your own Agent.

## Overview

#### LLM Agent
![llm_agnet](https://github.com/shibing624/agentica/blob/main/docs/llm_agentv2.png)

- **Planning**: Task decomposition, plan generation, reflection
- **Memory**: Short-term memory (prompt implementation), long-term memory (RAG implementation)
- **Tool use**: Function call capability, calling external APIs to obtain external information, including current date, calendar, code execution capability, access to specialized information sources, etc.

#### Agentica Assistant Architecture  
![agentica_arch](https://github.com/shibing624/agentica/blob/main/docs/agent_arch.png)

- **Planner**: Responsible for having the LLM generate a multi-step plan to complete complex tasks, generating interdependent "chain plans," defining the output of each step that depends on the previous step
- **Worker**: Accepts the "chain plan," iterates through each subtask in the plan, and calls tools to complete the task, can automatically reflect and correct errors to complete the task
- **Solver**: Integrates all these outputs into the final answer


## Features
`Agentica` is an Agent building tool with features:

- Quickly orchestrate Agents with simple code, supporting Reflection, Plan and Solve, RAG, Agent, Multi-Agent, Multi-Role, Workflow, etc.
- Agents support custom prompts and various tool calls
- Supports OpenAI/Azure/Deepseek/Moonshot/Claude/Ollama/Together API calls

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

1. Copy the [.env.example](https://github.com/shibing624/agentica/blob/main/.env.example) file to `~/.agentica/.env` and fill in the LLM API key (optional DEEPSEEK_API_KEY, MOONSHOT_API_KEY, OPENAI_API_KEY, etc.).

2. Build and run an Agent using `agentica`:

Automatically call the Google search tool, example [examples/web_search_deepseek_demo.py](https://github.com/shibing624/agentica/blob/main/examples/web_search_deepseek_demo.py)

```python
from agentica import Assistant, DeepseekLLM
from agentica.tools.search_serper import SearchSerperTool

m = Assistant(
  llm=DeepseekLLM(),
  description="You are a helpful ai assistant.",
  show_tool_calls=True,
  # Enable the assistant to search the knowledge base
  search_knowledge=False,
  tools=[SearchSerperTool()],
  # Enable the assistant to read the chat history
  read_chat_history=True,
  debug_mode=True,
)

r = m.run("Introduce Lin Daiyu in one sentence")
print(r, "".join(r))
r = m.run("Top 3 recent news in Beijing", stream=True, print_output=True)
print(r, "".join(r))
r = m.run("Summarize the previous Q&A", stream=False, print_output=False)
print(r)
```


## Web UI

[shibing624/ChatPilot](https://github.com/shibing624/ChatPilot) is compatible with `agentica` and can be interacted with through a Web UI.

Web Demo: https://chat.mulanai.com

![](https://github.com/shibing624/ChatPilot/blob/main/docs/shot.png)

```shell
git clone https://github.com/shibing624/ChatPilot.git
cd ChatPilot
pip install -r requirements.txt

cp .env.example .env

bash start.sh
```


## Examples

| Example                                                                                                                                    | Description                                                                                                                              |
|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| [examples/naive_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/naive_rag_demo.py)                             | Implements a basic RAG, answering questions based on a Txt document                                                                                                           |
| [examples/advanced_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/advanced_rag_demo.py)                       | Implements an advanced RAG, answering questions based on a PDF document, with new features: PDF file parsing, query rewriting, lexical + semantic multi-path retrieval, retrieval ranking (rerank)                                                               |
| [examples/python_assistant_demo.py](https://github.com/shibing624/agentica/blob/main/examples/python_assistant_demo.py)               | Implements Code Interpreter functionality, automatically generating and executing Python code                                                                                          |
| [examples/research_demo.py](https://github.com/shibing624/agentica/blob/main/examples/research_demo.py)                               | Implements Research functionality, automatically calling search tools, summarizing information, and writing scientific reports                                                                                              |
| [examples/team_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/team_news_article_demo.py)             | Implements team collaboration for writing news articles, multi-role implementation, delegating different roles to complete their respective tasks: researchers retrieve and analyze articles, writers write articles based on the layout, summarizing multi-role results                                                       |
| [examples/workflow_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_news_article_demo.py)     | Implements a workflow for writing news articles, multi-agent implementation, defining multiple Assistants and Tasks, calling search tools multiple times, and generating advanced layout news articles                                                            |
| [examples/workflow_investment_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_investment_demo.py)         | Implements an investment research workflow: stock information collection - stock analysis - writing analysis reports - reviewing reports, etc.                                                                                |
| [examples/crawl_webpage_demo.py](https://github.com/shibing624/agentica/blob/main/examples/crawl_webpage_demo.py)                     | Implements a webpage analysis workflow: crawling financing news from URLs - analyzing webpage content and format - extracting core information - summarizing and saving as md files                                                                          |
| [examples/find_paper_from_arxiv_demo.py](https://github.com/shibing624/agentica/blob/main/examples/find_paper_from_arxiv_demo.py)     | Implements a paper recommendation workflow: automatically searching multiple groups of papers from arxiv - deduplicating similar papers - extracting core paper information - saving as csv files                                                                        |
| [examples/remove_image_background_demo.py](https://github.com/shibing624/agentica/blob/main/examples/remove_image_background_demo.py) | Implements automatic image background removal functionality, including automatic library installation via pip, calling libraries to remove image backgrounds                                                                                          |
| [examples/text_classification_demo.py](https://github.com/shibing624/agentica/blob/main/examples/text_classification_demo.py)         | Implements an automatic training classification model workflow: reading training set files and understanding the format - Google searching for pytextclassifier library - crawling GitHub pages to understand how to call pytextclassifier - writing code and executing fasttext model training - checking the trained model prediction results |
| [examples/llm_os_demo.py](https://github.com/shibing624/agentica/blob/main/examples/llm_os_demo.py)                                   | Implements the initial design of LLM OS, designing an operating system based on LLM, which can call RAG, code executors, Shell, etc. through LLM, and collaborate with code interpreters, research assistants, investment assistants, etc. to solve problems.                                                |
| [examples/workflow_write_novel_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_write_novel_demo.py)       | Implements a workflow for writing novels: setting the novel outline - Google searching and reflecting on the outline - writing novel content - saving as md files                                                                                  |
| [examples/workflow_write_tutorial_demo.py](https://github.com/shibing624/agentica/blob/main/examples/workflow_write_tutorial_demo.py) | Implements a workflow for writing technical tutorials: setting the tutorial directory - reflecting on the directory content - writing tutorial content - saving as md files                                                                                  |
| [examples/self_evolving_agent_demo.py](https://github.com/shibing624/agentica/blob/main/examples/self_evolving_agent_demo.py)         | Implements a self-evolving agent based on long-term memory, which can adjust decisions based on historical Q&A information                                                                                              |


### LLM OS
The LLM OS design:

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/llmos.png" width="600" />

#### Run the LLM OS App

```shell
cd examples
streamlit run llm_os_demo.py
```

![llm_os](https://github.com/shibing624/agentica/blob/main/docs/llm_os_snap.png)

### Self-evolving Agent
The self-evolving agent design:

<img alt="LLM OS" src="https://github.com/shibing624/agentica/blob/main/docs/sage_arch.png" width="600" />

#### Feature

Self-evolving Agents with Reflective and Memory-augmented Abilities (SAGE)

Implement:
1. Use `PythonAssistant` as the SAGE agent and `AzureOpenAILLM` as the LLM, with code-interpreter functionality to execute Python code and automatically correct errors.
2. Use `CsvMemoryDb` as the memory for the SAGE agent to store user questions and answers, so that similar questions can be directly answered next time.

#### Run Self-evolving Agent App

```shell
cd examples
streamlit run self_evolving_agent_demo.py
```

![sage](https://github.com/shibing624/agentica/blob/main/docs/sage_snap.png)


## Contact

- Issue (suggestions)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
- Email me: xuming: xuming624@qq.com
- WeChat me: Add my *WeChat ID: xuming624, note: Name-Company-NLP* to join the NLP group.

<img src="https://github.com/shibing624/agentica/blob/main/docs/wechat.jpeg" width="200" />

## Citation

If you use `agentica` in your research, please cite it as follows:

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

The license is [The Apache License 2.0](/LICENSE), free for commercial use. Please include a link to `agentica` and the license in the product description.
## Contribute

The project code is still rough, if you have any improvements to the code, you are welcome to submit them back to this project. Before submitting, please note the following two points:

- Add corresponding unit tests in `tests`
- Use `python -m pytest` to run all unit tests and ensure all tests pass

Then you can submit a PR.

## Acknowledgements 

- [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- [https://github.com/simonmesmith/agentflow](https://github.com/simonmesmith/agentflow)
- [https://github.com/phidatahq/phidata](https://github.com/phidatahq/phidata)


Thanks for their great work!
