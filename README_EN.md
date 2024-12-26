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

**Agentica**: Effortlessly Build Intelligent, Reflective, and Collaborative Multimodal AI Agents!

## 📖 Introduction

**Agentica** can build AI agent, which have the components of planning, memory, and tool use. 

#### Agent Components
<img src="https://github.com/shibing624/agentica/blob/main/docs/llm_agentv2.png" width="800" />

- **Planning**: task decomposition, plan generation, reflection
- **Memory**: short-term memory (prompt implementation), long-term memory (RAG implementation)
- **Tool use**: function call capability, call external API to obtain external information, including current date, calendar, code execution capability, access to dedicated information sources, etc.

#### Agentica Workflow

**Agentica** can also build multi-agent systems and workflows.

**Agentica** can also build multi-agent systems and workflows.

<img src="https://github.com/shibing624/agentica/blob/main/docs/agent_arch.png" width="800" />

- **Planner**: responsible for LLM to generate a multi-step plan to complete complex tasks, generate interdependent "chain plans", and define the output of the previous step that each step depends on
- **Worker**: accepts the "chain plan", loops through each subtask in the plan, and calls tools to complete the task, which can automatically reflect and correct errors to complete the task
- **Solver**: The solver integrates all these outputs into the final answer

## 🔥 News
[2024/12/25] v0.2.0 version: Supports multimodal models, input can be text, pictures, audio, video, upgrade Assistant to Agent, Workflow supports disassembly and implementation of complex tasks, see [Release-v0.2.0](https://github.com/shibing624/agentica/releases/tag/0.2.0)

[2024/07/02] v0.1.0 version: Implemented Assistant based on LLM, can quickly use function call to build a large language model assistant, see [Release-v0.1.0](https://github.com/shibing624/agentica/releases/tag/0.1.0)

## 😊 Features
`Agentica` is a tool for building agents with the following features:

- **Agent Composition**: Quickly compose agents with simple code, supporting Reflection, Plan and Solve, RAG, Agent, Multi-Agent, Multi-Role, and Workflow functionalities.
- **Custom Prompts**: Agents support custom prompts and various tool calls.
- **LLM Integration**: Supports APIs from multiple large model providers such as OpenAI, Azure, Deepseek, Moonshot, Claude, Ollama, and Together.
- **Memory Capabilities**: Includes short-term and long-term memory functionalities.
- **Multi-Agent Collaboration**: Supports team collaboration with multiple agents and roles.
- **Workflow Automation**: Automates complex workflows by breaking down tasks into multiple agents, such as investment research, news article writing, and technical tutorial creation.
- **Self-Evolving Agents**: Agents with reflective and memory-augmented abilities that can self-evolve.
- **Web UI**: Compatible with ChatPilot, supporting web-based interaction through popular front-end frameworks like open-webui, streamlit, and gradio.

## Install

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
python web_search_moonshot_demo.py
```

1. Copy the [.env.example](https://github.com/shibing624/agentica/blob/main/.env.example) file to `~/.agentica/.env` and fill in the LLM API key (optional DEEPSEEK_API_KEY, MOONSHOT_API_KEY, OPENAI_API_KEY, etc.). Or you can set the environment variable directly.
    ```shell
    export OPENAI_API_KEY=your_openai_api_key
    export SERPER_API_KEY=your_serper_api_key
    ```

2. Build and run an Agent using `agentica`:

Automatically call the Google search tool, example [examples/11_web_search_moonshot_openai.py](https://github.com/shibing624/agentica/blob/main/examples/11_web_search_moonshot_openai.py)

```python
from agentica import Agent, OpenAIChat, SearchSerperTool

m = Agent(model=OpenAIChat(id='gpt-4o'), tools=[SearchSerperTool()], add_datetime_to_instructions=True)
r = m.run("Where will the next Olympics be held?")
print(r)
```


## Web UI

[shibing624/ChatPilot](https://github.com/shibing624/ChatPilot) is compatible with `agentica` and can be interacted with through a Web UI.

Web Demo: https://chat.mulanai.com

<img src="https://github.com/shibing624/ChatPilot/blob/main/docs/shot.png" width="800" />

```shell
git clone https://github.com/shibing624/ChatPilot.git
cd ChatPilot
pip install -r requirements.txt

cp .env.example .env

bash start.sh
```


## Examples

| Example                                                                                                                                                    | Description                                                                                                                                |
|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| [examples/01_llm_demo.py](https://github.com/shibing624/agentica/blob/main/examples/01_llm_demo.py)                                                         | LLM Q&A Demo                                                                                                                               |
| [examples/02_user_prompt_demo.py](https://github.com/shibing624/agentica/blob/main/examples/02_user_prompt_demo.py)                                         | Custom user prompt Demo                                                                                                                    |
| [examples/03_user_messages_demo.py](https://github.com/shibing624/agentica/blob/main/examples/03_user_messages_demo.py)                                     | Custom input user messages Demo                                                                                                            |
| [examples/04_memory_demo.py](https://github.com/shibing624/agentica/blob/main/examples/04_memory_demo.py)                                                   | Agent memory Demo                                                                                                                          |
| [examples/05_response_model_demo.py](https://github.com/shibing624/agentica/blob/main/examples/05_response_model_demo.py)                                   | Demo of responding in a specified format (pydantic's BaseModel)                                                                            |
| [examples/06_calc_with_csv_file_demo.py](https://github.com/shibing624/agentica/blob/main/examples/06_calc_with_csv_file_demo.py)                           | Demo of LLM loading CSV files and performing calculations to answer questions                                                              |
| [examples/07_create_image_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/07_create_image_tool_demo.py)                             | Demo of creating an image tool                                                                                                             |
| [examples/08_ocr_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/08_ocr_tool_demo.py)                                               | Demo of implementing an OCR tool                                                                                                           |
| [examples/09_remove_image_background_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/09_remove_image_background_tool_demo.py)       | Demo of automatically removing image backgrounds, including automatic pip installation and calling the library to remove image backgrounds |
| [examples/10_vision_demo.py](https://github.com/shibing624/agentica/blob/main/examples/10_vision_demo.py)                                                   | Vision understanding Demo                                                                                                                  |
| [examples/11_web_search_openai_demo.py](https://github.com/shibing624/agentica/blob/main/examples/11_web_search_openai_demo.py)                             | Web search Demo based on OpenAI's function call                                                                                            |
| [examples/12_web_search_moonshot_demo.py](https://github.com/shibing624/agentica/blob/main/examples/12_web_search_moonshot_demo.py)                         | Web search Demo based on Moonshot's function call                                                                                          |
| [examples/13_storage_demo.py](https://github.com/shibing624/agentica/blob/main/examples/13_storage_demo.py)                                                 | Agent storage Demo                                                                                                                         |
| [examples/14_custom_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/14_custom_tool_demo.py)                                         | Demo of custom tools and autonomous selection and calling by large models                                                                  |
| [examples/15_crawl_webpage_demo.py](https://github.com/shibing624/agentica/blob/main/examples/15_crawl_webpage_demo.py)                                     | Demo of a webpage analysis workflow: crawling financing news from URLs, analyzing webpage content and format, extracting core information, and summarizing it into a markdown file |
| [examples/16_get_top_papers_demo.py](https://github.com/shibing624/agentica/blob/main/examples/16_get_top_papers_demo.py)                                   | Demo of parsing daily papers and saving them in JSON format                                                                                |
| [examples/17_find_paper_from_arxiv_demo.py](https://github.com/shibing624/agentica/blob/main/examples/17_find_paper_from_arxiv_demo.py)                     | Demo of paper recommendation: automatically searching multiple groups of papers from arxiv, deduplicating similar papers, extracting core paper information, and saving it as a CSV file |
| [examples/18_agent_input_is_list.py](https://github.com/shibing624/agentica/blob/main/examples/18_agent_input_is_list.py)                                   | Demo showing that the Agent's message can be a list                                                                                        |
| [examples/19_naive_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/19_naive_rag_demo.py)                                             | Basic RAG implementation, answering questions based on a text document                                                                     |
| [examples/20_advanced_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/20_advanced_rag_demo.py)                                       | Advanced RAG implementation, answering questions based on a PDF document, with new features: PDF file parsing, query rewriting, multi-path mixed recall (literal + semantic), and recall ranking (rerank) |
| [examples/21_memorydb_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/21_reference_in_prompt_rag_demo.py)                            | Traditional RAG approach of placing reference materials in the prompt                                                                      |
| [examples/22_chat_pdf_app_demo.py](https://github.com/shibing624/agentica/blob/main/examples/22_chat_pdf_app_demo.py)                                       | Demo of in-depth conversation with a PDF document                                                                                          |
| [examples/23_python_agent_memory_demo.py](https://github.com/shibing624/agentica/blob/main/examples/23_python_agent_memory_demo.py)                         | Demo of a Code Interpreter with memory, automatically generating and executing Python code, and retrieving results from memory on subsequent executions |
| [examples/24_context_demo.py](https://github.com/shibing624/agentica/blob/main/examples/24_context_demo.py)                                                 | Demo of conversation with context                                                                                                          |
| [examples/25_tools_with_context_demo.py](https://github.com/shibing624/agentica/blob/main/examples/25_tools_with_context_demo.py)                           | Demo of tools with context parameters                                                                                                      |
| [examples/26_complex_translate_demo.py](https://github.com/shibing624/agentica/blob/main/examples/26_complex_translate_demo.py)                             | Demo of complex translation                                                                                                                |
| [examples/27_research_agent_demo.py](https://github.com/shibing624/agentica/blob/main/examples/27_research_agent_demo.py)                                   | Demo of Research functionality, automatically calling search tools, summarizing information, and writing a scientific report               |
| [examples/28_rag_integrated_langchain_demo.py](https://github.com/shibing624/agentica/blob/main/examples/28_rag_integrated_langchain_demo.py)               | RAG Demo integrated with LangChain                                                                                                         |
| [examples/29_rag_integrated_llamaindex_demo.py](https://github.com/shibing624/agentica/blob/main/examples/29_rag_integrated_llamaindex_demo.py)             | RAG Demo integrated with LlamaIndex                                                                                                        |
| [examples/30_text_classification_demo.py](https://github.com/shibing624/agentica/blob/main/examples/30_text_classification_demo.py)                         | Demo of an Agent that automatically trains a classification model: reading the training set file and understanding the format, Google searching for the pytextclassifier library, crawling the GitHub page to understand how to call pytextclassifier, writing code and executing fasttext model training, and checking the prediction results of the trained model |
| [examples/31_team_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/31_team_news_article_demo.py)                             | Team implementation: team collaboration to write a news article, multi-role implementation, delegating different roles to complete their respective tasks: researcher retrieves and analyzes articles, writer writes the article according to the layout, and the results of multiple roles are summarized |
| [examples/32_team_debate_demo.py](https://github.com/shibing624/agentica/blob/main/examples/32_team_debate_demo.py)                                         | Team implementation: Demo of a two-person debate based on delegation, Trump and Biden debate                                               |
| [examples/33_self_evolving_agent_demo.py](https://github.com/shibing624/agentica/blob/main/examples/33_self_evolving_agent_demo.py)                         | Demo of a self-evolving Agent                                                                                                              |
| [examples/34_llm_os_demo.py](https://github.com/shibing624/agentica/blob/main/examples/34_llm_os_demo.py)                                                   | Initial design of LLM OS, based on LLM design operating system, can call RAG, code executor, Shell, etc. through LLM, and collaborate with code interpreter, research assistant, investment assistant, etc. to solve problems |
| [examples/35_workflow_investment_demo.py](https://github.com/shibing624/agentica/blob/main/examples/35_workflow_investment_demo.py)                         | Workflow implementation for investment research: stock information collection, stock analysis, writing analysis reports, reviewing reports, and multiple tasks |
| [examples/36_workflow_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/36_workflow_news_article_demo.py)                     | Workflow implementation for writing news articles, multi-agent implementation, multiple calls to search tools, and generating advanced layout news articles |
| [examples/37_workflow_write_novel_demo.py](https://github.com/shibing624/agentica/blob/main/examples/37_workflow_write_novel_demo.py)                       | Workflow implementation for writing novels: setting the novel outline, Google searching to reflect on the outline, writing the novel content, and saving it as a markdown file |
| [examples/38_workflow_write_tutorial_demo.py](https://github.com/shibing624/agentica/blob/main/examples/38_workflow_write_tutorial_demo.py)                 | Workflow implementation for writing technical tutorials: setting the tutorial directory, reflecting on the directory content, writing the tutorial content, and saving it as a markdown file |
| [examples/39_audio_multi_turn_demo.py](https://github.com/shibing624/agentica/blob/main/examples/39_audio_multi_turn_demo.py)                               | Demo of multi-turn audio conversation based on OpenAI's voice API                                                                          |

### Self-evolving Agent
The self-evolving agent design:

<img alt="sage" src="https://github.com/shibing624/agentica/blob/main/docs/sage_arch.png" width="800" />

#### Feature

Self-evolving Agents with Reflective and Memory-augmented Abilities (SAGE)

Implement:
1. Use `PythonAgent` as the SAGE agent and `AzureOpenAIChat` as the LLM, with code-interpreter functionality to execute Python code and automatically correct errors.
2. Use `CsvMemoryDb` as the memory for the SAGE agent to store user questions and answers, so that similar questions can be directly answered next time.

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
