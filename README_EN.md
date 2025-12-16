[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# Agentica: Build AI Agent
[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![MseeP.ai](https://img.shields.io/badge/mseep.ai-agentica-blue)](https://mseep.ai/app/shibing624-agentica)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)

**Agentica is a lightweight and powerful Python framework for building, managing, and deploying autonomous AI agent.**

Whether you're creating a simple chatbot, a complex research assistant, or a collaborative team of specialized agents, Agentica provides the tools and abstractions to get you there faster. Our developer-first approach simplifies advanced features like RAG, multi-agent workflows, and long-term memory, making them accessible to everyone.

## 🚀 Why Agentica?

*   **Developer-First API**: A simple, intuitive, and object-oriented interface that's easy to learn and a joy to use.
*   **Modular & Extensible**: Swap components like LLMs, memory backends, and vector stores with ease.
*   **Batteries-Included**: Comes with a rich set of built-in tools (Web Search, Code Interpreter, File I/O), memory types, and advanced RAG capabilities out of the box.
*   **Advanced Features, Simplified**: Effortlessly implement complex patterns like multi-agent collaboration (Teams), task decomposition (Workflows), and self-reflection.
*   **Production-Ready**: Deploy your agents via a command-line interface, a web UI, or as a service. Standardized tool integration with the **Model Context Protocol (MCP)** is also supported.
*   **Agent Skill Support**: A prompt-based skill system that injects skill instructions into the System Prompt, usable by any model supporting tool calling.

## ✨ Key Features

*   **🤖 Core Agent Capabilities**: Build agents with sophisticated planning, reflection, short-term, and long-term memory, and robust tool-use abilities.
*   **🧩 Advanced Orchestration**:
    *   **Multi-Agent Teams**: Create teams of specialized agents that collaborate to solve problems.
    *   **Workflows**: Decompose complex tasks into a series of steps executed by different agents or tools.
*   **🛡️ Guardrails**:
    *   **Input/Output Guardrails**: Validate user input before agent processing and check agent output before returning.
    *   **Tool Guardrails**: Validate tool arguments before execution and filter sensitive data from results.
    *   **Three Behavior Modes**: allow, reject_content (reject but continue), raise_exception (halt execution).
*   **🎯 Agent Skill System**:
    *   **Prompt Engineering Technique**: Skills are text instructions, not code-level capability extensions.
    *   **Implementation**: Parse SKILL.md metadata and inject skill descriptions into the System Prompt.
    *   **Execution**: After reading skill instructions, the LLM uses basic tools (shell, python, file viewer) to execute tasks.
    *   **Model-Agnostic**: Any model supporting tool calling can use skills, as they are just text instructions.
    *   **Advantages**: Extensible, model-agnostic, easy to maintain (just update Markdown documents).
*   **🛠️ Rich Tooling Ecosystem**:
    *   An extensive collection of built-in tools (Web Search, OCR, Image Generation, Shell Commands).
    *   Easily create your own custom tools.
    *   First-class support for the **Model Context Protocol (MCP)** for standardized tool integration.
*   **📚 Flexible RAG Pipeline**:
    *   Built-in knowledge base management and document parsing (PDFs, text).
    *   Hybrid retrieval strategies and result reranking for maximum accuracy.
    *   Integrations with popular libraries like LangChain and LlamaIndex.
*   **🌌 Multi-Modal Support**: Build agents that can understand and generate text, images, audio, and video.
*   **🧠 Broad LLM Compatibility**: Works with dozens of models from providers like OpenAI, Azure, Deepseek, Moonshot, Anthropic, ZhipuAI, Ollama, Together, and more.
*   ** Evolvable Agents**: Agents with reflection and memory augmentation capabilities that can evolve on their own.

## 🏗️ System Architecture

<div align="center">
    <img src="https://github.com/shibing624/agentica/blob/main/docs/agentica_architecture.png" alt="Agentica Architecture" width="800"/>
</div>

Agentica's modular design allows for maximum flexibility and scalability. At its core are the `Agent`, `Model`, `Tool`, and `Memory` components, which can be easily combined and extended to create powerful applications.

## 💾 Installation

```bash
pip install -U agentica
```

To install from source:
```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install .
```

## ⚡ Quick Start

1.  **Set up your API keys.** Create a file at `~/.agentica/.env` or set environment variables.

    ```shell
    # For Moonshot AI
    export MOONSHOT_API_KEY="your-api-key"
    # For ZhipuAI
    export ZHIPUAI_API_KEY="your-api-key"
    # For Search tools
    export SERPER_API_KEY="your-serper-api-key"
    ```

2.  **Run your first agent!** This example creates an agent that can check the weather.

```python
    from agentica import Agent, Moonshot, WeatherTool

    # Initialize an agent with a model and a weather tool
    weather_agent = Agent(
        model=Moonshot(), 
        tools=[WeatherTool()],
        # Give the agent a sense of time for questions like "tomorrow"
        add_datetime_to_instructions=True  
    )

    # Ask the agent a question
    weather_agent.print_response("What's the weather like in Beijing tomorrow?")
    ```

    **Output:**
    ```markdown
    The weather forecast for Beijing tomorrow is as follows:

    - Morning: Clear, temperature around 18°C, light wind at 3 km/h.
    - Noon: Clear, temperature rises to 23°C, wind at 6-7 km/h.
    - Evening: Clear, temperature drops slightly to 21°C, strong wind at 35-44 km/h.
    - Night: Clear, temperature drops to 15°C, wind at 32-39 km/h.

    No precipitation is expected throughout the day, and visibility is good. Please be mindful of the strong winds in the evening.
    ```

## 📖 Core Concepts

*   **Agent**: The central component that thinks, makes decisions, and performs actions. It ties together the model, tools, and memory.
*   **Model**: The "brain" of the agent. It's typically a Large Language Model (LLM) that powers the agent's reasoning capabilities.
*   **Tool**: A function or capability the agent can use to interact with the outside world (e.g., search the web, run code, access a database).
*   **Memory**: Allows the agent to remember past interactions (short-term) and store critical information for later recall (long-term).
*   **Knowledge**: An external knowledge source (like a collection of documents) that the agent can query using Retrieval-Augmented Generation (RAG).
*   **Workflow/Team**: High-level constructs for orchestrating complex, multi-step tasks or managing collaboration between multiple agents.

## 🚀 Showcase: What You Can Build

Explore our comprehensive examples to see what's possible with Agentica:

| Example                                                                                                                                                    | Description                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [**Advanced RAG Agent**](https://github.com/shibing624/agentica/blob/main/examples/20_advanced_rag_demo.py)                                 | Build a powerful Q&A system over your PDF documents with query rewriting, hybrid retrieval, and reranking.                                                               |
| [**Multi-Agent Team**](https://github.com/shibing624/agentica/blob/main/examples/31_team_news_article_demo.py)                       | Assemble a team of agents (e.g., a researcher and a writer) to collaborate on writing a news article.                                                     |
| [**Self-Evolving Agent**](https://github.com/shibing624/agentica/blob/main/examples/33_self_evolving_agent_demo.py)                   | Create an agent that learns from its interactions and improves its knowledge base over time.                                                                                                                 |
| [**LLM OS**](https://github.com/shibing624/agentica/blob/main/examples/34_llm_os_demo.py)                                             | A fascinating experiment in building a conversational operating system powered by an LLM.                                                  |
| [**Investment Research Workflow**](https://github.com/shibing624/agentica/blob/main/examples/35_workflow_investment_demo.py)                   | Automate the entire investment research process, from data collection and analysis to report generation.                                                                                                   |
| [**Vision Agent**](https://github.com/shibing624/agentica/blob/main/examples/10_vision_demo.py)                                             | Build an agent that can understand and reason about images.                                                                                                                          |
| [**Guardrails**](https://github.com/shibing624/agentica/blob/main/examples/52_guardrails_demo.py)                                             | Demonstrate how to use input/output guardrails to validate agent and tool I/O, and filter sensitive data.                                                                                                                          |

[➡️ **See all examples**](https://github.com/shibing624/agentica/tree/main/examples)

## 🖥️ Deployment

### Command Line Interface (CLI)

Interact with your agents directly from the terminal.

```shell
# Install agentica
pip install -U agentica

# Run a single query
agentica --query "Where will the next Olympics be held?" --model_provider zhipuai --model_name glm-4-flash --tools web_search_pro

# Start an interactive chat session
agentica --model_provider zhipuai --model_name glm-4-flash --tools web_search_pro cogview --verbose 1
```

### Web UI

Agentica is fully compatible with [ChatPilot](https://github.com/shibing624/ChatPilot), giving you a feature-rich, Gradio-based web interface for your agents.

<div align="center">
    <img src="https://github.com/shibing624/ChatPilot/blob/main/docs/shot.png" width="800" />
</div>

Check out the [ChatPilot repository](https://github.com/shibing624/ChatPilot) for setup instructions.

## 🤝 Comparison with Other Frameworks

| Feature                | Agentica                                   | LangChain                                 | AutoGen                             | CrewAI                             |
|------------------------|--------------------------------------------|-------------------------------------------|-------------------------------------|------------------------------------|
| **Core Design**        | Agent-centric, modular, and intuitive      | Chain-centric, complex component graph    | Multi-agent conversation-focused    | Role-based multi-agent focus       |
| **Ease of Use**        | High (designed for simplicity)             | Moderate (steep learning curve)           | Moderate                            | High                               |
| **Multi-Agent**        | Native `Team` and `Workflow` support         | Requires custom implementation            | Core feature                        | Core feature                       |
| **RAG**                | Built-in, advanced pipeline                | Requires manual assembly of components    | Requires external integration       | Requires external integration      |
| **Tooling**            | Rich built-in tools + MCP support          | Large ecosystem, can be complex         | Basic tool support                  | Basic tool support                 |
| **Multi-Modal**        | ✅ Yes (Text, Image, Audio, Video)         | ✅ Yes (but can be complex to integrate)  | ❌ No (Primarily text-based)      | ❌ No (Primarily text-based)     |


## 💬 Community & Support

*   **GitHub Issues**: Have a question or a feature request? [Open an issue](https://github.com/shibing624/agentica/issues).
*   **WeChat**: Join our developer community! Add `xuming624` on WeChat and mention "agentica" to be added to the group.

<img src="https://github.com/shibing624/agentica/blob/main/docs/wechat.jpeg" width="200" />

## 📜 Citation

If you use Agentica in your research, please cite it as follows:

```bibtex
@misc{agentica,
  author = {Ming Xu},
  title = {Agentica: Effortlessly Build Intelligent, Reflective, and Collaborative Multimodal AI Agents},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/shibing624/agentica}},
}
```

## 📄 License

Agentica is licensed under the [Apache License 2.0](LICENSE).

## ❤️ Contributing

We welcome contributions of all kinds! Please check out our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## 🙏 Acknowledgements

Our work is inspired by and builds upon the shoulders of giants. We'd like to thank the teams behind these amazing projects:
- [langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [simonmesmith/agentflow](https://github.com/simonmesmith/agentflow)
