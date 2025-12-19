[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# Agentica: 构建 AI 智能体
[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![MseeP.ai](https://img.shields.io/badge/mseep.ai-agentica-blue)](https://mseep.ai/app/shibing624-agentica)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#%E7%A4%BE%E5%8C%BA%E4%B8%8E%E6%94%AF%E6%8C%81)

**Agentica 是一个轻量级、功能强大的 Python 框架，用于构建、管理和部署自主 AI 智能体。**

无论您是想创建一个简单的聊天机器人、一个复杂的研究助理，还是一个由专业智能体组成的协作团队，Agentica 都能为您提供所需的工具和抽象，助您更快实现目标。我们采用开发者优先的设计方法，简化了 RAG、多智能体工作流和长期记忆等高级功能，让每一位开发者都能轻松上手。

## 🚀 为什么选择 Agentica？

*   **开发者优先的 API**：简洁、直观、面向对象的接口，易于学习，使用愉快。
*   **模块化与可扩展**：轻松替换 LLM、记忆后端和向量存储等组件。
*   **功能完备**：开箱即用，提供丰富的内置工具（网络搜索、代码解释器、文件读写）、记忆类型和高级 RAG 功能。
*   **高级功能，简化实现**：轻松实现多智能体协作（团队）、任务分解（工作流）和自我反思等复杂模式。
*   **生产就绪**：通过命令行界面、Web UI 或作为服务部署您的智能体。同时支持**模型上下文协议（MCP）**，实现标准化工具集成。
*   **Agent Skill 支持**：基于 Prompt Engineering 的技能系统，将技能说明注入 System Prompt，任何支持 tool calling 的模型都可使用。

## ✨ 核心特性

*   **🤖 核心智能体能力**：构建具备复杂规划、反思、短期和长期记忆以及强大工具使用能力的智能体。
*   **🧩 高级编排**：
    *   **多智能体团队**：创建由专业智能体组成的团队，协作解决问题。
    *   **工作流**：将复杂任务分解为一系列步骤，由不同的智能体或工具执行。
*   **🛡️ 安全守卫（Guardrails）**：
    *   **输入/输出守卫**：在智能体处理前验证用户输入，在返回前检查智能体输出。
    *   **工具守卫**：在工具执行前验证参数，在返回结果前过滤敏感数据。
    *   **三种行为模式**：允许（allow）、拒绝并继续（reject_content）、抛出异常（raise_exception）。
*   **🎯 Agent Skill 技能系统**：
    *   **Prompt Engineering 技术**：Skill 是一种文本指令，不是代码级别的能力扩展。
    *   **实现方式**：解析 SKILL.md 的元数据，将技能描述注入 System Prompt。
    *   **执行流程**：LLM 阅读技能说明后，使用基础 tools（shell、python、file viewer）执行任务。
    *   **模型无关**：任何支持 tool calling 的模型都可以使用，因为 skill 只是文本指令。
    *   **优势**：可扩展、模型无关、易于维护（只需更新 Markdown 文档）。
*   **🛠️ 丰富的工具生态**：
    *   大量内置工具（网络搜索、OCR、图像生成、Shell 命令）。
    *   轻松创建您自己的自定义工具。
    *   一流的**模型上下文协议（MCP）**支持，实现标准化工具集成。
*   **📚 灵活的 RAG 流程**：
    *   内置知识库管理和文档解析（PDF、文本）。
    *   混合检索策略和结果重排序，以实现最高准确性。
    *   与 LangChain 和 LlamaIndex 等流行库集成。
*   **🌌 多模态支持**：构建能够理解和生成文本、图像、音频和视频的智能体。
*   **🧠 广泛的 LLM 兼容性**：支持来自 OpenAI、Azure、Deepseek、Moonshot、Anthropic、智谱AI、Ollama、Together 等提供商的数十种模型。
*   **💡 自我进化智能体**：具备反思和记忆增强能力的智能体，能够自我进化。

## 🏗️ 系统架构

<div align="center">
    <img src="https://github.com/shibing624/agentica/blob/main/docs/agentica_architecture.png" alt="Agentica Architecture" width="800"/>
</div>

Agentica 的模块化设计实现了最大的灵活性和可扩展性。其核心是 `Agent`、`Model`、`Tool` 和 `Memory` 组件，这些组件可以轻松组合和扩展，以创建强大的应用程序。

## 💾 安装

```bash
pip install -U agentica
```

从源码安装：
```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install .
```

## ⚡ 快速入门

1.  **设置您的 API 密钥。** 在 `~/.agentica/.env` 路径下创建一个文件，或设置环境变量。

    ```shell
    # 智谱AI ZhipuAI, glm-4.6v-flash 免费用，支持工具调用，128k
    export ZHIPUAI_API_KEY="your-api-key"
    ```

2.  **运行您的第一个智能体！** 这个例子创建了一个可以查询天气的智能体。

    ```python
    from agentica import Agent, ZhipuAI, WeatherTool

    # 初始化一个带模型和天气工具的智能体
    agent = Agent(
        model=ZhipuAI(),
        tools=[WeatherTool()],
        # 为智能体提供时间概念，以便回答“明天”等问题
        add_datetime_to_instructions=True  
    )

    # 向智能体提问
    agent.print_response("明天北京天气怎么样？")
    ```

    **输出：**
    ```markdown
    明天北京的天气预报如下：

    - 早晨：晴朗，气温约18°C，风速较小，约为3 km/h。
    - 中午：晴朗，气温升至23°C，风速6-7 km/h。
    - 傍晚：晴朗，气温略降至21°C，风速较大，为35-44 km/h。
    - 夜晚：晴朗转晴，气温下降至15°C，风速32-39 km/h。

    全天无降水，能见度良好。请注意傍晚时分的风速较大，外出时需注意安全。
    ```

## 📖 核心概念

*   **Agent**：思考、决策和执行动作的核心组件。它将模型、工具和记忆连接在一起。
*   **Model**：智能体的“大脑”。通常是一个大型语言模型（LLM），为智能体的推理能力提供动力。
*   **Tool**：智能体可用于与外部世界交互的功能或能力（例如，搜索网页、运行代码、访问数据库）。
*   **Memory**：允许智能体记住过去的交互（短期记忆）并存储关键信息以供日后调用（长期记忆）。
*   **Knowledge**：外部知识源（如文档集合），智能体可以使用检索增强生成（RAG）进行查询。
*   **Workflow/Team**：用于编排复杂、多步骤任务或管理多个智能体之间协作的高级结构。

## 🚀 功能展示：您可以构建什么

浏览我们全面的示例，了解 Agentica 的无限可能：

| 示例                                                                                                                                                    | 描述                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [**高级 RAG 智能体**](https://github.com/shibing624/agentica/blob/main/examples/20_advanced_rag_demo.py)                                 | 基于您的 PDF 文档构建一个强大的问答系统，具备查询重写、混合检索和重排序功能。                                                               |
| [**多智能体团队**](https://github.com/shibing624/agentica/blob/main/examples/31_team_news_article_demo.py)                       | 组建一个由专业智能体（如研究员和作家）组成的团队，协作撰写新闻文章。                                                     |
| [**自我进化智能体**](https://github.com/shibing624/agentica/blob/main/examples/33_self_evolving_agent_demo.py)                   | 创建一个能从交互中学习并随时间推移改进其知识库的智能体。                                                                                                                 |
| [**LLM 操作系统**](https://github.com/shibing624/agentica/blob/main/examples/34_llm_os_demo.py)                                             | 一个有趣的实验，旨在构建一个由 LLM 驱动的对话式操作系统。                                                  |
| [**投资研究工作流**](https://github.com/shibing624/agentica/blob/main/examples/35_workflow_investment_demo.py)                   | 自动化整个投资研究流程，从数据收集和分析到报告生成。                                                                                                   |
| [**视觉智能体**](https://github.com/shibing624/agentica/blob/main/examples/10_vision_demo.py)                                             | 构建一个能够理解和推理图像的智能体。                                                                                                                          |
| [**安全Guardrails**](https://github.com/shibing624/agentica/blob/main/examples/52_guardrails_demo.py)                                             | 演示如何使用输入/输出守卫验证智能体和工具的输入输出，过滤敏感数据。                                                                                                                          |

[➡️ **查看所有示例**](https://github.com/shibing624/agentica/tree/main/examples)

## 🖥️ 部署

### 命令行界面 (CLI)

直接从终端与您的智能体互动。

```shell
# 安装 agentica
pip install -U agentica

# 运行单个查询
agentica --query "下一届奥运会在哪里举办？" --model_provider zhipuai --model_name glm-4.6v-flash --tools baidu_search

# 启动交互式聊天会话
agentica --model_provider zhipuai --model_name glm-4.6v-flash
```

CLI show case (实现ClaudeCode效果):
<img src="https://github.com/shibing624/agentica/blob/main/docs/cli_snap.png" width="800" />

### Web UI

Agentica 与 [ChatPilot](https://github.com/shibing624/ChatPilot) 完全兼容，为您的智能体提供功能丰富、基于 Gradio 的 Web 界面。

<div align="center">
    <img src="https://github.com/shibing624/ChatPilot/blob/main/docs/shot.png" width="800" />
</div>

请查看 [ChatPilot 仓库](https://github.com/shibing624/ChatPilot)了解设置说明。

## 🤝 与其他框架的比较

| 特性                | Agentica                                   | LangChain                                 | AutoGen                             | CrewAI                             |
|------------------------|--------------------------------------------|-------------------------------------------|-------------------------------------|------------------------------------|
| **核心设计**        | 以智能体为中心，模块化且直观      | 以链为中心，复杂的组件图    | 专注于多智能体对话    | 专注于基于角色的多智能体       |
| **易用性**        | 高（为简洁而设计）             | 中（学习曲线陡峭）           | 中                            | 高                               |
| **多智能体**        | 原生支持 `Team` 和 `Workflow`         | 需要自定义实现            | 核心功能                        | 核心功能                       |
| **RAG**                | 内置高级流程                | 需要手动组装组件    | 需要外部集成       | 需要外部集成      |
| **工具**            | 丰富的内置工具 + MCP 支持          | 生态系统庞大，可能很复杂         | 基本的工具支持                  | 基本的工具支持                 |
| **多模态**        | ✅ 支持（文本、图像、音频、视频）         | ✅ 支持（但集成可能复杂）  | ❌ 不支持（主要基于文本）      | ❌ 不支持（主要基于文本）     |


## 💬 社区与支持

*   **GitHub Issues**：有任何问题或功能请求？[提交 issue](https://github.com/shibing624/agentica/issues)。
*   **微信**：加入我们的开发者社群！添加微信号 `xuming624`，并备注“agentica”，即可加入群聊。

<img src="https://github.com/shibing624/agentica/blob/main/docs/wechat.jpeg" width="200" />

## 📜 引用

如果您在研究中使用了 Agentica，请按以下格式引用：

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

## 📄 许可证

Agentica 采用 [Apache License 2.0](LICENSE) 授权。

## ❤️ 贡献

我们欢迎各种形式的贡献！请查看我们的[贡献指南](CONTRIBUTING.md)以开始。

## 🙏 致谢

我们的工作受到了许多优秀项目的启发和帮助。我们在此感谢以下项目团队：
- [langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [simonmesmith/agentflow](https://github.com/simonmesmith/agentflow)
