---
name: Analyzing Agentica Library
description: This skill provides a way to retrieve information from the Agentica library for analysis and decision-making.
---

# Analyzing Agentica Library

## Overview

This guide covers the essential operations for retrieving and answering questions about the Agentica library.
If you need to answer questions regarding the Agentica library, or look up specific information, functions/classes,
examples or guidance, this skill will help you achieve that.

## Quick Start

The skill provides the following key scripts:

- Search for guidance in the Agentica documentation and examples.
- Search for official examples and recommended implementations provided by Agentica.
- A quick interface to view Agentica's Python library by given a module name (e.g. agentica), and return the module's submodules, classes, and functions.

When being asked an Agentica-related question, you can follow the steps below to find the relevant information:

First decide which of the three scripts to use based on the user's question.
- If user asks for "how to use" types of questions, use the "Search for Guidance" script to find the relevant tutorial
- If user asks for "how to implement/build" types of questions, first search for relevant examples. If not found, then
  consider what functions are needed and search in the guide/tutorial
- If user asks for "how to initialize" types of questions, first search for relevant tutorials. If not found, then
  consider to search for the corresponding modules, classes, or functions in the library.


### Search for Examples

First ask for the user's permission to clone the agentica GitHub repository if you haven't done so:

```bash
git clone -b main https://github.com/shibing624/agentica
```

In this repo, the `examples` folder contains various examples demonstrating how to use different features of the
Agentica library.
They are organized by different functionalities. You should use shell command like `ls` or `cat` to
navigate and view the examples. Avoid using `find` command to search for examples, as the name of the example
files may not directly relate to the functionality being searched for.

### Search for Guidance

Similarly, first ensure you have cloned the agentica GitHub repository.

The source agentica documentation is located in the `docs` folder of the agentica GitHub repository.
To search for guidance, go to the `docs` folder and view the documentation files by shell command like `ls` or `cat`.

The main README.md file in the root directory also contains comprehensive usage instructions.


### Search for Targeted Modules

First, ensure you have installed the agentica library in your environment:

```bash
pip list | grep agentica
```

If not installed, ask the user for permission to install it by command:

```bash
pip install agentica
```

Then, run the following script to search for specific modules, classes, or functions. It's suggested to start with
`agentica` as the root module name, and then specify the submodule name you want to search for.

```bash
python view_agentica_module.py --module agentica
```

About detailed usage, please refer to the `./view_agentica_module.py` script (located in the same folder as this
SKILL.md file).

### Key Modules in Agentica

Here are the main modules you might want to explore:

- `agentica.agent`: The core Agent class for building AI agents
- `agentica.model`: Various LLM model implementations (OpenAI, DeepSeek, Qwen, etc.)
- `agentica.tools`: Built-in tools for agents (FileTool, ShellTool, SearchTool, etc.)
- `agentica.memory`: Memory management for agents
- `agentica.knowledge`: Knowledge base and RAG implementations
- `agentica.workflow`: Workflow orchestration for multi-agent systems
- `agentica.mcp`: MCP (Model Context Protocol) support
