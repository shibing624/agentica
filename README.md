[**🇨🇳中文**](https://github.com/shibing624/actionflow/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/actionflow/blob/main/README_EN.md)

<div align="center">
  <a href="https://github.com/shibing624/actionflow">
    <img src="https://raw.githubusercontent.com/shibing624/actionflow/main/docs/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# ActionFlow: LLMs Agent Workflows with Prompts and Tools
[![PyPI version](https://badge.fury.io/py/actionflow.svg)](https://badge.fury.io/py/actionflow)
[![Downloads](https://static.pepy.tech/badge/actionflow)](https://pepy.tech/project/actionflow)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/actionflow.svg)](https://github.com/shibing624/actionflow/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)


**actionflow**: A Human-Centric Framework for Large Language Model Agent Workflows

`ActionFlow`是一个LLMs驱动的工作流构建工具，支持如下功能：

- 通过自然语言（prompt）在`json`文件中编排复杂工作流
- 工作流的编排不仅支持多个prompt命令，还支持工具调用（tool_calls）
- 基于变量名动态更改prompt输入
- 支持OpenAI API和Moonshot API(kimi)调用

## Install

```
pip install -U actionflow
```

or

```
git clone https://github.com/shibing624/actionflow.git
cd actionflow
pip install -e .
```

## Usage

Create a `.env` file from [example.env](https://github.com/shibing624/actionflow/blob/main/example.env) and add your OpenAI API key.

Now you can run flows from the command line, like this:
```bash
cd examples
python run_flow_demo.py --flow_path flows/example.json
```
### Optional Arguments

#### Use `variables` to pass variables to your flow

```bash
python run_flow_demo.py --flow_path flows/example_with_variables.json --variables 'market=college students' 'price_point=$50'
```


## Create New Flows

Copy [examples/flows/example.json](https://github.com/shibing624/actionflow/blob/main/examples/flows/example.json) or create a flow from scratch in this format:

```json
{
    "system_message": "An optional message that guides the model's behavior.",
    "tasks": [
        {
            "action": "Instruct the LLM here!"
        },
        {
            "action": "Actions can have settings, including function calls and temperature, like so:",
            "settings": {
                "tool_name": "save_file",
                "temperature": 0.8
            }
        },
        {
            "action": "..."
        }
    ]
}
```

## Create New Tools

Copy [save_file.py](https://github.com/shibing624/actionflow/blob/main/actionflow/tools/save_file.py) and modify it, or follow these instructions (replace "tool_name" with your tool name):

1. **Create `tool_name.py` in the [actionflow/tools](https://github.com/shibing624/actionflow/tree/main/actionflow/tools) folder**.
2. **Create a class within called `ToolName`** that inherits from `BaseTool`.
3. **Add `get_definition()` and `execute()` in the class**. See descriptions of these in `BaseTool`.

That's it! You can now use your function in `tool_name` as shown above. 

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

Thanks for their great work!
