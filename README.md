[**🇨🇳中文**](https://github.com/shibing624/actionflow/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/actionflow/blob/main/README_EN.md)

<div align="center">
  <a href="https://github.com/shibing624/actionflow">
    <img src="https://raw.githubusercontent.com/shibing624/actionflow/main/docs/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# ActionFlow: Agent Workflows with Prompts and Tools
[![PyPI version](https://badge.fury.io/py/actionflow.svg)](https://badge.fury.io/py/actionflow)
[![Downloads](https://static.pepy.tech/badge/actionflow)](https://pepy.tech/project/actionflow)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/actionflow.svg)](https://github.com/shibing624/actionflow/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)


**actionflow**: A Human-Centric Framework for Large Language Model Agent Workflows, build your agent workflows quickly

**actionflow**: 快速构建你自己的Agent工作流

`ActionFlow`是一个Agent工作流构建工具，功能：

- 通过自然语言（prompt）在`json`文件中编排复杂工作流
- 工作流的编排不仅支持prompt命令，还支持工具调用（tool_calls）
- 基于变量名动态更改prompt输入
- 支持OpenAI API和Moonshot API(kimi)调用

## Install

```bash
pip install -U actionflow
```

or

```bash
git clone https://github.com/shibing624/actionflow.git
cd actionflow
pip install -e .
```

## Usage

1. 复制[example.env](https://github.com/shibing624/actionflow/blob/main/example.env)文件为`.env`，并粘贴OpenAI API key或者Moonshoot API key。

2. 运行actionflow示例：

```bash
cd examples
python run_flow_demo.py --flow_path flows/example.json
```
### 可选参数

#### 使用`variables`参数

```bash
python run_flow_demo.py --flow_path flows/example_with_variables.json --variables 'market=college students' 'price_point=$50'
```


## 新建工作流（ActionFlow）

复制 [examples/flows/example.json](https://github.com/shibing624/actionflow/blob/main/examples/flows/example.json) 或者按照如下格式创建一个工作流（json文件）：

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

## 新建工具（Tools）

复制 [actionflow/tools/save_file.py](https://github.com/shibing624/actionflow/blob/main/actionflow/tools/save_file.py) 并修改，或者按如下指引新增一个工具（记得替换`tool_name`为你的工具名）：
1. **在[actionflow/tools](https://github.com/shibing624/actionflow/tree/main/actionflow/tools)文件夹新增`tool_name.py`**
2. **新建类`ToolName`** 继承自`BaseTool`
3. **在类中新增`get_definition()`和`execute()`方法**，具体参考`BaseTool`

这样，你就可以在工作流中使用新增的`tool_name`工具。 

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
