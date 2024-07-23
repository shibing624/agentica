# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains tests for the LLM class.
"""

import os
from unittest.mock import MagicMock, patch

from openai import OpenAI

from agentica.config import DOTENV_PATH  # noqa
from agentica.message import Message

api_key = os.getenv("MOONSHOT_API_KEY")


@patch("agentica.llm.openai_llm.OpenAILLM")
def test_respond(mock_llm_class):
    """
    Tests the response method of the LLM class.

    The LLM class and its response method are mocked.
    """
    mock_llm_instance = MagicMock()
    mock_llm_instance.response.return_value = "Yes, I am here!"
    mock_llm_class.return_value = mock_llm_instance

    llm = mock_llm_class()
    user_message = Message(role="user", content="This is a test. Are you there?")
    messages = [user_message]
    res = llm.response(messages)
    print('res:', res)
    assert res == "Yes, I am here!", "Response is not as expected"
    assert res is not None, "Response is None"
    mock_llm_instance.response.assert_called_once_with(messages)


def test_use_kimi_tool():
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )

    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system",
             "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
            {"role": "user", "content": "编程判断 3214567 是否是素数。"}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "CodeRunner",
                "description": "代码执行器，支持运行 python 和 javascript 代码",
                "parameters": {
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript"]
                        },
                        "code": {
                            "type": "string",
                            "description": "代码写在这里"
                        }
                    },
                    "type": "object"
                }
            }
        }],
        tool_choice='auto',
        temperature=0.3,
    )

    print(completion.choices[0].message)


def test_use_kimi_tool_and_resp():
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )

    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system",
             "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
            {"role": "user", "content": " 3214567 是否是素数?"},
            {"role": "assistant", "content": " 3214567 是素数"},
            {"role": "user", "content": " 1 + 3 =?"},

        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "CodeRunner",
                "description": "代码执行器，支持运行 python 和 javascript 代码",
                "parameters": {
                    "properties": {
                        "language": {
                            "type": "string",
                            "enum": ["python", "javascript"]
                        },
                        "code": {
                            "type": "string",
                            "description": "代码写在这里"
                        }
                    },
                    "type": "object"
                }
            }
        }],
        temperature=0.3,
    )

    print(completion.choices[0].message)

    mm = [
        {'role': 'system',
         'content': 'You must follow these instructions carefully:\n<instructions>\n1. The current time is 2024-07-05 23:54:34.272742\n</instructions>'},
        {'role': 'user', 'content': '北京最近的新闻'},
        {'role': 'assistant', 'content': 'Running tool calls...'},
        {'role': 'user',
         'content': "<tool_results><result><tool_name>search_google</tool_name><stdout>7月起北京300家药店可使用医保个人账户线上购药 · 首都医科大学宣武医院党委：“生命之舟”守护“未来之城” · 报告：中国医疗健康产业已进入高速增长期 · 北京首个社区医学专家 ...>"}
    ]

    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=mm,
        temperature=0.3,
    )

    print(completion.choices[0].message)


def test_kimi_team_delegate():
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )
    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system",
             "content": "You must follow these instructions carefully:\n<instructions>\n1. you should closely respond to your opponent's latest argument, state your position, defend your arguments, and attack your opponent's arguments, craft a strong and emotional response in 80 words\n2. The current time is 2024-07-23 16:48:20.271048\n</instructions>\n\nYou can delegate tasks to the following assistants:\n<assistants>\nAssistant 1:\nName: Biden\n\nAssistant 2:\nName: Trump\n</assistants>"},
            {"role": "user",
             "content": "Trump and Biden are in a debate, Biden speak first, and then Trump speak, and then Biden speak, and so on, in 3 turns.\n    Now begin. 请调用delegate_task_to_biden发起对话"},

        ],
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'delegate_task_to_biden',
                    'description': 'Use this function to delegate a task to biden\n        Args:\n            task_description (str): A clear and concise description of the task the assistant should achieve.\n        Returns:\n            str: The result of the delegated task.\n',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_description': {'type': 'string'}}}
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'delegate_task_to_trump',
                    'description': 'Use this function to delegate a task to trump\n        Args:\n            task_description (str): A clear and concise description of the task the assistant should achieve.\n        Returns:\n            str: The result of the delegated task.\n',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task_description': {'type': 'string'}}}
                }
            }
        ],
        tool_choice='auto',
        temperature=0.3,
    )

    print(completion.choices[0].message)
