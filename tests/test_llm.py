# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains tests for the LLM class.
"""

from unittest.mock import MagicMock, patch
from openai import OpenAI
import os
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