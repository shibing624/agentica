# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

usage：
from openai import OpenAI

def send_messages(messages):
    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=messages,
        tools=tools
    )
    return response.choices[0].message

client = OpenAI(
    api_key="<your api key>",
    base_url="https://api.deepseek.com",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather of an location, the user shoud supply a location first",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"]
            },
        }
    },
]

messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
message = send_messages(messages)
print(f"User>\t {messages[0]['content']}")

tool = message.tool_calls[0]
messages.append(message)

messages.append({"role": "tool", "tool_call_id": tool.id, "content": "24℃"})
message = send_messages(messages)
print(f"Model>\t {message.content}")

"""
from os import getenv
from typing import Optional, Dict, Any

from openai import OpenAI as OpenAIClient, AsyncOpenAI as AsyncOpenAIClient

from agentica.llm.openai_chat import OpenAIChat


class Deepseek(OpenAIChat):
    name: str = "Deepseek"
    model: str = "deepseek-coder"
    api_key: Optional[str] = getenv("DEEPSEEK_API_KEY")
    base_url: str = "https://api.deepseek.com/v1"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    request_params: Optional[Dict[str, Any]] = None
    client_params: Optional[Dict[str, Any]] = None
    # -*- Provide the client manually
    client: Optional[OpenAIClient] = None
    async_client: Optional[AsyncOpenAIClient] = None
