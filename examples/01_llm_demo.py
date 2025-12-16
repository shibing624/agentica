# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: LLM demo, demonstrates how to use different LLM models with streaming response
"""
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Message, Yi, AzureOpenAIChat, DeepSeek, OpenAIChat, Moonshot, ZhipuAI


def get_model(model_name):
    models = {
        "AzureOpenAIChat": AzureOpenAIChat,
        "OpenAIChat": OpenAIChat,
        "DeepSeek": DeepSeek,
        "Yi": Yi,
        "Moonshot": Moonshot,
        "ZhipuAI": ZhipuAI,
        # Add more models here，eg: "Ollama": Ollama
    }
    return models.get(model_name, AzureOpenAIChat)()


def main():
    parser = argparse.ArgumentParser(description="LLM Demo")
    parser.add_argument('--model', type=str, default='ZhipuAI', help='Model name to use')
    parser.add_argument('--query', type=str, default='一句话介绍林黛玉', help='Query to send to the model')
    args = parser.parse_args()

    model = get_model(args.model)
    print(model)
    messages = [Message(role="user", content=args.query)]
    response = model.response_stream(messages)
    for i in response:
        if i.content:
            print(i.content, end='', flush=True)


if __name__ == "__main__":
    main()
