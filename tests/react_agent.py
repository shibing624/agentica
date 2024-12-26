# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from loguru import logger
import re
from openai import OpenAI

action_registry = {}


def register_action(action_name):
    def decorator(func):
        action_registry[action_name] = func
        return func

    return decorator


def parse_response(response):
    thoughts = []
    actions = []
    lines = response.split('\n')
    for line in lines:
        if line.startswith("思考:"):
            thoughts.append(line[len("思考:"):].strip())
        elif line.startswith("动作:"):
            actions.append(line[len("动作:"):].strip())
    return thoughts, actions


def parse_action(action_str):
    match = re.match(r'^(\w+)\((.*)\)$', action_str)
    if match:
        action_name = match.group(1)
        params = match.group(2)
        return action_name, params
    else:
        return None, None


DEFAULT_SYSTEM_PROMPT = """你是一个助理，使用ReACT模式来回答问题。首先在“思考:”部分描述你的思考过程，然后在“动作:”部分描述你要采取的行动。每个部分都以相应的标签开头。
根据前一次的动作结果调整你的思考和行动，直到得到最终答案。确保每个动作都以“动作: 动作名称(参数)”的格式描述。
可用的动作包括: calculate(expression)。
"""


class ReACTAgent:
    def __init__(self, model_name="gpt-4o-mini", system_prompt=DEFAULT_SYSTEM_PROMPT):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.action_registry = action_registry
        client = OpenAI()
        self.client = client
        self.logger = logger

    def get_model_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content

    def run(self, user_query, max_iterations=5):
        context = []
        for _ in range(max_iterations):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query},
                *context
            ]
            self.logger.info(f"Messages: {messages}")
            response = self.get_model_response(messages)
            self.logger.info(f"Model response: {response}")
            thoughts, actions = parse_response(response)
            if not actions:
                break
            action_str = actions[0]
            self.logger.info(f"Executing action: {action_str}")
            action_result = self.execute_action(action_str)
            if action_result.startswith("动作"):
                action_result = action_result.split("结果: ")[-1]
            self.logger.info(f"Action result: {action_result}")
            context.append({
                "role": "assistant",
                "content": f"思考: {thoughts}\n动作: {action_str}\n结果: {action_result}"
            })
        return response

    def execute_action(self, action_str):
        action_name, params = parse_action(action_str)
        if action_name in self.action_registry:
            try:
                result = self.action_registry[action_name](params)
                return f"动作 '{action_name}' 执行成功，结果: {result}"
            except Exception as e:
                return f"动作 '{action_name}' 执行失败，错误: {str(e)}"
        else:
            return f"未知动作 '{action_name}'"


if __name__ == '__main__':
    @register_action("calculate")
    def calculate(expression):
        """Calculate the given mathematical expression."""
        try:
            result = eval(expression)
            return result
        except Exception as e:
            return f"Error in calculation: {str(e)}"


    agent = ReACTAgent()

    # 运行agent
    response = agent.run("What is the result of 33332.22 * 212222.323 - 1222.1 =?")
    print(response)
