# -*- coding: utf-8 -*-
"""
A minimal implementation of a ReACT agent with support for multiple tools.
"""
import re
from loguru import logger
from openai import OpenAI

# Registry to hold available actions
action_registry = {}


def register_action(action_name):
    """Decorator to register an action in the action registry."""

    def decorator(func):
        action_registry[action_name] = func
        return func

    return decorator


def parse_response(response):
    """Parse the model's response into thoughts and actions."""
    thoughts = []
    actions = []
    lines = response.split('\n')
    for line in lines:
        if line.startswith("Thought:"):
            thoughts.append(line[len("Thought:"):].strip())
        elif line.startswith("Action:"):
            actions.append(line[len("Action:"):].strip())
    return thoughts, actions


def parse_action(action_str):
    """Parse an action string into action name and parameters."""
    match = re.match(r'^(\w+)\((.*)\)$', action_str)
    if match:
        action_name = match.group(1)
        params = match.group(2)
        # Remove parameter names and keep only values
        params = re.sub(r'\w+=', '', params)
        logger.debug(f"Action name: {action_name}, Params: {params}")
        return action_name, params
    else:
        logger.warning(f"Invalid action string: {action_str}")
        return None, None


TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

PROMPT_REACT = """You are an assistant that uses the ReACT framework to answer questions. 

First, describe your thought process in the "Thought:" section, then describe the action you will take in the "Action:" section. 
Each section should start with its respective label. 
Adjust your thoughts and actions based on the results of previous actions until you arrive at the final answer. 
Ensure that each action is described in the format "Action: action_name(parameters)".
Available actions include: 

{tools_text}

Begin!
"""


class ReactAgent:
    """A ReACT agent that interacts with the OpenAI API to perform actions based on user queries."""

    def __init__(self, model_name="gpt-4o", tools=None):
        self.model_name = model_name
        self.tools = tools or []
        self.action_registry = action_registry
        self.client = OpenAI()
        self.logger = logger
        self.system_prompt = self._generate_system_prompt()

    def _generate_system_prompt(self):
        tools_text = "\n".join([TOOL_DESC.format(**tool) for tool in self.tools])
        return PROMPT_REACT.format(tools_text=tools_text)

    def get_model_response(self, messages):
        """Get a response from the OpenAI model."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content

    def run(self, user_query, max_iterations=5):
        """Run the agent with the given user query."""
        context = []
        for _ in range(max_iterations):
            messages = [
                {"role": "system", "content": self.system_prompt.format(query=user_query)},
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
            if action_result.startswith("Action"):
                action_result = action_result.split("Result: ")[-1]
            self.logger.info(f"Action result: {action_result}")
            context.append({
                "role": "assistant",
                "content": f"Thought: {thoughts}\nAction: {action_str}\nResult: {action_result}"
            })
        return response

    def execute_action(self, action_str):
        """Execute an action based on the action string."""
        action_name, params = parse_action(action_str)
        if action_name in self.action_registry:
            try:
                self.logger.debug(f"Executing action: {action_name} with params: {params}")
                result = self.action_registry[action_name](params)
                self.logger.debug(f"Action '{action_name}' executed successfully, Result: {result}")
                return f"Action '{action_name}' executed successfully, Result: {result}"
            except Exception as e:
                self.logger.warning(f"Action '{action_name}' failed, Error: {str(e)}")
                return f"Action '{action_name}' failed, Error: {str(e)}"
        else:
            return f"Unknown action '{action_name}'"


if __name__ == '__main__':
    @register_action("calculate")
    def calculate(expression=""):
        """Calculate the result of a mathematical expression."""
        try:
            result = eval(expression)
            return result
        except Exception as e:
            return f"Error in calculation: {str(e)}"


    @register_action("reverse_string")
    def reverse_string(s=""):
        """Reverse the input string."""
        try:
            logger.debug(f"Reversing string: {s}")
            r = s[::-1]
            logger.debug(f"Reversed string: {r}")
            return r
        except Exception as e:
            return f"Error in reversing string: {str(e)}"


    @register_action("text_length")
    def text_length(s):
        """Calculate the length of the input text string."""
        try:
            logger.debug(f"Calculating text length: {s}")
            r = len(s)
            logger.debug(f"Text length: {r}")
            return r
        except Exception as e:
            return f"Error in calculating text length: {str(e)}"


    tools = [
        {
            "name_for_model": "calculate",
            "name_for_human": "Calculator",
            "description_for_model": "useful for performing mathematical calculations.",
            "parameters": "expression: a mathematical expression to evaluate."
        },
        {
            "name_for_model": "reverse_string",
            "name_for_human": "String Reverser",
            "description_for_model": "useful for reversing strings.",
            "parameters": "the string to reverse."
        },
        # get text string length
        {
            "name_for_model": "text_length",
            "name_for_human": "Text Length Calculator",
            "description_for_model": "useful for calculating the length of a text string.",
            "parameters": "the text string to calculate the length."
        }
    ]

    agent = ReactAgent(tools=tools)

    # Run the agent with a query that requires both tools
    response = agent.run(
        "What is the result of reversing the string '123hello111---232323k23你好，水电费不着调。。。。--23' and then calculating the length of the reversed string, 不要包括引号")
    print(response)
