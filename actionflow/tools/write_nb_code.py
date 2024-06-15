# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from https://github.com/geekan/MetaGPT
"""
from __future__ import annotations

from loguru import logger

from actionflow.llm import LLM, Settings
from actionflow.output import Output
from actionflow.tool import BaseTool
from actionflow.utils import NotebookCodeParser

SYSTEM_PROMPT = (
    "As a data scientist, you need to help the user achieve their goal step by step in a "
    "continuous Jupyter notebook environment. Ensure that the code you generate is executable "
    "within the same Jupyter notebook and leverages pre-defined tools whenever possible."
)

PROMPT_TEMPLATE = """
# Task Description
{task_description}

# Plan Status
{plan_status}

# Tool Info
{tool_info}

# Constraints
- If the current task is in Plan Status, focus on it; otherwise, address the User Requirement directly.
- Ensure the new code is executable in the same Jupyter notebook as the previously executed code.
- Prioritize using pre-defined tools for the same functionality.

# Output
While some concise thoughts are helpful, code is absolutely required. Always output one and only one code block 
in your response. Output code in the following format:
```python
your code
```
"""


class WriteNbCode(BaseTool):

    def get_definition(self) -> dict:
        """
        Returns a dictionary that defines the function. It includes the function's name, description, and parameters.

        :return: A dictionary that defines the function.
        :rtype: dict
        """
        return {
            "type": "function",
            "function": {
                "name": "write_nb_code",
                "description": "Generate Python code based on the provided task description, "
                               "plan status, and tool info.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "The prompt that describes the task."
                        },
                        "plan_status": {
                            "type": "string",
                            "default": "In Progress",
                            "description": "The current status of the plan."

                        },
                        "tool_info": {
                            "type": "string",
                            "default": "No additional tools required.",
                            "description": "Information about the tools available."
                        }
                    },
                    "required": ["task_description"],
                },
            }
        }

    def execute(
            self,
            task_description: str,
            plan_status: str = "",
            tool_info: str = "",
    ) -> str:
        user_content = PROMPT_TEMPLATE.format(
            task_description=task_description,
            plan_status=plan_status,
            tool_info=tool_info,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # LLM call
        try:
            settings = Settings(temperature=0)
            rsp_message = LLM().respond(settings, messages)
            code = NotebookCodeParser.parse_code(block="", text=rsp_message.content)
            return code
        except Exception as e:
            logger.error(f"Error during LLM response: {e}")
            return ""


if __name__ == '__main__':
    output = Output('o')
    write_code_tool = WriteNbCode(output)

    # Example 1
    task_description = "Write a Python function that adds two numbers."
    plan_status = "In Progress"
    tool_info = "No additional tools required."

    generated_code = write_code_tool.execute(task_description, plan_status, tool_info)
    print(generated_code)

    # Example 2
    task_description = "Write a Python class that represents a simple calculator with " \
                       "add, subtract, multiply, and divide methods."
    generated_code = write_code_tool.execute(task_description)
    print(generated_code)

    # Example 3
    task_description = """
    Write a Python class named `DataProcessor` that includes the following methods:
    1. `__init__(self, data: list)`: Initializes the class with a list of data.
    2. `clean_data(self) -> list`: Cleans the data by removing any None values.
    3. `compute_statistics(self) -> dict`: Computes and returns basic statistics (mean, median, mode) of the data.
    4. `save_to_file(self, filename: str)`: Saves the cleaned data to a specified file.
    5. `load_from_file(self, filename: str)`: Loads data from a specified file and updates the class data.
    """

    plan_status = "In Progress"
    tool_info = "No additional tools required."

    generated_code = write_code_tool.execute(task_description, plan_status, tool_info)
    print(generated_code)
    import os

    os.removedirs(output.data_dir)
