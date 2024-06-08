# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from https://github.com/geekan/MetaGPT
"""

from loguru import logger

from actionflow.llm import LLM, Settings
from actionflow.output import Output
from actionflow.tool import BaseTool
from actionflow.utils import CodeParser

SYSTEM_PROMPT = (
    "You are an AI assistant that helps users create detailed plans for their tasks. "
    "Your goal is to generate a clear and executable plan based on the provided task description. "
    "Ensure that the plan is broken down into manageable subtasks and follows the given constraints."
)

PROMPT_TEMPLATE: str = """
# Task Description:
{task_description}

# Task:
Based on the task description, write a plan or modify an existing plan to achieve the goal. 
A plan consists of one to {max_subtasks} tasks.
If you are modifying an existing plan, carefully follow the instructions and avoid unnecessary changes. 
Provide the entire plan unless instructed to modify only one task.
Use the same language as the user input with [Task Description] to answer the question.
If you encounter errors in the current task, revise and output the current single task only.
Output a list of JSON objects in the following format:
```json
[
    {{
        "task_id": str = "unique identifier for a task in plan, can be an ordinal",
        "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
        "instruction": "what you should do in this task, one short phrase or sentence",
    }},
    ...
]
```
"""


class WritePlan(BaseTool):

    def __init__(self, output: Output):
        """
        Initializes the WritePlan object.
        """
        super().__init__(output)
        self.llm = LLM()

    def get_definition(self) -> dict:
        """
        Returns a dictionary that defines the function. It includes the function's name, description, and parameters.

        :return: A dictionary that defines the function.
        :rtype: dict
        """
        return {
            "type": "function",
            "function": {
                "name": "write_plan",
                "description": "Generate a plan based on the provided task description and constraints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "The description of the task to generate a plan for."
                        },
                        "max_subtasks": {
                            "type": "integer",
                            "default": 5,
                            "description": "The maximum number of subtasks in the plan."
                        }
                    },
                    "required": ["task_description"],
                },
            }
        }

    def execute(
            self,
            task_description: str,
            max_subtasks: int = 5,
    ) -> str:
        user_content = PROMPT_TEMPLATE.format(
            task_description=task_description,
            max_subtasks=max_subtasks
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        # LLM call
        try:
            settings = Settings()
            rsp_message = self.llm.respond(settings, messages)
            plan = CodeParser.parse_code(block="", text=rsp_message.content)
            return plan
        except Exception as e:
            logger.error(f"Error during LLM response: {e}")
            return ""


if __name__ == '__main__':
    output = Output('o')
    write_plan_tool = WritePlan(output)
    # Example 1
    task_description = """
    Develop a web application that allows users to upload images,
    apply filters, and download the edited images. The application should have user authentication,
    image processing capabilities.
    """
    generated_plan = write_plan_tool.execute(task_description, max_subtasks=5)
    print(generated_plan)

    # Example 2
    task_description = "坐高铁从北京到武汉"
    generated_plan = write_plan_tool.execute(task_description, max_subtasks=3)
    print(generated_plan)

    import os

    os.removedirs(output.output_dir)
