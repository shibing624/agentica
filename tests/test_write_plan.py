from unittest.mock import patch, MagicMock

import pytest
import shutil
import os
from actionflow.output import Output
from actionflow.tools.write_plan import WritePlan

@pytest.fixture
def write_plan_tool():
    with patch('actionflow.llm.LLM') as MockLLM:
        mock_llm = MockLLM.return_value
        output = Output('outputs')
        tool = WritePlan(output)
        yield tool


def test_get_definition(write_plan_tool):
    definition = write_plan_tool.get_definition()
    expected_definition = {
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
    assert definition == expected_definition

    if os.path.exists('outputs'):
        shutil.rmtree('outputs')


@patch('actionflow.llm.LLM.respond')
def test_execute(mock_respond, write_plan_tool):
    mock_response = MagicMock()
    mock_response.content = """
        [
            {
                "task_id": "1",
                "dependent_task_ids": [],
                "instruction": "Collect data"
            },
            {
                "task_id": "2",
                "dependent_task_ids": ["1"],
                "instruction": "Clean data"
            },
            {
                "task_id": "3",
                "dependent_task_ids": ["2"],
                "instruction": "Train model"
            }
        ]
    """
    mock_respond.return_value = mock_response

    task_description = "Develop a machine learning model to predict house prices."
    max_subtasks = 5

    generated_plan = write_plan_tool.execute(task_description, max_subtasks)
    print(generated_plan.strip())
    assert write_plan_tool is not None
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')


