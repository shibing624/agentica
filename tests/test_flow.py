# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module contains tests for the Flow class.
"""

import json
import os
import shutil
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import patch

import pytest

from actionflow.assistant import Assistant
from actionflow.llm import Settings


def mock_llm_respond(
        settings: Settings,
        messages: List[Dict[str, str]],
        tool_calls: Optional[List[Dict[str, str]]] = None,
) -> SimpleNamespace:
    """
    Mock the LLM respond method.
    """
    if settings.tool_name != "none":
        return SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=SimpleNamespace(
                name=settings.tool_name,
                arguments=json.dumps({"arg1": "value1", "arg2": "value2"}),
            ),
        )
    elif messages[-1]["role"] == "function":
        return SimpleNamespace(
            role="assistant",
            content=f"Response to function call {messages[-1]['name']}.",
        )
    else:
        return SimpleNamespace(
            role="assistant",
            content=f"Response to user message {messages[-1]['content']}.",
        )


def mock_function_definition(function_name: str) -> dict:
    """
    Mock a function definition.
    """
    return {"name": f"{function_name} definition"}


def mock_function_execute(args_json: str) -> str:
    """
    Mock a function execute method.
    """
    return f"Response to function call with these arguments: {args_json}."


@pytest.fixture
def flows_path():
    """
    Get the path to the test flows directory.
    """
    return os.path.dirname(os.path.abspath(__file__))


def test_file_not_found(flows_path):
    """
    Test that a FileNotFoundError is raised if the flow file does not exist.
    """
    with pytest.raises(FileNotFoundError):
        _ = Assistant("file_not_found")


def test_flow_basic(flows_path):
    """
    Test that we can load and run a basic flow.
    """
    with patch("actionflow.flow.LLM") as MockLLM:
        mock_llm = MockLLM.return_value
        mock_llm.respond.side_effect = mock_llm_respond

        flow = Assistant("test_flow_basic.json")

        assert flow.system_message == "Test system message."
        assert len(flow.tasks) == 3

        # Test that settings get loaded correctly
        with open(os.path.join(flows_path, "test_flow_basic.json"), "r") as file:
            flow_json = json.load(file)
            flow_json_test_settings = flow_json["tasks"][0]["settings"]
            flow_object_settings = [
                s for s in flow.tasks[0].settings.__dict__ if s != "tool_name" and s != "tool_choice"
            ]
            for setting in flow_object_settings:
                print("setting:", setting, "flow_json_test_settings[setting]:", flow_json_test_settings[setting])
                assert setting is not None

        # Test that we use default settings if there are none provided
        assert flow.tasks[1].settings == Settings()

        flow.run()

        assert MockLLM.call_count == 1
        assert mock_llm.respond.call_count > 0

        last_call = mock_llm.respond.call_args
        last_messages = last_call[0][1]
        print("last_messages:", last_messages)
        if os.path.exists('outputs'):
            shutil.rmtree('outputs')


def test_flow_with_variables(flows_path):
    """
    Test that we can load and run a flow with variables.
    """
    variables = {
        "system_message_variable": "system_message_variable_value",
        "task_1_variable": "task_1_variable_value",
    }
    flow = Assistant("test_flow_with_variables.json", variables, flows_path)

    # Test that we set variables correctly
    assert flow.system_message == "System message with system_message_variable_value."
    assert flow.tasks[0].action == "Task 1 action with task_1_variable_value."
    assert (
            flow.tasks[1].action
            == "Task 2 action with {task_2_curly_bracket_non_variable}."
    )

    # Test that we raise an error if we provide extra variables
    variables["extra_variable"] = "extra_variable_value"
    with pytest.raises(
            ValueError, match="Extra variables provided: {'extra_variable'}."
    ):
        _ = Assistant("test_flow_with_variables.json", variables, flows_path)

    # Test that we raise an error if we don't provide all variables
    variables.pop("extra_variable")
    variables.pop("system_message_variable")
    with pytest.raises(
            ValueError, match="Missing variable values for: {'system_message_variable'}."
    ):
        _ = Assistant("test_flow_with_variables.json", variables, flows_path)

    if os.path.exists('outputs'):
        shutil.rmtree('outputs')


def test_flow_with_functions(flows_path):
    """
    Test that we can load and run a flow with functions.
    """
    with patch("actionflow.flow.Tool") as MockFunction:
        with patch("actionflow.flow.LLM") as MockLLM:
            mock_function = MockFunction.return_value
            mock_function.definition.side_effect = mock_function_definition
            mock_function.execute.side_effect = mock_function_execute

            mock_llm = MockLLM.return_value
            mock_llm.respond.side_effect = mock_llm_respond

            flow = Assistant("test_flow_with_functions.json")

            # Ensure that we have the correct number of functions
            assert len(flow.tools) == 2

            flow.run()

            last_call = mock_llm.respond.call_args
            last_messages = last_call[0][1]
            print("last_messages:", last_messages)

            # Ensure that we're responding to functions correctly
            assert len(last_messages[-1]["content"]) > 0
            if os.path.exists('outputs'):
                shutil.rmtree('outputs')
