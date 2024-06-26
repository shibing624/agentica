import pytest
import pytest
from actionflow.output import Output
from actionflow.tools.write_nb_code import WriteNbCode
import shutil
import os

class MockResponse:
    def __init__(self, content):
        self.content = content

class MockLLM:
    def respond(self, s, messages):
        return MockResponse("def example_function():\n    return \"Hello, World!\"")

@pytest.fixture
def write_code():
    output = Output('outputs')
    write_code_tool = WriteNbCode(output)
    write_code_tool.llm = MockLLM()  # Mock the LLM
    return write_code_tool

def test_get_definition(write_code):
    definition = write_code.get_definition()
    assert definition["type"] == "function"
    assert definition["function"]["name"] == "write_code"
    assert "description" in definition["function"]
    assert "parameters" in definition["function"]

def test_execute(write_code):
    task_description = "Write a function that returns 'Hello, World!'"
    code = write_code.execute(task_description)
    print('gen code:',code)
    expected_code = "def example_function():\n    return \"Hello, World!\""
    assert code.strip() == expected_code.strip()
    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
