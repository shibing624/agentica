# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from https://github.com/geekan/MetaGPT
"""
from __future__ import annotations

import base64
import re
import time
from typing import Literal, Tuple

try:
    from nbclient import NotebookClient
    from nbclient.exceptions import CellTimeoutError, DeadKernelError
    import nbformat
    from nbformat import NotebookNode
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_output
except ImportError:
    raise ImportError("The `nbclient` package is not installed. Please install it via `pip install nbclient nbformat`.")

from rich.box import MINIMAL
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from agentica.tools.base import Tool
from agentica.utils.log import logger


class RunNbCodeWrapper:
    """Run notebook code block, return result to llm, and display it."""

    nb: NotebookNode
    nb_client: NotebookClient
    console: Console
    interaction: str
    timeout: int = 600

    def __init__(
            self,
            nb=nbformat.v4.new_notebook(),
            timeout=600,
    ):
        self.nb = nb
        self.timeout = timeout
        self.console = Console()
        self.interaction = "ipython" if self.is_ipython() else "terminal"
        self.nb_client = NotebookClient(nb, timeout=timeout)

    def build(self):
        if self.nb_client.kc is None or not self.nb_client.kc.is_alive():
            self.nb_client.create_kernel_manager()
            self.nb_client.start_new_kernel()
            self.nb_client.start_new_kernel_client()

    def terminate(self):
        """Kill NotebookClient"""
        if self.nb_client.km is not None and self.nb_client.km.is_alive():
            self.nb_client.km.shutdown_kernel(now=True)
            self.nb_client.km.cleanup_resources()

            channels = [
                self.nb_client.kc.stdin_channel,
                self.nb_client.kc.hb_channel,
                self.nb_client.kc.control_channel,
            ]

            for channel in channels:
                if channel.is_alive():
                    channel.stop()

            self.nb_client.kc = None
            self.nb_client.km = None

    def reset(self):
        """Reset NotebookClient"""
        self.terminate()
        time.sleep(1)
        self.build()
        self.nb_client = NotebookClient(self.nb, timeout=self.timeout)

    def add_code_cell(self, code: str):
        self.nb.cells.append(new_code_cell(source=code))

    def add_markdown_cell(self, markdown: str):
        self.nb.cells.append(new_markdown_cell(source=markdown))

    def _display(self, code: str, language: Literal["python", "markdown"] = "python"):
        if language == "python":
            code = Syntax(code, "python", theme="paraiso-dark", line_numbers=True)
            self.console.print(code)
        elif language == "markdown":
            display_markdown(code)
        else:
            raise ValueError(f"Only support for python, markdown, but got {language}")

    def add_output_to_cell(self, cell: NotebookNode, output: str):
        """Add outputs of code execution to notebook cell."""
        if "outputs" not in cell:
            cell["outputs"] = []
        cell["outputs"].append(new_output(output_type="stream", name="stdout", text=str(output)))

    def parse_outputs(self, outputs: list, keep_len: int = 2000) -> Tuple[bool, str]:
        """Parses the outputs received from notebook execution."""
        assert isinstance(outputs, list)
        parsed_output, is_success = [], True
        for i, output in enumerate(outputs):
            output_text = ""
            if output["output_type"] == "stream" and not any(
                    tag in output["text"]
                    for tag in
                    ["| INFO     | agentica", "| ERROR    | agentica", "| WARNING  | agentica", "DEBUG"]
            ):
                output_text = output["text"]
            elif output["output_type"] == "display_data":
                if "image/png" in output["data"]:
                    self.show_bytes_figure(output["data"]["image/png"], self.interaction)
                else:
                    logger.info(
                        f"{i}th output['data'] from nbclient outputs dont have image/png, continue next output ..."
                    )
            elif output["output_type"] == "execute_result":
                output_text = output["data"]["text/plain"]
            elif output["output_type"] == "error":
                output_text, is_success = "\n".join(output["traceback"]), False

            # handle coroutines that are not executed asynchronously
            if output_text.strip().startswith("<coroutine object"):
                output_text = "Executed code failed, you need use key word 'await' to run a async code."
                is_success = False

            output_text = remove_escape_and_color_codes(output_text)
            # The useful information of the exception is at the end,
            # the useful information of normal output is at the begining.
            output_text = output_text[:keep_len] if is_success else output_text[-keep_len:]

            parsed_output.append(output_text)
        return is_success, ",".join(parsed_output)

    def show_bytes_figure(self, image_base64: str, interaction_type: str):
        image_bytes = base64.b64decode(image_base64)
        if interaction_type == "ipython":
            from IPython.display import Image, display

            display(Image(data=image_bytes))
        else:
            import io

            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            image.show()

    def is_ipython(self) -> bool:
        try:
            # 如果在Jupyter Notebook中运行，__file__ 变量不存在
            from IPython import get_ipython

            if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
                return True
            else:
                return False
        except NameError:
            return False

    def run_cell(self, cell: NotebookNode, cell_index: int) -> Tuple[bool, str]:
        """Set timeout for run code."""
        try:
            self.nb_client.execute_cell(cell, cell_index)
            return self.parse_outputs(cell.outputs)
        except CellTimeoutError:
            assert self.nb_client.km is not None
            self.nb_client.km.interrupt_kernel()
            time.sleep(1)
            error_msg = "Cell execution timed out: Execution exceeded the time limit and was stopped; consider optimizing your code for better performance."
            return False, error_msg
        except DeadKernelError:
            self.reset()
            return False, "DeadKernelError"
        except Exception as e:
            return False, str(e)

    def run(self, code: str, language: Literal["python", "markdown"] = "python") -> Tuple[str, bool]:
        """
        Return the output of code execution, and a success indicator (bool) of code execution.
        """
        self._display(code, language)

        if language == "python":
            self.add_code_cell(code=code)
            self.build()
            cell_index = len(self.nb.cells) - 1
            success, outputs = self.run_cell(self.nb.cells[-1], cell_index)
            return outputs, success

        elif language == "markdown":
            # add markdown content to markdown cell in a notebook.
            self.add_markdown_cell(code)
            # return True, there is no execution failure for markdown cell.
            return code, True
        else:
            raise ValueError(f"Only support for language: python, markdown, but got {language}, ")


def remove_escape_and_color_codes(input_str: str):
    # 使用正则表达式去除jupyter notebook输出结果中的转义字符和颜色代码
    # Use regular expressions to get rid of escape characters and color codes in jupyter notebook output.
    pattern = re.compile(r"\x1b\[[0-9;]*[mK]")
    result = pattern.sub("", input_str)
    return result


def display_markdown(content: str):
    # Use regular expressions to match blocks of code one by one.
    matches = re.finditer(r"```(.+?)```", content, re.DOTALL)
    start_index = 0
    content_panels = []
    # Set the text background color and text color.
    style = "black on white"
    # Print the matching text and code one by one.
    for match in matches:
        text_content = content[start_index: match.start()].strip()
        code_content = match.group(0).strip()[3:-3]  # Remove triple backticks

        if text_content:
            content_panels.append(Panel(Markdown(text_content), style=style, box=MINIMAL))

        if code_content:
            content_panels.append(Panel(Markdown(f"```{code_content}"), style=style, box=MINIMAL))
        start_index = match.end()

    # Print remaining text (if any).
    remaining_text = content[start_index:].strip()
    if remaining_text:
        content_panels.append(Panel(Markdown(remaining_text), style=style, box=MINIMAL))

    # Display all panels in Live mode.
    with Live(auto_refresh=False, console=Console(), vertical_overflow="visible") as live:
        live.update(Group(*content_panels))
        live.refresh()


class RunNbCodeTool(Tool):
    """Run notebook code block, return result to llm, and display it."""

    def __init__(self, timeout: int = 600):
        super().__init__(name="run_nb_code_tool")
        self.executor = RunNbCodeWrapper(timeout=timeout)
        self.register(self.execute_nb_code)

    def execute_nb_code(self, code: str) -> str:
        """Execute a code block in a Jupyter Notebook and return the result.

        Args:
            code: str, The code to execute.

        Example:
            from agentica.tools.run_nb_code_tool import RunNbCodeTool
            m = RunNbCodeTool()
            r = m.execute_nb_code("import math\nprint(math.sqrt(79192201))")
            print(r)

        Returns:
            str, The result of the code execution.
        """
        try:
            outputs, success = self.executor.run(code)
            return outputs if success else "Execution failed"
        except Exception as e:
            return f"Execution failed with error: {str(e)}"


if __name__ == '__main__':
    executor = RunNbCodeWrapper()
    output, success = executor.run("print('Hello, world!')")
    print("Output:", output)
    print("Success:", success)

    # 添加并显示 Python 代码单元
    executor.add_code_cell("a = 10\nb = 20\nc = a + b\nprint(c)")
    executor._display("a = 10\nb = 20\nc = a + b\nprint(c)", "python")

    # 添加并显示 Markdown 单元
    executor.add_markdown_cell("# This is a markdown cell\n\nThis cell contains **bold** text and `inline code`.")
    executor._display("# This is a markdown cell\n\nThis cell contains **bold** text and `inline code`.", "markdown")

    # 构建 NotebookClient
    executor.build()

    # 运行第一个代码单元
    output, success = executor.run("a = 10\nb = 20\nc = a + b\nprint(c)", "python")
    print("Output:", output)
    print("Success:", success)

    # 运行第二个代码单元
    # 8899*8899=79192201
    output, success = executor.run("import math\nprint(math.sqrt(79192201))", "python")
    print("Output:", output)
    print("Success:", success)

    # 运行 Markdown 单元
    output, success = executor.run(
        "# This is a markdown cell\n\nThis cell contains **bold** text and `inline code`.",
        "markdown")
    print("Output:", output)
    print("Success:", success)

    # 终止 NotebookClient
    executor.terminate()

    # 重置 NotebookClient
    executor.reset()

    # 测试解析输出
    outputs = [
        {"output_type": "stream", "text": "This is a stream output\n"},
        {"output_type": "execute_result", "data": {"text/plain": "This is an execute result"}},
        {"output_type": "error", "traceback": ["Traceback (most recent call last):", "Error: Something went wrong"]}
    ]
    is_success, parsed_output = executor.parse_outputs(outputs)
    print("Parsed Output:", parsed_output)
    print("Is Success:", is_success)

    # 测试移除转义和颜色代码
    cleaned_output = remove_escape_and_color_codes("\x1b[31mThis is red text\x1b[0m")
    print("Cleaned Output:", cleaned_output)

    m = RunNbCodeTool()
    r = m.execute_nb_code("import math\nprint(math.sqrt(79192201))")
    print(type(r), '\n\n', r)
