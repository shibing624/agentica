import sys

sys.path.append('..')

from agentica import Agent, OpenAIChat, WeatherTool, ShellTool, Tool, FileTool


def multiply(first_int: int, second_int: int) -> str:
    """Multiply two integers together."""
    return str(first_int * second_int)


class AddTool(Tool):
    def __init__(self):
        super().__init__()
        self.register(self.run_add)

    def run_add(self, first_int: float, second_int: float) -> str:
        """Add two number together."""
        return str(first_int + second_int)


def exponentiate(base: int, exponent: int) -> str:
    """Exponentiate the base to the exponent power."""
    return str(base ** exponent)


def get_char_len(text: str) -> str:
    """Get the length of the text."""
    return str(len(text))


m = Agent(
    model=OpenAIChat(id='gpt-4o'),
    tools=[multiply, AddTool(), exponentiate, get_char_len, WeatherTool(), ShellTool(), FileTool()],
    debug_mode=True
)
m.print_response("3乘以10000005是啥?")
m.print_response("将3的五次方乘以(12和3的和). step by step to show the result. 最后统计一下结果的字符长度。")
m.print_response("明天北京天气多少度？温度 乘以 2333 = ？")
