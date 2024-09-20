import sys

sys.path.append('..')

from agentica import Assistant, AzureOpenAILLM


def multiply(first_int: int, second_int: int) -> str:
    """Multiply two integers together."""
    return str(first_int * second_int)


def add(first_int: int, second_int: int) -> str:
    """Add two integers."""
    return str(first_int + second_int)


def exponentiate(base: int, exponent: int) -> str:
    """Exponentiate the base to the exponent power."""
    return str(base ** exponent)


def get_char_len(text: str) -> str:
    """Get the length of the text."""
    return str(len(text))


assistant = Assistant(
    llm=AzureOpenAILLM(model='gpt-4o'),
    tools=[multiply, add, exponentiate, get_char_len],
    debug_mode=True
)
r = assistant.run("3乘以10000005是啥?")
print(r, "".join(r))
r = assistant.run("将3的五次方乘以(12和3的和). step by step to show the result. 最后统计一下结果的字符长度。")
print(r, "".join(r))
