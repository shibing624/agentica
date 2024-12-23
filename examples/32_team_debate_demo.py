# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 模仿两个人辩论的例子，拜登和特朗普
"""
import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat

Biden = Agent(
    model=OpenAIChat(model="gpt-3.5-turbo-1106"),
    name="Biden",
    description="Suppose you are Biden, you are in a debate with Trump.",
    show_tool_calls=True,
    debug_mode=True,
)

Trump = Agent(
    model=OpenAIChat(model="gpt-3.5-turbo-1106"),
    name="Trump",
    description="Suppose you are Trump, you are in a debate with Biden.",
    show_tool_calls=True,
    debug_mode=True,
)

debate = Agent(
    model=OpenAIChat(model="gpt-4o-mini"),
    name="Debate",
    team=[Biden, Trump],
    instructions=[
        "you should closely respond to your opponent's latest argument, state your position, defend your arguments, "
        "and attack your opponent's arguments, craft a strong and emotional response in 80 words",
    ],
    show_tool_calls=True,
    save_response_to_file="outputs/debate.md",
)
debate.print_response(
    """Trump and Biden are in a debate, Biden speak first, and then Trump speak, and then Biden speak, and so on, in 3 turns.
    Now begin.""",
    stream=True,
    markdown=True,
)