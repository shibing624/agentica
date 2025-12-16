# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Team debate demo, demonstrates multi-agent debate between Biden and Trump
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat

Biden = Agent(
    model=OpenAIChat(model="gpt-4o"),
    name="Biden",
    description="Suppose you are Biden, you are in a debate with Trump.中文发言",
    show_tool_calls=True,
    # debug=True,
)

Trump = Agent(
    model=OpenAIChat(model="gpt-4o"),
    name="Trump",
    description="Suppose you are Trump, you are in a debate with Biden.中文发言",
    show_tool_calls=True,
    # debug=True,
)

debate = Agent(
    model=OpenAIChat(model="gpt-4o"),
    name="Debate",
    team=[Biden, Trump],
    instructions=[
        "you should closely respond to your opponent's latest argument, state your position, defend your arguments, "
        "and attack your opponent's arguments, craft a strong and emotional response in 80 words.中文发言",
    ],
    show_tool_calls=True,
    save_response_to_file="outputs/debate.md",
    debug=True,
)

debate.print_response("特朗普和拜登正在进行一场辩论，特朗普先发言，然后拜登发言，接着特朗普再发言，如此交替，共进行三轮。现在开始。中文发言")

