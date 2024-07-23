# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 模仿两个人辩论的例子，拜登和特朗普
"""
import sys

sys.path.append('..')
from agentica import Assistant, MoonshotLLM

llm = MoonshotLLM(model='moonshot-v1-32k')
Biden = Assistant(
    llm=llm,
    name="Biden",
    description="Suppose you are Biden, you are in a debate with Trump.",
    show_tool_calls=True,
    add_chat_history_to_messages=True,
    debug_mode=True,
)

Trump = Assistant(
    llm=llm,
    name="Trump",
    description="Suppose you are Trump, you are in a debate with Biden.",
    show_tool_calls=True,
    add_chat_history_to_messages=True,
    debug_mode=True,
)

debate = Assistant(
    llm=llm,
    name="Debate",
    team=[Biden, Trump],
    instructions=[
        "you should closely respond to your opponent's latest argument, state your position, defend your arguments, "
        "and attack your opponent's arguments, craft a strong and emotional response in 20 words， 每次发言简短有力",
        "biden 说话时你调用delegate_task_to_biden工具，trump 说话时你调用delegate_task_to_trump工具，轮流发言，请发言多次，"
        "都是中文发言, biden 再次发言时，你调用delegate_task_to_biden工具，trump 再次发言时，你调用delegate_task_to_trump工具。",
    ],
    add_chat_history_to_messages=True,
    show_tool_calls=True,
    output_dir="outputs",
    output_file_name="debate_kimi.md",
    debug_mode=True,
)
r = debate.run(
    """Trump and Biden are in a debate, Biden speak first, and then Trump speak, and then Biden speak. 2个人轮流辩论发言，各说3次，观点要求深刻，反击简短有力."""
)
print("".join(r))
