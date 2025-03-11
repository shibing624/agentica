import sys

sys.path.append('..')
from agentica import Agent
from agentica import Moonshot, Qwen, OpenAIChat,Doubao

m = Agent(
    # model=Qwen(api_key='sk-8713fa90974545f5abf7464714d3b011', id='qwq-plus'),
    # model=Doubao(api_key='e249abdf-51d4-482a-b5c8-8777a73990c2', id='ep-20250227113433-vv7hr'),
    model=OpenAIChat(id='o3-mini'),
    system_prompt="分享一个两句话的故事。",
    user_prompt="12000年的爱。",
    debug_mode=True,
)
# r = m.run(stream=True)
# for i in r:
#     print(i)
m.print_response(stream=True)