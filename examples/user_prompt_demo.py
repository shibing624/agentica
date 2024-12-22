import sys

sys.path.append('..')
from agentica import Agent
from agentica import DeepSeekChat

m = Agent(
    llm=DeepSeekChat(),
    system_prompt="分享一个两句话的故事。",
    user_prompt="12000年的爱。",
    debug_mode=True,
)
r = m.run()
print(r)
