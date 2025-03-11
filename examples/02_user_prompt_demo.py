import sys

sys.path.append('..')
from agentica import Agent
from agentica import OpenAIChat

m = Agent(
    model=OpenAIChat(),
    debug_mode=True,
)
m.print_response(message="分享一个两句话的故事,12000年的爱。", stream=True)
