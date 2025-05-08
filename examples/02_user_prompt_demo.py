import sys

sys.path.append('..')
from agentica import Agent
from agentica import OpenAIChat, AzureOpenAIChat

m = Agent(
    model=AzureOpenAIChat(id="gpt-4.1"),
    debug_mode=True,
)
m.print_response(message="你是谁？详细介绍自己，你的知识库到哪天", stream=True)
