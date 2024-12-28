import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat
from agentica.tools.dalle_tool import DalleTool

m = Agent(tools=[DalleTool()], show_tool_calls=True)
r = m.run("画一匹斑马在太空行走")
print(r)
