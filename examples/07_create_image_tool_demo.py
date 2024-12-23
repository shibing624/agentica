import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat
from agentica.tools.create_image_tool import CreateImageTool

m = Agent(tools=[CreateImageTool()], show_tool_calls=True)
r = m.run("画一匹斑马在太空行走")
print(r)
