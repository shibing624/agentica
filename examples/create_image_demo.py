import sys

sys.path.append('..')
from agentica import Assistant, OpenAILLM
from agentica.tools.create_image_tool import CreateImageTool


assistant = Assistant(tools=[CreateImageTool()])
r = assistant.run("画一匹斑马在太空行走")
print(r, "".join(r))
