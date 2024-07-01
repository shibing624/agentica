import sys

sys.path.append('..')
from agentica import Assistant, OpenAILLM
from agentica.tools.create_image import CreateImageTool


assistant = Assistant(tools=[CreateImageTool()])
assistant.run("画一匹斑马在太空行走")
