import sys

sys.path.append('..')
from actionflow import Assistant, OpenAILLM
from actionflow.tools.create_image import CreateImageTool


assistant = Assistant(tools=[CreateImageTool()])
assistant.print_response("画一匹斑马在太空行走")
