from rich.pretty import pprint
import sys

sys.path.append('..')
from agentica.assistant import Assistant, AssistantMemory
from agentica import AzureOpenAILLM

llm = AzureOpenAILLM()
assistant = Assistant(llm=llm)

# -*- Print a response
assistant.run("Share a 5 word horror story.")
assistant.run("What's the weather like today?")
assistant.run("我前面问了些啥")

# -*- Get the memory
memory: AssistantMemory = assistant.memory

# -*- Print Chat History
print("============ Chat History ============")
pprint(memory.chat_history)

# -*- Print LLM Messages
print("============ LLM Messages ============")
pprint(memory.llm_messages)
