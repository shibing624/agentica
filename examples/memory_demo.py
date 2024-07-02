import sys

from rich.pretty import pprint

sys.path.append('..')
from agentica.assistant import Assistant, AssistantMemory
from agentica import AzureOpenAILLM

llm = AzureOpenAILLM()
assistant = Assistant(llm=llm)

# -*- Print a response
r = assistant.run("Share a 5 word horror story.")
print(r, "".join(r))
r = assistant.run("What's the weather like today?")
print("".join(r))
r = assistant.run("我前面问了些啥")
print("".join(r))

# -*- Get the memory
memory: AssistantMemory = assistant.memory

# -*- Print Chat History
print("============ Chat History ============")
pprint(memory.chat_history)

# -*- Print LLM Messages
print("============ LLM Messages ============")
pprint(memory.llm_messages)
