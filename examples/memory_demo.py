import sys

from rich.pretty import pprint

sys.path.append('..')
from agentica.assistant import Assistant, AssistantMemory
from agentica import AzureOpenAILLM
from agentica.memory import CsvMemoryDb

llm = AzureOpenAILLM()
assistant = Assistant(
    llm=llm,
    memory=AssistantMemory(db=CsvMemoryDb(), user_id="test"),
    add_chat_history_to_messages=True,
    debug_mode=True
)

# -*- Print a response
r = assistant.run("一句话介绍北京")
print(r, "".join(r))
r = assistant.run("大多数时候天气如何?")
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
