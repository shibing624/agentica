import sys

from rich.pretty import pprint

sys.path.append('..')
from agentica.assistant import Assistant, AssistantMemory
from agentica import AzureOpenAIChat
from agentica.memory import CsvMemoryDb

llm = AzureOpenAIChat()
assistant = Assistant(
    llm=llm,
    memory=AssistantMemory(db=CsvMemoryDb(), user_id="李四"),
    add_chat_history_to_messages=True,
    create_memories=True,
    debug_mode=True
)

# -*- Print a response
r = assistant.run("李四住在北京，一家三口住大别墅，记住这个，你一句话介绍李四家庭情况")
print(r, "".join(r))
r = assistant.run("李四家里那边天气如何?是哪个气候带")
print("".join(r))
r = assistant.run("介绍李四的情况")
print("".join(r))

# -*- Get the memory
memory: AssistantMemory = assistant.memory

# -*- Print Chat History
print("============ Chat History ============")
pprint(memory.chat_history)

# -*- Print LLM Messages
print("============ LLM Messages ============")
pprint(memory.llm_messages)

# -*- Print Assistant Memory
print("============ Assistant Memory ============")
pprint(memory.memories)
