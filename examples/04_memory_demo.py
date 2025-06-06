import sys

from rich.pretty import pprint

sys.path.append('..')
from agentica import Agent, AgentMemory
from agentica import OpenAIChat
from agentica.memorydb import CsvMemoryDb

m = Agent(
    model=OpenAIChat(),
    add_history_to_messages=False,
    debug_mode=False
)

# -*- Print a response
r = m.run("李四住在北京，一家三口住大别墅")
print(r)
r = m.run("我前面问了啥")
print(r)
pprint(m.memory.messages)
pprint(m.memory.memories)
print("=====================")

# add history to messages
print("=== Add history to messages ===")
m = Agent(
    model=OpenAIChat(),
    memory=AgentMemory(db=CsvMemoryDb(csv_file_path='outputs/test.csv'), user_id="李四", create_user_memories=True),
    add_history_to_messages=True,
    debug_mode=False
)

# -*- Print a response
r = m.run("李四住在北京，一家三口住大别墅")
print(r)
r = m.run("我前面问了啥")
print(r)
r = m.run("李四住在北京，一家三口住大别墅，记住这个，你一句话介绍李四家庭情况")
print(r)
r = m.run("李四家里那边天气如何?是哪个气候带")
print(r)
r = m.run("介绍李四的情况")
print(r)
r = m.run("我前面问了啥")
print(r)
# -*- Get the memory
memory = m.memory

# -*- Print LLM Messages
print("============ LLM Messages ============")
pprint(memory.messages)

# -*- Print Agent Memory
print("============ Agent Memory ============")
pprint(memory.memories)
