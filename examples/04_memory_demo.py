import sys
import os
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
memory_file = "outputs/memory.csv"
if os.path.exists(memory_file):
    os.remove(memory_file)
m = Agent(
    model=OpenAIChat(),
    memory=AgentMemory(db=CsvMemoryDb(csv_file_path='outputs/memory.csv'), create_user_memories=True),
    add_history_to_messages=True,
    debug_mode=False
)

# -*- Print a response
r = m.run("李四住在北京，一家三口住大别墅")
print(r)
r = m.run("李四住在北京，一家三口住大别墅，记住这个，你一句话介绍李四家庭情况")
print(r)
r = m.run("李四家里那边天气如何?是哪个气候带")
print(r)
r = m.run("我前面问了啥")
print(r)
# # 示例
# ## 示例一
# 输入：你好。 输出：{{"facts" : []}}
# ## 示例二
# 输入：今天气温是18摄氏度。 输出：{{"facts" : []}}
# ## 示例三
# 输入：你好，我在寻找一家位于什刹海的烤鸭店。 输出：{{"facts" : ["在寻找一家位于什刹海的烤鸭店"]}}
# ## 示例四
# 输入：昨天，我和李明在下午三点见面，一起讨论了新项目。 输出：{{"facts" : ["昨天和李明在三点见面"]}}
# ## 示例五
# 输入：我的名字是林瀚，我是一名软件工程师 输出：{{"facts" : ["姓名是是林瀚", "职业是软件工程师"]}}
# ## 示例六
# 输入：我最喜欢的电影是《花样年华》 输出：{{"facts" : ["最喜欢的电影是《花样年华》"]}}
print(m.run("你好，我在寻找一家位于什刹海的烤鸭店。"))
print(m.run("昨天，我和李明在下午三点见面，一起讨论了新项目。"))
print(m.run("我的名字是林瀚，我是一名软件工程师"))
print(m.run("我最喜欢的电影是《花样年华》"))
print(m.run("今天气温是18摄氏度。"))
print(m.run("hi"))
print(m.run("我前面问了啥"))
# -*- Get the memory
memory = m.memory

# -*- Print LLM Messages
print("============ LLM Messages ============")
pprint(memory.messages)

# -*- Print Agent Memory
print("============ Agent Memory ============")
pprint(memory.memories)
