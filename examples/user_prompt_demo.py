import sys

sys.path.append('..')
from agentica import Assistant
from agentica.llm.moonshot import Moonshot

assistant = Assistant(
    llm=Moonshot(),
    system_prompt="分享一个两句话的故事。",
    user_prompt="12000年的爱。",
    debug_mode=True,
)
r = assistant.run()
print("".join(r))
