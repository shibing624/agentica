import sys

sys.path.append('..')
from agentica import Assistant

assistant = Assistant(
    system_prompt="Share a 2 sentence story about",
    user_prompt="Love in the year 12000.",
    debug_mode=True,
)
assistant.run()
