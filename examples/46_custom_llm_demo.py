import sys
from pathlib import Path

sys.path.append('..')

from agentica import Agent, OpenAIChat, OpenAILike, Message


def main():
    model = OpenAILike(id='gemini-2.5-pro', api_key='your_api_key',
                       base_url='your_base_url')
    messages = [Message(role="user", content="一句话介绍林黛玉")]
    r = model.response(messages)
    print('model:', model)
    print(r)

    m = Agent(
        model=model,
        debug_mode=True,
    )
    m.print_response(message="你是谁？详细介绍自己，你的知识库到哪天", stream=True)


if __name__ == "__main__":
    main()
