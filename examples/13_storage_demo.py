import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat
from agentica import SqlAgentStorage

if __name__ == '__main__':
    m = Agent(
        model=OpenAIChat(),
        add_history_to_messages=True,
        storage=SqlAgentStorage(table_name="ai", db_file="outputs/tmp.db"),
    )
    r = m.run("How many people live in Canada?")
    print(r)
    r = m.run("What is their national anthem called?")  # 他们的国歌叫什么？
    print(r)

    sess = m.storage.get_all_sessions()
    print(f"Total sessions: {len(sess)}")
    for s in sess:
        print(s)
