# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Database demo, demonstrates how to use SqliteDb for conversation persistence
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat
from agentica.db.sqlite import SqliteDb

if __name__ == '__main__':
    db = SqliteDb(db_file="outputs/tmp1.db")
    m = Agent(
        model=OpenAIChat(),
        add_history_to_messages=True,
        db=db,
    )
    r = m.run("How many people live in Canada?")
    print(r)
    r = m.run("What is their national anthem called?")  # 他们的国歌叫什么？
    print(r)

    sess = m.db.get_all_sessions()
    print(f"Total sessions: {len(sess)}")
    for s in sess:
        print(s)
