# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Multi-round deep research demo, demonstrates multi-round reasoning with tools
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, PythonAgent, ShellTool, JinaTool, SearchSerperTool, OpenAIChat, ZhipuAIChat, Message, BaiduSearchTool, DeepSeekChat
from agentica.tools.url_crawler_tool import UrlCrawlerTool

agent =   Agent(
    model = DeepSeekChat(id="deepseek-reasoner"),
    tools = [ 
        UrlCrawlerTool(),
        BaiduSearchTool()
    ],
    enable_multi_round=True,
    max_rounds=40,
    max_tokens=40000,
    show_tool_calls=True,
    debug=True
)

agent.print_response("刘翔破世界记录多少岁", stream=True)

agent.print_response("20世纪二十年代中在上海成立的刊物成为了我国知名学生运动的先导，在此次运动中占据领导地位的高校在近百年后有一名在21世纪初某少儿电视剧中扮演重要角色的演员入学，那么请问在此电视剧中的男一号是什么时间结婚")
