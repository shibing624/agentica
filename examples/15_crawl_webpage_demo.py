# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from agentica import AzureOpenAIChat, Agent
from agentica.tools.jina_tool import JinaTool
from agentica.tools.file_tool import FileTool

m = Agent(
    tools=[JinaTool(), FileTool()],
    debug_mode=True,
)
query = """
从36kr创投平台https://pitchhub.36kr.com/financing-flash 抓取所有初创企业融资的信息;
下面是一个大致流程, 你根据每一步的运行结果对当前计划中的任务做出适当调整:
- 爬取并保存url内容，保存为36kr_finance_date.md文件, date是当前日期;
- 理解md文件，筛选最近3天的初创企业融资快讯, 打印前5个，需要提取快讯的核心信息，包括标题、链接、时间、融资规模、融资阶段等;
- 将全部结果存在本地md中
"""
r = m.run(query)
print(r)
