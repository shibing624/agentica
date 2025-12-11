# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Agent input is list demo, demonstrates how to use Agent with list-type image inputs
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent, OpenAIChat

m = Agent(model=OpenAIChat(id='gpt-4o-mini'))

# Single Image
m.print_response(
    [
        {"type": "text", "text": "描述图片"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://img-blog.csdnimg.cn/img_convert/0ab31bdb17bebbab9c8f4185f3655b6d.jpeg"
            },
        },
    ],
    stream=True
)

# Multiple Images
m.print_response(
    [
        {
            "type": "text",
            "text": "一句话说明两张图片的不同",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://i-blog.csdnimg.cn/blog_migrate/38e2071af527a1e864ee31bb2a5c2025.png"
            },
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://img-blog.csdnimg.cn/img_convert/0ab31bdb17bebbab9c8f4185f3655b6d.jpeg"
            },
        },
    ],
    stream=True
)
