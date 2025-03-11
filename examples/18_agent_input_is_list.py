# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat

m = Agent()

# Single Image
m.print_response(
    [
        {"type": "text", "text": "描述图片"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
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
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
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
