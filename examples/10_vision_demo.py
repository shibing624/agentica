import sys

sys.path.append('..')
from agentica import Agent, OpenAIChat

m = Agent(model=OpenAIChat(id="gpt-4o"))

# Single Image
r = m.run(
    "描述图片",
    images=[
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    ]
)
print(r)

# Multiple Images
r = m.run(
    "一句话说明两张图片的不同",
    images=[
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        "https://img-blog.csdnimg.cn/img_convert/0ab31bdb17bebbab9c8f4185f3655b6d.jpeg",
    ]
)
print(r)
