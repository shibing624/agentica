import sys

sys.path.append('..')
from agentica import Agent, AzureOpenAIChat

m = Agent(
    llm=AzureOpenAIChat(model="gpt-4-turbo", max_tokens=4096),
    debug_mode=True,
)

# Single Image
r = m.run(
    [
        {"type": "text", "text": "描述图片"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            },
        },
    ]
)
print(r)

# Multiple Images
r = m.run(
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
)
print(r)
