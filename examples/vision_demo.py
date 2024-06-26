import sys

sys.path.append('..')
from actionflow import Assistant, AzureOpenAILLM

assistant = Assistant(
    llm=AzureOpenAILLM(model="gpt-4-turbo", max_tokens=4096),
)

# Single Image
assistant.print_response(
    [
        {"type": "text", "text": "描述图片"},
        {
            "type": "image_url",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        },
    ]
)

# Multiple Images
assistant.print_response(
    [
        {
            "type": "text",
            "text": "一句话说明两张图片的不同",
        },
        {
            "type": "image_url",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        },
        {
            "type": "image_url",
            "image_url": "https://img-blog.csdnimg.cn/img_convert/0ab31bdb17bebbab9c8f4185f3655b6d.jpeg",
        },
    ],
)
