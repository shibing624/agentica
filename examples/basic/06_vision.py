# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Vision demo - Demonstrates how to use Agent with image inputs

This example shows how to:
1. Analyze a single image
2. Compare multiple images
3. Use vision with custom prompts
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat


def main():
    # Create agent with vision-capable model
    agent = Agent(model=OpenAIChat(id="gpt-4o"))

    # Example 1: Single image analysis
    print("=" * 60)
    print("Example 1: Single Image Analysis")
    print("=" * 60)
    
    response = agent.run_sync(
        "描述这张图片的内容",
        images=[
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
        ]
    )
    print(response)

    # Example 2: Multiple images comparison
    print("\n" + "=" * 60)
    print("Example 2: Multiple Images Comparison")
    print("=" * 60)
    
    response = agent.run_sync(
        "一句话说明两张图片的不同",
        images=[
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            "https://img-blog.csdnimg.cn/img_convert/0ab31bdb17bebbab9c8f4185f3655b6d.jpeg",
        ]
    )
    print(response)

    # Example 3: Base64 encoded image
    print("\n" + "=" * 60)
    print("Example 3: Base64 Encoded Image")
    print("=" * 60)
    
    import base64
    
    # Read local image and convert to base64
    image_path = os.path.join(os.path.dirname(__file__), "..", "data", "chinese.jpg")
    with open(image_path, "rb") as f:
        image_data = f.read()
    base64_image = base64.b64encode(image_data).decode("utf-8")
    base64_image_with_prefix = f"data:image/jpeg;base64,{base64_image}"
    
    response = agent.run_sync(
        "这张图片里有什么？",
        images=[base64_image_with_prefix]
    )
    print(response)


if __name__ == "__main__":
    main()
