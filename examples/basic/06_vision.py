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
    
    response = agent.run(
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
    
    response = agent.run(
        "一句话说明两张图片的不同",
        images=[
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            "https://img-blog.csdnimg.cn/img_convert/0ab31bdb17bebbab9c8f4185f3655b6d.jpeg",
        ]
    )
    print(response)


if __name__ == "__main__":
    main()
