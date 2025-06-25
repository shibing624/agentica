# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 测试真正的并行Memory Classification优化
"""
import os
import sys
import time
import asyncio

sys.path.append('..')

from agentica import Agent
from agentica.memory import AgentMemory, MemoryClassifier
from agentica.memorydb import CsvMemoryDb, MemoryRow
from agentica.model.openai import OpenAIChat

def create_test_agent(name: str, memory_file: str) -> Agent:
    """创建测试Agent"""
    return Agent(
        model=OpenAIChat(id='gpt-4o-mini'),
        name=name,
        user_id=name,
        description=f"你是{name}，一个AI助手",
        memory=AgentMemory(
            db=CsvMemoryDb(csv_file_path=memory_file),
            create_user_memories=True,
            update_user_memories_after_run=True,
            classifier=MemoryClassifier(model=OpenAIChat(id='gpt-4o-mini'))
        ),
        debug=True
    )

def setup_test_memories(agent: Agent, num_memories: int = 10):
    """为Agent设置测试记忆数据"""
    print(f"为{agent.name}创建{num_memories}条测试记忆...")
    
    for i in range(num_memories):
        agent.memory.db.upsert_memory(
            MemoryRow(
                user_id=agent.user_id,
                memory={
                    "memory": f"用户记忆 {i}: 用户喜欢{i % 3}类型的活动",
                    "input_text": f"我喜欢{i % 3}类型的活动"
                }
            )
        )
    print(f"已创建{num_memories}条记忆")

async def test_parallel_optimization():
    """测试并行优化效果"""
    print("=== 测试并行Memory Classification优化 ===")
    
    # 创建两个相同配置的Agent

    
    # 测试消息 - 这些消息应该会触发memory classification
    test_messages = [
        "你好，我是张三，我喜欢编程和阅读",
        "我住在北京，工作是软件工程师",
        "你还记得我之前说过什么吗？"
    ]
    
    print(f"\n测试{len(test_messages)}条消息的处理时间...")
    
    # 测试优化后的异步版本
    print("\n--- 测试优化后的异步版本 (并行Memory Classification) ---")
    total_time_optimized = 0
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n消息 {i}: {message[:30]}...")
        
        start_time = time.time()
        agent_optimized = create_test_agent("优化Agent", "../outputs/optimized_memory.csv")
        setup_test_memories(agent_optimized, 10)
        response = await agent_optimized.arun(message, stream=False)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time_optimized += elapsed_time
        
        print(f"  响应时间: {elapsed_time:.2f}秒")
        print(f"  响应: {response.content[:80]}...")
    
    print(f"\n优化版本总时间: {total_time_optimized:.2f}秒")
    print(f"平均时间: {total_time_optimized/len(test_messages):.2f}秒")
    
    return total_time_optimized

async def test_memory_classification_timing():
    """专门测试Memory Classification的时间"""
    print("\n=== 测试Memory Classification时间 ===")
    
    test_input = "我喜欢编程和阅读，这是我的爱好"
    
    # 测试同步分类
    print("测试同步Memory Classification...")
    messages = [
        "你好，我是张三",
        "我喜欢运动和音乐",
        "你还记得我之前说过什么吗？"
    ]

    total_time = 0
    for i, message in enumerate(messages, 1):
        print(f"\n消息 {i}: {message}")
        start_time = time.time()
        agent = create_test_agent("同步", "../outputs/classification_test.csv")
        setup_test_memories(agent, 5)


        sync_result = agent.memory.should_update_memory(test_input)
        sync_time = time.time() - start_time
        total_time += sync_time
        print(f"  同步分类时间: {sync_time:.3f}秒, 结果: {sync_result}")
    print(f"\n同步Agent总时间: {total_time:.2f}秒")
    print(f"agent memories: {agent.memory.memories}")

    messages = [
        "你好，我是张三",
        "我喜欢运动和音乐",
        "你还记得我之前说过什么吗？"
    ]

    total_time = 0
    for i, message in enumerate(messages, 1):
        print(f"\n消息 {i}: {message}")
        # 测试异步分类
        print("测试异步Memory Classification...")
        start_time = time.time()
        agent = create_test_agent("异步", "../outputs/classification_test.csv")
        setup_test_memories(agent, 5)

        async_result = await agent.memory.ashould_update_memory(test_input)
        elapsed_time = time.time() - start_time
        print(f"  异步分类时间: {elapsed_time:.3f}秒, 结果: {async_result}")
        total_time += elapsed_time

    print(f"\n异步Agent总时间: {total_time:.2f}秒")
    print(f"agent memories: {agent.memory.memories}")

async def test_real_world_scenario():
    """测试真实世界场景"""
    print("\n=== 真实世界场景测试 ===")
    

    # 模拟真实对话场景
    conversation = [
        "我是一名软件工程师，专门做AI开发",
        "谢谢你的分析，你还记得我的职业吗？"
    ]
    
    print("开始真实对话测试...")
    total_time = 0
    agent = create_test_agent("异步Agent", "../outputs/real_world_test.csv")
    setup_test_memories(agent, 15)
    for i, message in enumerate(conversation, 1):
        print(f"\n轮次 {i}: {message}")
        
        start_time = time.time()

        response = await agent.arun(message, stream=False)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        
        print(f"  时间: {elapsed_time:.2f}秒")
        print(f"  回复: {response.content[:100]}...")
        
        # 检查记忆是否正确更新
        memory_count = len(agent.memory.memories) if agent.memory.memories else 0
        print(f"  当前记忆数量: {memory_count}")
    
    print(f"\n真实对话总时间: {total_time:.2f}秒")
    print(f"平均每轮时间: {total_time/len(conversation):.2f}秒")
    
    # 验证记忆效果
    final_memory_count = len(agent.memory.memories) if agent.memory.memories else 0
    print(f"最终记忆数量: {final_memory_count}")

async def main():
    """主函数"""
    print("Memory Classification 并行优化测试")
    print("=" * 50)
    
    # 确保输出目录存在
    os.makedirs("../outputs", exist_ok=True)
    
    try:
        # 1. 测试Memory Classification时间
        # await test_memory_classification_timing()
        #
        # # 2. 测试并行优化效果
        # optimized_time = await test_parallel_optimization()
        
        # 3. 测试真实世界场景
        await test_real_world_scenario()
        

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main())
