# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Parallelization demo - Demonstrates how to run multiple agents in parallel

This example shows:
1. Running multiple agents concurrently using asyncio.gather
2. Time comparison between parallel and sequential execution
3. Picking the best result from multiple outputs
"""
import sys
import os
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat


# Create a Chinese translator agent
chinese_agent = Agent(
    name="chinese_agent",
    model=OpenAIChat(id='gpt-4o'),
    instructions="You translate the user's message to Chinese. Only output the translation, nothing else.",
)

# Create a translation picker agent
translation_picker = Agent(
    name="translation_picker",
    model=OpenAIChat(id='gpt-4o'),
    instructions="""You are a translation expert. Pick the best Chinese translation from the given options.
Consider accuracy, fluency, and naturalness. Only output the best translation, nothing else.""",
)


async def main():
    """Main function to run parallel translations and pick the best one."""
    msg = ("Artificial intelligence has revolutionized the way we interact with technology, "
           "enabling machines to learn from experience, adapt to new inputs, and perform human-like tasks.")
    print(f"Input message: {msg}\n")

    # ========== Parallel execution ==========
    print("=" * 50)
    print("Running 3 translations in PARALLEL...")
    print("=" * 50)
    
    parallel_start = time.time()
    res_1, res_2, res_3 = await asyncio.gather(
        chinese_agent.arun(msg),
        chinese_agent.arun(msg),
        chinese_agent.arun(msg),
    )
    parallel_time = time.time() - parallel_start

    # Collect translation outputs
    outputs = [
        res_1.content,
        res_2.content,
        res_3.content,
    ]

    translations = "\n\n".join([f"Translation {i+1}: {t}" for i, t in enumerate(outputs)])
    print(f"\nTranslations:\n{translations}")
    print(f"\n[Parallel] 3 translations completed in {parallel_time:.2f}s")

    # ========== Sequential execution for comparison ==========
    print("\n" + "=" * 50)
    print("Running 3 translations SEQUENTIALLY for comparison...")
    print("=" * 50)
    
    sequential_start = time.time()
    seq_res_1 = await chinese_agent.arun(msg)
    seq_res_2 = await chinese_agent.arun(msg)
    seq_res_3 = await chinese_agent.arun(msg)
    sequential_time = time.time() - sequential_start

    seq_outputs = [seq_res_1.content, seq_res_2.content, seq_res_3.content]
    seq_translations = "\n\n".join([f"Translation {i+1}: {t}" for i, t in enumerate(seq_outputs)])
    print(f"\nTranslations:\n{seq_translations}")
    print(f"\n[Sequential] 3 translations completed in {sequential_time:.2f}s")

    # ========== Time comparison ==========
    print("\n" + "=" * 50)
    print("TIME COMPARISON")
    print("=" * 50)
    print(f"Parallel time:   {parallel_time:.2f}s")
    print(f"Sequential time: {sequential_time:.2f}s")
    print(f"Speedup:         {sequential_time / parallel_time:.2f}x faster with parallel execution")

    # ========== Pick the best translation ==========
    print("\n" + "=" * 50)
    print("Picking the best translation...")
    print("=" * 50)
    
    best_translation = await translation_picker.arun(
        f"Input: {msg}\n\nTranslations:\n{translations}"
    )

    print(f"\nBest translation: {best_translation.content}")


if __name__ == "__main__":
    asyncio.run(main())
