# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Token Tracking Demo - Monitor token usage and costs

This example demonstrates:
1. Tracking token usage per request
2. Monitoring cumulative token usage
3. Token compression for long conversations
4. Cost estimation
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat
from agentica.utils.tokens import count_text_tokens


async def basic_token_tracking():
    """Demo: Basic token usage tracking."""
    print("=" * 60)
    print("Demo 1: Basic Token Tracking")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="TokenTracker",
        debug_mode=True,
    )

    # Run a query
    response = await agent.run("What is machine learning? Explain in 2-3 sentences.")

    # Get token usage from response
    if hasattr(response, 'metrics') and response.metrics:
        print(f"\nToken Usage:")
        print(f"  Input tokens: {response.metrics.get('input_tokens', 'N/A')}")
        print(f"  Output tokens: {response.metrics.get('output_tokens', 'N/A')}")
        print(f"  Total tokens: {response.metrics.get('total_tokens', 'N/A')}")

    print(f"\nResponse: {response.content}")


async def multi_turn_token_tracking():
    """Demo: Track tokens across multiple turns."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-turn Token Tracking")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="MultiTurnTracker",
        add_history_to_messages=True,
        debug_mode=True,
    )

    total_input_tokens = 0
    total_output_tokens = 0

    questions = [
        "What is Python?",
        "What are its main features?",
        "Give me a simple code example.",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n--- Turn {i} ---")
        print(f"Question: {question}")

        response = await agent.run(question)

        if hasattr(response, 'metrics') and response.metrics:
            input_tokens = response.metrics.get('input_tokens', 0)
            output_tokens = response.metrics.get('output_tokens', 0)
            # Handle case where tokens might be a list (cumulative) or int
            if isinstance(input_tokens, list):
                input_tokens = input_tokens[-1] if input_tokens else 0
            if isinstance(output_tokens, list):
                output_tokens = output_tokens[-1] if output_tokens else 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            print(f"Tokens this turn: {input_tokens} in / {output_tokens} out")

        print(f"Response: {response.content[:100]}...")

    print(f"\n--- Summary ---")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total tokens: {total_input_tokens + total_output_tokens}")


def token_counter_demo():
    """Demo: Using token counting utilities (pure computation, no async needed)."""
    print("\n" + "=" * 60)
    print("Demo 3: Token Counting Utilities")
    print("=" * 60)

    # Count tokens in text
    text = "This is a sample text to count tokens. It demonstrates the token counting utility."
    token_count = count_text_tokens(text)
    print(f"\nText: {text}")
    print(f"Token count: {token_count}")

    # Count tokens in messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
    ]

    total = 0
    for msg in messages:
        count = count_text_tokens(msg["content"])
        total += count
        print(f"  {msg['role']}: {count} tokens")
    print(f"  Total content tokens: {total}")


def cost_estimation_demo():
    """Demo: Estimate costs based on token usage (pure computation)."""
    print("\n" + "=" * 60)
    print("Demo 4: Cost Estimation")
    print("=" * 60)

    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }

    def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        if model not in PRICING:
            return 0.0
        rates = PRICING[model]
        input_cost = (input_tokens / 1_000_000) * rates["input"]
        output_cost = (output_tokens / 1_000_000) * rates["output"]
        return input_cost + output_cost

    model = "gpt-4o-mini"
    input_tokens = 1500
    output_tokens = 500

    cost = estimate_cost(model, input_tokens, output_tokens)

    print(f"\nModel: {model}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Estimated cost: ${cost:.6f}")

    print("\n--- Cost Comparison (same usage) ---")
    for model_name in PRICING:
        cost = estimate_cost(model_name, input_tokens, output_tokens)
        print(f"  {model_name}: ${cost:.6f}")


async def token_compression_demo():
    """Demo: Token compression for long conversations."""
    print("\n" + "=" * 60)
    print("Demo 5: Token Compression")
    print("=" * 60)

    long_conversation = """
    User: Can you explain what machine learning is?
    Assistant: Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.

    User: What are the main types?
    Assistant: There are three main types:
    1. Supervised Learning - learns from labeled data
    2. Unsupervised Learning - finds patterns in unlabeled data
    3. Reinforcement Learning - learns through trial and error

    User: Can you give examples of each?
    Assistant: Sure! Examples include:
    - Supervised: spam detection, image classification
    - Unsupervised: customer segmentation, anomaly detection
    - Reinforcement: game playing AI, robotics
    """

    original_tokens = count_text_tokens(long_conversation)
    print(f"\nOriginal conversation tokens: {original_tokens}")

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="Summarizer",
    )
    summary = await agent.run(
        f"Please summarize the following conversation in 2-3 sentences:\n{long_conversation}"
    )

    if summary and summary.content:
        compressed_tokens = count_text_tokens(summary.content)
        compression_ratio = (1 - compressed_tokens / original_tokens) * 100
        print(f"Compressed tokens: {compressed_tokens}")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print(f"\nSummary:\n{summary.content}")


async def main():
    await basic_token_tracking()
    await multi_turn_token_tracking()
    token_counter_demo()
    cost_estimation_demo()
    await token_compression_demo()


if __name__ == "__main__":
    asyncio.run(main())
