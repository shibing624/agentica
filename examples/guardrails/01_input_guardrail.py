# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Input guardrail demo - Demonstrates input validation for agents

This example shows how to:
1. Create input guardrails using decorators
2. Block harmful or inappropriate inputs
3. Filter profanity and sensitive content
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    input_guardrail,
)


# ============================================================================
# Define Input Guardrails
# ============================================================================

@input_guardrail
def check_topic(ctx, agent, input_data) -> GuardrailFunctionOutput:
    """Check if the user is asking about appropriate topics."""
    input_str = str(input_data).lower()

    # Block harmful topics
    blocked_topics = ["hack", "cheat", "steal", "illegal", "violence"]
    for topic in blocked_topics:
        if topic in input_str:
            return GuardrailFunctionOutput.block(
                output_info={"reason": f"Blocked topic: {topic}"}
            )

    return GuardrailFunctionOutput.allow(output_info="Topic check passed")


@input_guardrail(name="profanity_filter", run_in_parallel=False)
def filter_profanity(ctx, agent, input_data) -> GuardrailFunctionOutput:
    """Filter out profanity from user input."""
    input_str = str(input_data).lower()
    
    # Example profanity words (in real use, use a comprehensive list)
    profanity_words = ["badword1", "badword2"]

    for word in profanity_words:
        if word in input_str:
            return GuardrailFunctionOutput.block(
                output_info={"reason": "Profanity detected"}
            )

    return GuardrailFunctionOutput.allow()


@input_guardrail(name="length_check")
def check_input_length(ctx, agent, input_data) -> GuardrailFunctionOutput:
    """Check if input is within acceptable length."""
    input_str = str(input_data)
    max_length = 5000

    if len(input_str) > max_length:
        return GuardrailFunctionOutput.block(
            output_info={"reason": f"Input too long: {len(input_str)} chars (max: {max_length})"}
        )

    return GuardrailFunctionOutput.allow()


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_input_guardrails():
    """Demonstrate input guardrails."""
    print("=" * 60)
    print("Input Guardrails Demo")
    print("=" * 60)

    # Test 1: Normal input
    print("\n1. Test normal input:")
    result = await check_topic.run(None, "What is the capital of France?", None)
    if result.output.tripwire_triggered:
        print(f"   🚨 Blocked: {result.output.output_info}")
    else:
        print(f"   ✅ Allowed: {result.output.output_info}")

    # Test 2: Blocked topic
    print("\n2. Test blocked topic 'hack':")
    result = await check_topic.run(None, "How do I hack into a computer?", None)
    if result.output.tripwire_triggered:
        print(f"   🚨 Blocked: {result.output.output_info}")
    else:
        print(f"   ✅ Allowed: {result.output.output_info}")

    # Test 3: Length check
    print("\n3. Test input length:")
    short_input = "Hello"
    result = await check_input_length.run(None, short_input, None)
    if result.output.tripwire_triggered:
        print(f"   🚨 Blocked: {result.output.output_info}")
    else:
        print(f"   ✅ Allowed (length: {len(short_input)})")

    # Test 4: Guardrail info
    print("\n4. Guardrail information:")
    print(f"   check_topic.get_name() = '{check_topic.get_name()}'")
    print(f"   filter_profanity.get_name() = '{filter_profanity.get_name()}'")
    print(f"   check_input_length.get_name() = '{check_input_length.get_name()}'")


async def main():
    """Run all demos."""
    print("Agentica Input Guardrails Demo")
    print("=" * 60)
    print("""
This demo shows how to use input guardrails to:
- Validate user inputs before processing
- Block harmful or inappropriate content
- Filter profanity and sensitive topics
    """)

    await demo_input_guardrails()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
