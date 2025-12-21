# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Output guardrail demo - Demonstrates output validation for agents

This example shows how to:
1. Create output guardrails using decorators
2. Filter sensitive data from outputs
3. Check output length and format
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import (
    GuardrailFunctionOutput,
    output_guardrail,
)


# ============================================================================
# Define Output Guardrails
# ============================================================================

@output_guardrail
def check_output_length(ctx, agent, output) -> GuardrailFunctionOutput:
    """Ensure output is not too long."""
    output_str = str(output)
    max_length = 10000

    if len(output_str) > max_length:
        return GuardrailFunctionOutput.block(
            output_info={"reason": f"Output too long: {len(output_str)} chars"}
        )

    return GuardrailFunctionOutput.allow()


@output_guardrail(name="sensitive_data_filter")
def filter_sensitive_output(ctx, agent, output) -> GuardrailFunctionOutput:
    """Filter sensitive data from agent output."""
    output_str = str(output).lower()

    sensitive_patterns = ["credit card", "social security", "password", "ssn"]
    for pattern in sensitive_patterns:
        if pattern in output_str:
            return GuardrailFunctionOutput.block(
                output_info={"reason": f"Sensitive data detected: {pattern}"}
            )

    return GuardrailFunctionOutput.allow()


@output_guardrail(name="pii_filter")
def filter_pii(ctx, agent, output) -> GuardrailFunctionOutput:
    """Filter personally identifiable information (PII) from output."""
    import re
    output_str = str(output)

    # Check for email patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, output_str):
        return GuardrailFunctionOutput.block(
            output_info={"reason": "Email address detected in output"}
        )

    # Check for phone number patterns (simplified)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    if re.search(phone_pattern, output_str):
        return GuardrailFunctionOutput.block(
            output_info={"reason": "Phone number detected in output"}
        )

    return GuardrailFunctionOutput.allow()


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_output_guardrails():
    """Demonstrate output guardrails."""
    print("=" * 60)
    print("Output Guardrails Demo")
    print("=" * 60)

    # Test 1: Normal output
    print("\n1. Test normal output:")
    result = await check_output_length.run(None, "Short response", None)
    if result.output.tripwire_triggered:
        print(f"   🚨 Blocked: {result.output.output_info}")
    else:
        print(f"   ✅ Allowed")

    # Test 2: Sensitive data
    print("\n2. Test output with sensitive data:")
    result = await filter_sensitive_output.run(None, "Your password is 12345", None)
    if result.output.tripwire_triggered:
        print(f"   🚨 Blocked: {result.output.output_info}")
    else:
        print(f"   ✅ Allowed")

    # Test 3: PII filter - email
    print("\n3. Test output with email:")
    result = await filter_pii.run(None, "Contact me at john@example.com", None)
    if result.output.tripwire_triggered:
        print(f"   🚨 Blocked: {result.output.output_info}")
    else:
        print(f"   ✅ Allowed")

    # Test 4: PII filter - phone
    print("\n4. Test output with phone number:")
    result = await filter_pii.run(None, "Call me at 555-123-4567", None)
    if result.output.tripwire_triggered:
        print(f"   🚨 Blocked: {result.output.output_info}")
    else:
        print(f"   ✅ Allowed")

    # Test 5: Clean output
    print("\n5. Test clean output:")
    result = await filter_pii.run(None, "The weather is nice today.", None)
    if result.output.tripwire_triggered:
        print(f"   🚨 Blocked: {result.output.output_info}")
    else:
        print(f"   ✅ Allowed")


async def main():
    """Run all demos."""
    print("Agentica Output Guardrails Demo")
    print("=" * 60)
    print("""
This demo shows how to use output guardrails to:
- Validate agent outputs before returning
- Filter sensitive data like passwords
- Remove PII (emails, phone numbers)
    """)

    await demo_output_guardrails()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
