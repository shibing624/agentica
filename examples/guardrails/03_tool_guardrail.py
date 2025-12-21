# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tool guardrail demo - Demonstrates tool-level input/output validation

This example shows how to:
1. Create tool input guardrails
2. Create tool output guardrails
3. Attach guardrails to Tool classes
"""
import asyncio
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import (
    Tool,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    ToolContext,
    tool_input_guardrail,
    tool_output_guardrail,
)


# ============================================================================
# Define Tool Guardrails
# ============================================================================

@tool_input_guardrail
def reject_sensitive_words(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Reject tool calls that contain sensitive words in arguments."""
    try:
        args = json.loads(data.context.tool_arguments) if data.context.tool_arguments else {}
    except json.JSONDecodeError:
        return ToolGuardrailFunctionOutput(output_info="Invalid JSON arguments")

    # Check for suspicious content
    sensitive_words = ["password", "hack", "exploit", "malware", "ACME"]
    for key, value in args.items():
        value_str = str(value).lower()
        for word in sensitive_words:
            if word.lower() in value_str:
                return ToolGuardrailFunctionOutput.reject_content(
                    message=f"🚨 Tool call blocked: contains '{word}'",
                    output_info={"blocked_word": word, "argument": key},
                )

    return ToolGuardrailFunctionOutput.allow(output_info="Input validated")


@tool_output_guardrail
def block_sensitive_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Block tool outputs that contain sensitive data like SSN."""
    output_str = str(data.output).lower()

    # Check for sensitive data patterns
    if "ssn" in output_str or "123-45-6789" in output_str:
        return ToolGuardrailFunctionOutput.raise_exception(
            output_info={"blocked_pattern": "SSN", "tool": data.context.tool_name},
        )

    return ToolGuardrailFunctionOutput.allow(output_info="Output validated")


@tool_output_guardrail
def reject_phone_numbers(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Reject function output containing phone numbers."""
    output_str = str(data.output)
    if "555-1234" in output_str:
        return ToolGuardrailFunctionOutput.reject_content(
            message="User data not retrieved as it contains a phone number which is restricted.",
            output_info={"redacted": "phone_number"},
        )
    return ToolGuardrailFunctionOutput.allow(output_info="Phone number check passed")


# ============================================================================
# Define Tools with Guardrails
# ============================================================================

class EmailTool(Tool):
    """A tool for sending emails with input guardrails."""

    def __init__(self):
        super().__init__(name="email_tool")
        self.register(self.send_email)
        self.tool_input_guardrails = [reject_sensitive_words]

    def send_email(self, to: str, subject: str, body: str) -> str:
        """Send an email to the specified recipient."""
        return f"Email sent to {to} with subject '{subject}'"


class UserDataTool(Tool):
    """A tool for getting user data with output guardrails."""

    def __init__(self):
        super().__init__(name="user_data_tool")
        self.register(self.get_user_data)
        self.tool_output_guardrails = [block_sensitive_output]

    def get_user_data(self, user_id: str) -> dict:
        """Get user data by ID."""
        return {
            "user_id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "ssn": "123-45-6789",  # Sensitive data that should be blocked!
        }


# ============================================================================
# Demo Functions
# ============================================================================

async def demo_tool_guardrails():
    """Demonstrate tool guardrails."""
    print("=" * 60)
    print("Tool Guardrails Demo")
    print("=" * 60)

    # Test 1: Tool input guardrail - allow
    print("\n1. Tool input guardrail - Normal arguments (should allow):")
    context = ToolContext(tool_name="send_email", tool_arguments='{"to": "john@example.com", "subject": "Hello"}')
    data = ToolInputGuardrailData(context=context, agent=None)
    result = await reject_sensitive_words.run(data)
    if result.output.is_allow():
        print(f"   ✅ Allowed: {result.output.output_info}")
    elif result.output.is_reject_content():
        print(f"   ❌ Rejected: {result.output.get_reject_message()}")

    # Test 2: Tool input guardrail - reject
    print("\n2. Tool input guardrail - Contains 'ACME' (should reject):")
    context = ToolContext(tool_name="send_email", tool_arguments='{"to": "john@example.com", "body": "About ACME corp"}')
    data = ToolInputGuardrailData(context=context, agent=None)
    result = await reject_sensitive_words.run(data)
    if result.output.is_allow():
        print(f"   ✅ Allowed: {result.output.output_info}")
    elif result.output.is_reject_content():
        print(f"   🚨 Rejected: {result.output.get_reject_message()}")
        print(f"      Details: {result.output.output_info}")

    # Test 3: Tool output guardrail - raise exception
    print("\n3. Tool output guardrail - Contains SSN (should raise exception):")
    context = ToolContext(tool_name="get_user_data")
    output = {"user_id": "123", "ssn": "123-45-6789"}
    data = ToolOutputGuardrailData(context=context, agent=None, output=output)
    result = await block_sensitive_output.run(data)
    if result.output.is_raise_exception():
        print(f"   🚨 Exception triggered!")
        print(f"      Details: {result.output.output_info}")
    elif result.output.is_allow():
        print(f"   ✅ Allowed")

    # Test 4: Tool output guardrail - reject content
    print("\n4. Tool output guardrail - Contains phone number (should reject):")
    context = ToolContext(tool_name="get_contact_info")
    output = {"name": "Jane", "phone": "555-1234"}
    data = ToolOutputGuardrailData(context=context, agent=None, output=output)
    result = await reject_phone_numbers.run(data)
    if result.output.is_reject_content():
        print(f"   🚨 Rejected: {result.output.get_reject_message()}")
    elif result.output.is_allow():
        print(f"   ✅ Allowed")


async def main():
    """Run all demos."""
    print("Agentica Tool Guardrails Demo")
    print("=" * 60)
    print("""
This demo shows how to use tool guardrails to:
- Validate tool arguments before execution
- Filter sensitive data from tool outputs
- Block or reject content based on rules
    """)

    await demo_tool_guardrails()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
