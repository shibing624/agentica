# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Guardrails Demo - Input/Output validation for Agents and Tools

This example demonstrates:
1. Agent-level Input Guardrails - Check user input before agent processing
2. Agent-level Output Guardrails - Validate agent output before returning
3. Tool-level Input Guardrails - Validate tool arguments before execution
4. Tool-level Output Guardrails - Check tool output for sensitive data

Run: python 52_guardrails_demo.py
"""
import asyncio
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import (
    Agent,
    # Agent Guardrails
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    input_guardrail,
    output_guardrail,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    # Tool Guardrails
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    ToolContext,
    tool_input_guardrail,
    tool_output_guardrail,
    ToolOutputGuardrailTripwireTriggered,
    # Tools
    Tool,
)


# =============================================================================
# Part 1: Define Tools with Guardrails
# =============================================================================

class EmailTool(Tool):
    """A tool for sending emails with input guardrails."""

    def __init__(self):
        super().__init__(name="email_tool")
        self.register(self.send_email)
        # Attach guardrails to the tool
        self.tool_input_guardrails = [reject_sensitive_words]

    def send_email(self, to: str, subject: str, body: str) -> str:
        """Send an email to the specified recipient.

        Args:
            to: Email recipient address.
            subject: Email subject line.
            body: Email body content.

        Returns:
            Confirmation message.
        """
        return f"Email sent to {to} with subject '{subject}'"


class UserDataTool(Tool):
    """A tool for getting user data with output guardrails."""

    def __init__(self):
        super().__init__(name="user_data_tool")
        self.register(self.get_user_data)
        # Attach output guardrail
        self.tool_output_guardrails = [block_sensitive_output]

    def get_user_data(self, user_id: str) -> dict:
        """Get user data by ID.

        Args:
            user_id: The user's unique identifier.

        Returns:
            User data dictionary.
        """
        # Simulate returning sensitive data
        return {
            "user_id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "ssn": "123-45-6789",  # Sensitive data that should be blocked!
            "phone": "555-1234",
        }


class ContactTool(Tool):
    """A tool for getting contact info with output guardrails."""

    def __init__(self):
        super().__init__(name="contact_tool")
        self.register(self.get_contact_info)
        # Attach output guardrail
        self.tool_output_guardrails = [reject_phone_numbers]

    def get_contact_info(self, user_id: str) -> dict:
        """Get contact info by ID.

        Args:
            user_id: The user's unique identifier.

        Returns:
            Contact information dictionary.
        """
        return {
            "user_id": user_id,
            "name": "Jane Smith",
            "email": "jane@example.com",
            "phone": "555-1234",
        }


# =============================================================================
# Part 2: Define Tool Guardrails
# =============================================================================

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
                # Reject tool call and inform the model
                return ToolGuardrailFunctionOutput.reject_content(
                    message=f"🚨 Tool call blocked: contains '{word}'",
                    output_info={"blocked_word": word, "argument": key},
                )

    return ToolGuardrailFunctionOutput.allow(output_info="Input validated")


@tool_output_guardrail
def block_sensitive_output(data: ToolOutputGuardrailData) -> ToolGuardrailFunctionOutput:
    """Block tool outputs that contain sensitive data like SSN."""
    output_str = str(data.output).lower()

    # Check for sensitive data patterns (SSN)
    if "ssn" in output_str or "123-45-6789" in output_str:
        # Use raise_exception to halt execution completely for sensitive data
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


# =============================================================================
# Part 3: Define Agent Guardrails
# =============================================================================

@input_guardrail
def check_homework_topic(ctx, agent, input_data) -> GuardrailFunctionOutput:
    """Check if the user is asking about homework/educational topics."""
    input_str = str(input_data).lower()

    # Block non-educational requests
    blocked_topics = ["hack", "cheat", "steal", "illegal"]
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
    profanity_words = ["badword1", "badword2"]  # Example placeholder

    for word in profanity_words:
        if word in input_str:
            return GuardrailFunctionOutput.block(
                output_info={"reason": "Profanity detected"}
            )

    return GuardrailFunctionOutput.allow()


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

    sensitive_patterns = ["credit card", "social security", "password"]
    for pattern in sensitive_patterns:
        if pattern in output_str:
            return GuardrailFunctionOutput.block(
                output_info={"reason": f"Sensitive data detected: {pattern}"}
            )

    return GuardrailFunctionOutput.allow()


# =============================================================================
# Part 4: Demo Functions
# =============================================================================

async def demo_tool_guardrails():
    """Demonstrate tool guardrails (standalone, without full Agent integration)."""
    print("\n" + "=" * 60)
    print("Tool Guardrails Demo (Standalone)")
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


async def demo_agent_guardrails():
    """Demonstrate agent-level guardrails (standalone, without Agent integration)."""
    print("\n" + "=" * 60)
    print("Agent Guardrails Demo (Standalone)")
    print("=" * 60)

    # Demonstrate running guardrails manually
    print("\n1. Test input guardrail - Normal question:")
    try:
        result = await check_homework_topic.run(None, "What is the capital of France?", None)
        if result.output.tripwire_triggered:
            print(f"   🚨 Blocked: {result.output.output_info}")
        else:
            print(f"   ✅ Allowed: {result.output.output_info}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n2. Test input guardrail - Blocked topic 'hack':")
    try:
        result = await check_homework_topic.run(None, "How do I hack into a computer?", None)
        if result.output.tripwire_triggered:
            print(f"   🚨 Blocked: {result.output.output_info}")
        else:
            print(f"   ✅ Allowed: {result.output.output_info}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n3. Test output guardrail - Normal output:")
    try:
        result = await check_output_length.run(None, "Short response", None)
        if result.output.tripwire_triggered:
            print(f"   🚨 Blocked: {result.output.output_info}")
        else:
            print(f"   ✅ Allowed")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n4. Test output guardrail - Sensitive data detected:")
    try:
        result = await filter_sensitive_output.run(None, "Your password is 12345", None)
        if result.output.tripwire_triggered:
            print(f"   🚨 Blocked: {result.output.output_info}")
        else:
            print(f"   ✅ Allowed")
    except Exception as e:
        print(f"   ❌ Error: {e}")


async def demo_guardrail_decorators():
    """Demonstrate guardrail decorator usage."""
    print("\n" + "=" * 60)
    print("Guardrail Decorators Demo")
    print("=" * 60)

    # Show how decorators work
    print("\n1. Input guardrail decorator:")
    print(f"   check_homework_topic.get_name() = '{check_homework_topic.get_name()}'")
    print(f"   check_homework_topic.run_in_parallel = {check_homework_topic.run_in_parallel}")

    print("\n2. Named input guardrail:")
    print(f"   filter_profanity.get_name() = '{filter_profanity.get_name()}'")
    print(f"   filter_profanity.run_in_parallel = {filter_profanity.run_in_parallel}")

    print("\n3. Output guardrail decorator:")
    print(f"   check_output_length.get_name() = '{check_output_length.get_name()}'")

    print("\n4. Named output guardrail:")
    print(f"   filter_sensitive_output.get_name() = '{filter_sensitive_output.get_name()}'")

    print("\n5. Tool input guardrail:")
    print(f"   reject_sensitive_words.get_name() = '{reject_sensitive_words.get_name()}'")

    print("\n6. Tool output guardrail:")
    print(f"   block_sensitive_output.get_name() = '{block_sensitive_output.get_name()}'")


async def main():
    """Run all demos."""
    print("=" * 60)
    print("Agentica Guardrails Demo")
    print("=" * 60)
    print("""
This demo shows how to use guardrails to:
- Validate agent inputs before processing
- Check agent outputs before returning
- Validate tool arguments before execution
- Filter sensitive data from tool outputs
    """)

    # Demo 1: Decorator usage
    await demo_guardrail_decorators()

    # Demo 2: Agent guardrails
    await demo_agent_guardrails()

    # Demo 3: Tool guardrails
    await demo_tool_guardrails()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
