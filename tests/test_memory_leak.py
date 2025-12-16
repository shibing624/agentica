# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Test for memory leak issues in Agent framework

This test checks for circular references that could cause memory leaks,
similar to the issue fixed in OpenAI Agents SDK (PR #2014).

The key fix is using weakref for Function._agent to break the circular reference:
Agent -> Model -> functions (Dict[str, Function]) -> Function._agent -> Agent

测试	说明
test_agent_team_no_circular_reference	验证 Agent team 不会创建循环引用
test_function_agent_weakref	验证 Function._agent 使用 weakref
test_model_functions_agent_weakref	验证 Model.functions 中的 weakref 行为
test_run_response_no_agent_reference	验证 RunResponse 不持有 Agent 引用
test_agent_memory_no_agent_reference	验证 AgentMemory 不持有 Agent 引用
test_agent_with_real_model_no_memory_leak	验证真实 Model（不调用 API）无内存泄漏
test_multiple_agents_no_memory_leak	验证创建/销毁多个 Agent 无内存泄漏
test_agent_with_team_tools_no_leak	验证带 team 和 tools 的 Agent 无内存泄漏

"""
import gc
import weakref
from unittest import TestCase


class TestMemoryLeak(TestCase):
    """Test cases for memory leak detection in Agent framework."""

    def test_agent_team_no_circular_reference(self):
        """Test that Agent team doesn't create circular references that prevent GC."""
        from agentica import Agent

        # Create a leader agent with a team member
        member_agent = Agent(name="Member")
        leader_agent = Agent(name="Leader", team=[member_agent])

        # Create weak references to track if agents can be garbage collected
        leader_ref = weakref.ref(leader_agent)
        member_ref = weakref.ref(member_agent)

        # Delete strong references
        del leader_agent
        del member_agent

        # Force garbage collection
        gc.collect()

        # Agents should be collected (no circular reference)
        self.assertIsNone(leader_ref(), "Leader agent should be garbage collected")
        self.assertIsNone(member_ref(), "Member agent should be garbage collected")

    def test_function_agent_weakref(self):
        """Test that Function._agent uses weakref and doesn't prevent Agent GC."""
        from agentica import Agent
        from agentica.tools.base import Function

        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        agent = Agent(name="TestAgent", tools=[dummy_tool])

        # Update model to trigger tool processing
        agent.update_model()

        # Create weak reference
        agent_ref = weakref.ref(agent)

        # Verify Function._agent is accessible
        if agent.model and agent.model.functions:
            func = list(agent.model.functions.values())[0]
            self.assertIsNotNone(func._agent, "Function._agent should return the agent")
            self.assertEqual(func._agent.name, "TestAgent")

        # Delete strong reference
        del agent

        # Force garbage collection
        gc.collect()

        # Agent should be collected because Function uses weakref
        self.assertIsNone(agent_ref(), "Agent should be garbage collected (Function uses weakref)")

    def test_model_functions_agent_weakref(self):
        """Test Model.functions holding Agent references via Function._agent weakref."""
        from agentica import Agent
        from agentica.model.openai import OpenAIChat

        def my_tool(query: str) -> str:
            """Search tool."""
            return f"Result for {query}"

        # Create agent with tool
        agent = Agent(
            name="ToolAgent",
            model=OpenAIChat(model="gpt-4o-mini"),
            tools=[my_tool]
        )
        agent.update_model()

        # Get the model
        model = agent.model

        # Verify Function._agent references exist
        if model.functions:
            for name, func in model.functions.items():
                self.assertIsNotNone(func._agent, f"Function {name} should have _agent reference")

        # Create weak ref to agent
        agent_ref = weakref.ref(agent)

        # Delete agent but keep model reference
        del agent
        gc.collect()

        # Agent should be collected even though model is still referenced
        # because Function._agent uses weakref
        self.assertIsNone(agent_ref(), "Agent should be garbage collected (weakref)")

        # Function._agent should now return None
        if model.functions:
            for name, func in model.functions.items():
                self.assertIsNone(func._agent, f"Function {name}._agent should be None after agent GC")

    def test_run_response_no_agent_reference(self):
        """Verify RunResponse doesn't hold Agent reference (good design)."""
        from agentica.run_response import RunResponse

        # Check RunResponse fields
        fields = RunResponse.model_fields.keys()

        # Verify no 'agent' field - only agent_id (string)
        self.assertNotIn('agent', fields, "RunResponse should not have agent field")
        self.assertIn('agent_id', fields, "RunResponse should have agent_id field (string)")

    def test_agent_memory_no_agent_reference(self):
        """Test that AgentMemory.runs doesn't hold Agent references."""
        from agentica.memory import AgentMemory, AgentRun
        from agentica.run_response import RunResponse
        from agentica.model.message import Message

        memory = AgentMemory()

        # Add a run
        run = AgentRun(
            message=Message(role="user", content="test"),
            response=RunResponse(content="response")
        )
        memory.add_run(run)

        # AgentRun should not have agent field
        self.assertNotIn('agent', AgentRun.model_fields.keys())

    def test_agent_with_real_model_no_memory_leak(self):
        """Test that agent with real model (no API call) doesn't cause memory leak."""
        from agentica import Agent
        from agentica.model.openai import OpenAIChat

        def dummy_tool(x: str) -> str:
            """A dummy tool for testing."""
            return f"Result: {x}"

        # Create agent with real model (but don't call run() to avoid API call)
        agent = Agent(
            name="TestAgent",
            model=OpenAIChat(model="gpt-4o-mini"),
            tools=[dummy_tool]
        )

        # update_model() doesn't call API, just initializes tools
        agent.update_model()

        # Create weak reference
        agent_ref = weakref.ref(agent)

        # Verify Function._agent is working
        if agent.model and agent.model.functions:
            func = list(agent.model.functions.values())[0]
            self.assertIsNotNone(func._agent)

        # Delete agent
        del agent
        gc.collect()

        # Agent should be garbage collected (weakref in Function._agent)
        self.assertIsNone(agent_ref(), "Agent should be garbage collected")

    def test_multiple_agents_no_memory_leak(self):
        """Test creating and destroying multiple agents doesn't leak memory."""
        from agentica import Agent

        def tool_a(x: str) -> str:
            """Tool A."""
            return x

        def tool_b(y: int) -> int:
            """Tool B."""
            return y * 2

        refs = []

        # Create multiple agents
        for i in range(5):
            agent = Agent(
                name=f"Agent_{i}",
                tools=[tool_a, tool_b]
            )
            agent.update_model()
            refs.append(weakref.ref(agent))
            del agent

        # Force garbage collection
        gc.collect()

        # All agents should be collected
        for i, ref in enumerate(refs):
            self.assertIsNone(ref(), f"Agent_{i} should be garbage collected")

    def test_agent_with_team_tools_no_leak(self):
        """Test agent with team and tools doesn't leak memory."""
        from agentica import Agent

        def search_tool(query: str) -> str:
            """Search tool."""
            return f"Found: {query}"

        # Create team members with tools
        researcher = Agent(name="Researcher", tools=[search_tool])
        writer = Agent(name="Writer")

        # Create leader with team
        leader = Agent(
            name="Leader",
            team=[researcher, writer],
            tools=[search_tool]
        )
        leader.update_model()

        # Create weak references
        leader_ref = weakref.ref(leader)
        researcher_ref = weakref.ref(researcher)
        writer_ref = weakref.ref(writer)

        # Delete all agents
        del leader
        del researcher
        del writer
        gc.collect()

        # All should be collected
        self.assertIsNone(leader_ref(), "Leader should be garbage collected")
        self.assertIsNone(researcher_ref(), "Researcher should be garbage collected")
        self.assertIsNone(writer_ref(), "Writer should be garbage collected")


if __name__ == "__main__":
    import unittest
    unittest.main()
