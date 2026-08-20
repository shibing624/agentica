import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestDeepAgentDefaults(unittest.TestCase):
    """DeepAgent should be the batteries-included default."""

    def test_deep_agent_defaults_enable_skills_and_auto_load_mcp(self):
        from agentica.agent.deep import DeepAgent
        from agentica.tools.skill_tool import SkillTool

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ) as load_mcp_tools, patch(
            "agentica.agent.base.Agent._merge_tool_system_prompts"
        ):
            agent = DeepAgent(model=MagicMock(), workspace=tmpdir)

        self.assertTrue(agent.tool_config.auto_load_mcp)
        self.assertTrue(any(isinstance(tool, SkillTool) for tool in agent.tools))
        load_mcp_tools.assert_called_once()

    def test_deep_agent_with_a_registry_registers_delegate(self):
        """The SDK path: no config.yaml involved — the model object itself is
        the credential source the delegated worker inherits."""
        from agentica.agent.deep import DeepAgent
        from agentica.model.openai import OpenAIChat
        from agentica.tools.background_processes import BackgroundProcessRegistry

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ), patch("agentica.agent.base.Agent._merge_tool_system_prompts"):
            agent = DeepAgent(
                model=OpenAIChat(
                    id="internal-only-model", api_key="sk-sdk", base_url="http://llm.internal/v1"
                ),
                workspace=tmpdir,
                include_skills=False,
                background_process_registry=BackgroundProcessRegistry(),
            )

        delegates = [t for t in agent.tools if t.name == "builtin_delegate_tool"]
        self.assertEqual(len(delegates), 1)

    def test_deep_agent_without_a_registry_has_no_delegate(self):
        from agentica.agent.deep import DeepAgent

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ), patch("agentica.agent.base.Agent._merge_tool_system_prompts"):
            agent = DeepAgent(model=MagicMock(), workspace=tmpdir, include_skills=False)

        self.assertNotIn(
            "builtin_delegate_tool", [t.name for t in agent.tools]
        )

    def test_deep_agent_model_exposes_file_tools(self):
        from agentica.agent.deep import DeepAgent
        from agentica.model.openai import OpenAIChat

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ):
            agent = DeepAgent(
                model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
                workspace=tmpdir,
                include_skills=False,
            )
            agent.update_model()

        self.assertIn("read_file", agent.model.functions)
        self.assertIn("ls", agent.model.functions)
        tool_names = {tool["function"]["name"] for tool in agent.model.tools}
        self.assertIn("read_file", tool_names)
        self.assertIn("ls", tool_names)

    def test_deep_agent_enables_experience_capture_by_default(self):
        """DeepAgent is the self-evolving flagship: experience + all capture_* on."""
        from agentica.agent.deep import DeepAgent

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ), patch(
            "agentica.agent.base.Agent._merge_tool_system_prompts"
        ):
            agent = DeepAgent(model=MagicMock(), workspace=tmpdir)

        self.assertTrue(
            agent.enable_experience_capture,
            "DeepAgent.enable_experience_capture must default True",
        )
        self.assertTrue(agent.experience_config.capture_tool_errors)
        self.assertTrue(agent.experience_config.capture_user_corrections)
        self.assertFalse(agent.experience_config.capture_success_patterns)
        self.assertIsNone(agent.experience_config.skill_upgrade)
        # auto_extract_memory: fallback memory extraction after each run.
        self.assertTrue(agent.long_term_memory_config.auto_extract_memory)
        self.assertTrue(agent.long_term_memory_config.auto_archive)

    def test_deep_agent_respects_explicit_experience_false(self):
        """Passing enable_experience_capture=False must override the default."""
        from agentica.agent.deep import DeepAgent

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ), patch(
            "agentica.agent.base.Agent._merge_tool_system_prompts"
        ):
            agent = DeepAgent(
                model=MagicMock(),
                workspace=tmpdir,
                enable_experience_capture=False,
            )

        self.assertFalse(agent.enable_experience_capture)

    def test_deep_agent_wires_default_auxiliary_model(self):
        """DeepAgent must default auxiliary_model to the main model instance.

        No hardcoded provider/model: reusing the main model lets the whole
        stack run on a single API key. Users pass an explicit auxiliary_model
        to offload side tasks onto a cheaper sibling.
        """
        from agentica.agent.deep import DeepAgent

        main_model = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ), patch(
            "agentica.agent.base.Agent._merge_tool_system_prompts"
        ):
            agent = DeepAgent(model=main_model, workspace=tmpdir)

        self.assertIs(agent.auxiliary_model, main_model)
        # CompressionManager must have been wired with the auxiliary model.
        cm = agent.tool_config.compression_manager
        self.assertIsNotNone(cm)
        self.assertIs(cm.model, agent.auxiliary_model)

    def test_deep_agent_explicit_auxiliary_model_passthrough(self):
        """An explicit auxiliary_model must be honored, not overridden."""
        from agentica.agent.deep import DeepAgent
        from agentica.model.openai import OpenAIChat

        custom_auxiliary = OpenAIChat(id="gpt-4o-mini-custom", api_key="fake_openai_key")
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ), patch(
            "agentica.agent.base.Agent._merge_tool_system_prompts"
        ):
            agent = DeepAgent(
                model=MagicMock(),
                workspace=tmpdir,
                auxiliary_model=custom_auxiliary,
            )

        self.assertIs(agent.auxiliary_model, custom_auxiliary)


class TestDeepAgentSignature(unittest.TestCase):
    """DeepAgent forwards to Agent by name, not through **kwargs."""

    def test_deep_agent_declares_every_agent_parameter(self):
        """Parity is the reason **kwargs could be dropped without losing reach.

        Adding a parameter to Agent without adding it here silently makes it
        unreachable through the preset, which is why this asserts the set
        difference rather than a hand-picked sample.
        """
        import inspect
        from agentica.agent.base import Agent
        from agentica.agent.deep import DeepAgent

        agent_params = set(inspect.signature(Agent.__init__).parameters) - {"self"}
        deep_params = set(inspect.signature(DeepAgent.__init__).parameters) - {"self"}

        self.assertEqual(agent_params - deep_params, set())
        self.assertFalse(
            any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in inspect.signature(DeepAgent.__init__).parameters.values()
            ),
            "DeepAgent must not take **kwargs: a forwarded typo then fails "
            "inside Agent.__init__ with no mention of DeepAgent",
        )

    def test_unknown_parameter_raises_naming_deep_agent(self):
        from agentica.agent.deep import DeepAgent

        # Built at runtime so the static keyword-arg checker does not flag the
        # deliberate typo this test exists to catch.
        bad_kwargs = {"model": MagicMock(), "include_web_serch": False}
        with self.assertRaises(TypeError) as ctx:
            DeepAgent(**bad_kwargs)

        message = str(ctx.exception)
        self.assertIn("DeepAgent", message)
        self.assertIn("include_web_serch", message)

    def test_passthrough_agent_parameters_take_effect(self):
        """The params in-repo callers rely on must reach Agent, not vanish."""
        from agentica.agent.deep import DeepAgent

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ), patch(
            "agentica.agent.base.Agent._merge_tool_system_prompts"
        ):
            agent = DeepAgent(
                model=MagicMock(),
                workspace=tmpdir,
                description="a description",
                instructions=["do the thing"],
                session_id="sig-smoke",
                session_base_dir=tmpdir,
                enable_session_log=False,
            )

        self.assertEqual(agent.description, "a description")
        self.assertEqual(agent.instructions, ["do the thing"])
        self.assertIsNone(agent._session_log)


if __name__ == "__main__":
    unittest.main()
