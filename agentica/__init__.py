# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Agentica - Build AI Agents with ease.

═══════════════════════════════════════════════════════════════
v1.3.6+ Recommended Import Style (clearer; aligns with v2.0 plan)
═══════════════════════════════════════════════════════════════

Core (always default-installed, eager top-level imports)::

    from agentica import Agent, DeepAgent, tool, Workspace    # core SDK
    from agentica import OpenAIChat                            # default LLM (openai is a hard dep)
    from agentica import (                                     # builtin tools (no extra deps)
        BuiltinFileTool, BuiltinExecuteTool,
        BuiltinFetchUrlTool, BuiltinWebSearchTool,
        BuiltinTodoTool, BuiltinTaskTool, BuiltinMemoryTool,
    )

Other LLM providers (lazy, avoid heavy SDK import at startup)::

    from agentica.model.anthropic.claude import Claude        # pip install anthropic
    from agentica.model.ollama.chat import Ollama
    from agentica.model.kimi.chat import KimiChat
    from agentica.tools.shell_tool import ShellTool           # specific external tools

Optional extras (need ``pip install agentica[xxx]``)::

    from agentica.knowledge import Knowledge       # pip install agentica[rag]
    from agentica.vectordb import InMemoryVectorDb # pip install agentica[rag]
    from agentica.mcp import MCPClient             # pip install agentica[mcp]
    from agentica.acp import ACPServer             # pip install agentica[acp]
    from agentica.gateway.main import app          # pip install agentica[gateway]
    from agentica.db import SqliteDb               # pip install agentica[sql]

═══════════════════════════════════════════════════════════════
Backward Compatibility
═══════════════════════════════════════════════════════════════

Old-style top-level imports (e.g. ``from agentica import Knowledge``) also
work via lazy ``__getattr__`` and are considered stable v1.x public API.
Both styles are supported long-term; pick whichever you prefer.

See ``docs/API.md`` for the Tier 1/2/3 stability contract.
"""

import importlib
import threading
from typing import TYPE_CHECKING

from . import api_registry

_api_registry = api_registry
del api_registry

# ── Version ──
from agentica.version import __version__  # noqa: F401

# ── Config ──
from agentica.config import (
    AGENTICA_HOME,
    AGENTICA_DOTENV_PATH,
    AGENTICA_LOG_LEVEL,
    AGENTICA_LOG_FILE,
    AGENTICA_WORKSPACE_DIR,
    AGENTICA_PROJECTS_DIR,
    AGENTICA_CRON_DIR,
)

# ── Logging ──
from agentica.utils.log import set_log_level_to_debug, logger, set_log_level_to_info
from agentica.utils.io import write_audio_to_file

# ── Core Model ──
# OpenAIChat is eager: openai is a hard runtime dependency (requirements.txt).
# Other model providers stay lazy to avoid pulling heavy SDKs (anthropic, ollama, litellm) at import time.
from agentica.model.message import Message, MessageReferences, UserMessage, AssistantMessage, SystemMessage, ToolMessage
from agentica.model.content import Media, Video, Audio, Image
from agentica.model.usage import Usage, RequestUsage, TokenDetails
from agentica.model.providers import create_provider, list_providers
from agentica.model.openai.chat import OpenAIChat

# ── Backward-compatible provider aliases ──
def DeepSeekChat(**kwargs):
    return create_provider(_api_registry.PROVIDER_ALIAS_TO_SLUG["DeepSeekChat"], **kwargs)

DeepSeek = DeepSeekChat

def MoonshotChat(**kwargs):
    return create_provider(_api_registry.PROVIDER_ALIAS_TO_SLUG["MoonshotChat"], **kwargs)

Moonshot = MoonshotChat

def ArkChat(**kwargs):
    return create_provider(_api_registry.PROVIDER_ALIAS_TO_SLUG["ArkChat"], **kwargs)

Ark = ArkChat

def TogetherChat(**kwargs):
    return create_provider(_api_registry.PROVIDER_ALIAS_TO_SLUG["TogetherChat"], **kwargs)

Together = TogetherChat

def GrokChat(**kwargs):
    return create_provider(_api_registry.PROVIDER_ALIAS_TO_SLUG["GrokChat"], **kwargs)

Grok = GrokChat

def YiChat(**kwargs):
    return create_provider(_api_registry.PROVIDER_ALIAS_TO_SLUG["YiChat"], **kwargs)

Yi = YiChat

def QwenChat(**kwargs):
    return create_provider(_api_registry.PROVIDER_ALIAS_TO_SLUG["QwenChat"], **kwargs)

Qwen = QwenChat

def ZhipuAIChat(**kwargs):
    return create_provider(_api_registry.PROVIDER_ALIAS_TO_SLUG["ZhipuAIChat"], **kwargs)

ZhipuAI = ZhipuAIChat

# ── Memory ──
from agentica.memory import (
    AgentRun, SessionSummary, MemorySummarizer, WorkingMemory,
    MemoryType, MemoryEntry,
    WorkflowRun, WorkflowMemory,
)

# ── Database (base types only) ──
from agentica.db.base import BaseDb, SessionRow, MemoryRow, MetricsRow

# ── Run Response ──
from agentica.run_response import RunResponse, RunEvent, RunResponseExtraData, ToolCallInfo, pprint_run_response
from agentica.run_context import RunContext, RunSource, RunStatus, TaskAnchor
from agentica.run_events import RunEventRecord, RunEventType
from agentica.learning_report import LearningReport, LearningStatus, write_learning_report

# ── Document ──
from agentica.document import Document

# ── Tool base ──
from agentica.tools.base import Tool, ModelTool, Function, FunctionCall
from agentica.tools.decorators import tool  # @tool decorator for defining tool functions

# ── Builtin tools (eager: lightweight, used by 90% of custom Agent setups) ──
from agentica.tools.buildin_tools import (
    BuiltinFileTool, BuiltinExecuteTool, BuiltinFetchUrlTool,
    BuiltinWebSearchTool, BuiltinTodoTool, BuiltinMemoryTool,
)
from agentica.tools.builtin_task_tool import BuiltinTaskTool

# ── Compression ──
from agentica.compression import CompressionManager

# ── Token counting ──
from agentica.utils.tokens import count_tokens, count_text_tokens, count_image_tokens, count_message_tokens, count_tool_tokens

# ── Agent (core) ──
from agentica.agent import Agent, AgentCancelledError
from agentica.agent.deep import DeepAgent
from agentica.agent.config import PromptConfig, ToolConfig, WorkspaceMemoryConfig, HistoryConfig, SandboxConfig, ToolRuntimeConfig, SkillRuntimeConfig, ExperienceConfig, SkillUpgradeConfig
from agentica.run_config import RunConfig
from agentica.workflow import Workflow, WorkflowSession
from agentica.hooks import AgentHooks, RunHooks, ConversationArchiveHooks, MemoryExtractHooks, ExperienceCaptureHooks

# ── Experience system ──
from agentica.experience import (
    ExperienceEventStore,
    ExperienceCompiler,
    CompiledExperienceStore,
    SkillEvolutionManager,
    SkillLifecycleHooks,
    NoopSkillLifecycleHooks,
)

# ── Workspace ──
from agentica.workspace import Workspace, WorkspaceConfig

# ── Critic (actor-critic protocol + refine composer) ──
from agentica.critic import (
    Critic,
    CritiqueResult,
    CritiqueStyle,
    RefineResult,
    RefineRound,
    SchemaCritic,
    ExecCritic,
    AgentCritic,
    refine,
)

_LAZY_CACHE = {}
_LAZY_LOCK = threading.Lock()


def __getattr__(name: str):
    """Lazy import handler for optional modules."""
    if name in _api_registry.LAZY_IMPORTS:
        if name not in _LAZY_CACHE:
            with _LAZY_LOCK:
                if name not in _LAZY_CACHE:
                    module_path = _api_registry.LAZY_IMPORTS[name]
                    module = importlib.import_module(module_path)
                    attr_name = _api_registry.LAZY_ATTR_OVERRIDES.get(name, name)
                    try:
                        _LAZY_CACHE[name] = getattr(module, attr_name)
                    except AttributeError:
                        # Fallback: original name
                        _LAZY_CACHE[name] = getattr(module, name)
        return _LAZY_CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List all available names including lazy imports."""
    eager_names = [name for name in globals() if not name.startswith('_')]
    return sorted(set(eager_names) | set(_api_registry.LAZY_IMPORTS.keys()))


if TYPE_CHECKING:
    from agentica.db.sqlite import SqliteDb  # noqa: F401
    from agentica.db.postgres import PostgresDb  # noqa: F401
    from agentica.db.memory import InMemoryDb  # noqa: F401
    from agentica.db.json import JsonDb  # noqa: F401
    from agentica.db.mysql import MysqlDb  # noqa: F401
    from agentica.db.redis import RedisDb  # noqa: F401
    from agentica.model.litellm.chat import LiteLLMChat  # noqa: F401
    from agentica.model.litellm.chat import LiteLLMChat as LiteLLM  # noqa: F401
    from agentica.model.kimi.chat import KimiChat  # noqa: F401
    from agentica.model.anthropic.claude import Claude  # noqa: F401
    from agentica.model.ollama.chat import Ollama  # noqa: F401
    from agentica.knowledge.base import Knowledge  # noqa: F401
    from agentica.knowledge.llamaindex_knowledge import LlamaIndexKnowledge  # noqa: F401
    from agentica.knowledge.langchain_knowledge import LangChainKnowledge  # noqa: F401
    from agentica.vectordb.base import SearchType, Distance, VectorDb  # noqa: F401
    from agentica.vectordb.memory_vectordb import InMemoryVectorDb  # noqa: F401
    from agentica.embedding.base import Embedding  # noqa: F401
    from agentica.embedding.openai import OpenAIEmbedding  # noqa: F401
    from agentica.embedding.azure_openai import AzureOpenAIEmbedding  # noqa: F401
    from agentica.embedding.hash import HashEmbedding  # noqa: F401
    from agentica.embedding.ollama import OllamaEmbedding  # noqa: F401
    from agentica.embedding.together import TogetherEmbedding  # noqa: F401
    from agentica.embedding.fireworks import FireworksEmbedding  # noqa: F401
    from agentica.embedding.zhipuai import ZhipuAIEmbedding  # noqa: F401
    from agentica.embedding.http import HttpEmbedding  # noqa: F401
    from agentica.embedding.jina import JinaEmbedding  # noqa: F401
    from agentica.embedding.gemini import GeminiEmbedding  # noqa: F401
    from agentica.embedding.huggingface import HuggingfaceEmbedding  # noqa: F401
    from agentica.embedding.mulanai import MulanAIEmbedding  # noqa: F401
    from agentica.rerank.base import Rerank  # noqa: F401
    from agentica.rerank.jina import JinaRerank  # noqa: F401
    from agentica.rerank.zhipuai import ZhipuAIRerank  # noqa: F401
    from agentica.guardrails import (  # noqa: F401
        GuardrailOutput, InputGuardrail, OutputGuardrail,
        InputGuardrailResult, OutputGuardrailResult,
        input_guardrail, output_guardrail,
        InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered,
        ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail,
        ToolInputGuardrailData, ToolOutputGuardrailData, ToolContext,
        tool_input_guardrail, tool_output_guardrail,
        ToolInputGuardrailTripwireTriggered, ToolOutputGuardrailTripwireTriggered,
        run_input_guardrails, run_output_guardrails,
        run_tool_input_guardrails, run_tool_output_guardrails,
    )
    from agentica.tools.search_serper_tool import SearchSerperTool  # noqa: F401
    from agentica.tools.dalle_tool import DalleTool  # noqa: F401
    from agentica.tools.shell_tool import ShellTool  # noqa: F401
    from agentica.tools.code_tool import CodeTool  # noqa: F401
    from agentica.tools.mcp_tool import McpTool, CompositeMultiMcpTool  # noqa: F401
    from agentica.mcp.config import MCPConfig  # noqa: F401


__all__ = _api_registry.PUBLIC_API_ALL
