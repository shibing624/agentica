# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Agentica - Build AI Agents with ease.

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
    AGENTICA_CACHE_DIR,
)

# ── Unified config (config.yaml) shared by SDK + CLI ──
from agentica.global_config import (
    global_config_path,
    load_global_config,
    save_global_config,
    get_profile,
    get_profiles,
    get_active_profile_name,
    set_active_profile,
    upsert_profile,
    apply_global_config,
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
from agentica.model.openai.chat import OpenAIChat

# ── OpenAI-Compatible provider factories ──
# Each factory directly constructs OpenAIChat with hardcoded provider metadata.
# Users override defaults via kwargs: DeepSeekChat(id="deepseek-reasoner", api_key="...").
from os import getenv as _getenv


def _apply_defaults(kwargs: dict, **defaults) -> dict:
    """Set provider defaults without overriding anything the caller passed.

    Uses ``setdefault`` per key so user-supplied ``id`` / ``name`` /
    ``provider`` / ``base_url`` / ``api_key`` / ``context_window`` always
    win. Also normalizes the legacy ``model=`` alias to ``id=``.
    """
    if "model" in kwargs:
        kwargs["id"] = kwargs.pop("model")
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    return kwargs


def DeepSeekChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id=_getenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash"),
            name="DeepSeek",
            provider="DeepSeek",
            base_url="https://api.deepseek.com",
            api_key=_getenv("DEEPSEEK_API_KEY"),
            context_window=1_000_000,
        )
    )


DeepSeek = DeepSeekChat


def MoonshotChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="kimi-k2.5",
            name="MoonShot",
            provider="MoonShot",
            base_url="https://api.moonshot.cn/v1",
            api_key=_getenv("MOONSHOT_API_KEY"),
        )
    )


Moonshot = MoonshotChat


def ArkChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id=_getenv("ARK_MODEL_NAME", "doubao-1.5-pro-32k"),
            name="Ark",
            provider="ByteDance Volcengine Ark",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=_getenv("ARK_API_KEY"),
        )
    )


Ark = ArkChat


def TogetherChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            name="Together",
            provider="Together",
            base_url="https://api.together.xyz/v1",
            api_key=_getenv("TOGETHER_API_KEY"),
        )
    )


Together = TogetherChat


def GrokChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="grok-beta",
            name="Grok",
            provider="xAI",
            base_url="https://api.x.ai/v1",
            api_key=_getenv("XAI_API_KEY"),
        )
    )


Grok = GrokChat


def YiChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="yi-lightning",
            name="Yi",
            provider="01.ai",
            base_url="https://api.lingyiwanwu.com/v1",
            api_key=_getenv("YI_API_KEY"),
        )
    )


Yi = YiChat


def QwenChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="qwen-max",
            name="Qwen",
            provider="Alibaba",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=_getenv("DASHSCOPE_API_KEY"),
        )
    )


Qwen = QwenChat


def ZhipuAIChat(**kwargs):
    # ZAI_API_KEY is the canonical name; ZHIPUAI_API_KEY kept as legacy fallback.
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="glm-4.7-flash",
            name="ZhipuAI",
            provider="ZhipuAI",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=_getenv("ZAI_API_KEY") or _getenv("ZHIPUAI_API_KEY"),
        )
    )


ZhipuAI = ZhipuAIChat


def NvidiaChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id=_getenv("NVIDIA_MODEL_NAME", "deepseek-ai/deepseek-v4-flash"),
            name="Nvidia",
            provider="Nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=_getenv("NVIDIA_API_KEY"),
        )
    )


def SambanovaChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="Meta-Llama-3.1-8B-Instruct",
            name="Sambanova",
            provider="Sambanova",
            base_url="https://api.sambanova.ai/v1",
            api_key=_getenv("SAMBANOVA_API_KEY"),
        )
    )


def OpenRouterChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="gpt-4o",
            name="OpenRouter",
            provider="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=_getenv("OPENROUTER_API_KEY"),
        )
    )


def FireworksChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="accounts/fireworks/models/firefunction-v2",
            name="Fireworks",
            provider="Fireworks",
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=_getenv("FIREWORKS_API_KEY"),
        )
    )


def InternLMChat(**kwargs):
    return OpenAIChat(
        **_apply_defaults(
            kwargs,
            id="internlm2.5-latest",
            name="InternLM",
            provider="InternLM",
            base_url="https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions",
            api_key=_getenv("INTERNLM_API_KEY"),
        )
    )


# Slug-keyed dispatch (used by gateway/services/model_factory.py and similar)
PROVIDER_FACTORIES = {
    "deepseek": DeepSeekChat,
    "moonshot": MoonshotChat,
    "ark": ArkChat,
    "together": TogetherChat,
    "xai": GrokChat,
    "yi": YiChat,
    "qwen": QwenChat,
    "zhipuai": ZhipuAIChat,
    "nvidia": NvidiaChat,
    "sambanova": SambanovaChat,
    "openrouter": OpenRouterChat,
    "fireworks": FireworksChat,
    "internlm": InternLMChat,
}

# ── Pluggable provider registry ──
from agentica.provider_registry import (
    register_provider,
    unregister_provider,
    get_provider_factory,
    list_providers,
    create_provider,
)

# ── Auxiliary task model router ──
from agentica.auxiliary_router import AuxiliaryModelRouter

# ── Think scrubber ──
from agentica.think_scrubber import (
    scrub_reasoning,
    contains_reasoning_leak,
    sanitize_assistant_content_for_history,
)

# ── Doctor / environment health check ──
from agentica.diagnostics import run_doctor, DoctorReport, DoctorCheck

# ── Memory ──
from agentica.memory import (
    AgentRun,
    SessionSummary,
    MemorySummarizer,
    WorkingMemory,
    MemoryType,
    MemoryEntry,
    WorkflowRun,
    WorkflowMemory,
)

# ── Database (base types only) ──
from agentica.db.base import BaseDb, SessionRow, MemoryRow, MetricsRow

# ── Run Response ──
from agentica.run_response import (
    RunResponse,
    RunEvent,
    RunBreakReason,
    RunResponseExtraData,
    ToolCallInfo,
    pprint_run_response,
)
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
    BuiltinFileTool,
    BuiltinExecuteTool,
    BuiltinFetchUrlTool,
    BuiltinWebSearchTool,
    BuiltinTodoTool,
    BuiltinMemoryTool,
)
from agentica.tools.builtin_task_tool import BuiltinTaskTool

# ── Compression ──
from agentica.compression import CompressionManager

# ── Checkpoint / rollback primitive ──
from agentica.checkpoint import CheckpointManager, Checkpoint, CheckpointFile

# ── Token counting ──
from agentica.utils.tokens import (
    count_tokens,
    count_text_tokens,
    count_image_tokens,
    count_message_tokens,
    count_tool_tokens,
)

# ── Agent (core) ──
from agentica.agent import Agent, AgentCancelledError
from agentica.agent.deep import DeepAgent
from agentica.agent.config import (
    PromptConfig,
    ToolConfig,
    WorkspaceMemoryConfig,
    HistoryConfig,
    SandboxConfig,
    ToolRuntimeConfig,
    SkillRuntimeConfig,
    ExperienceConfig,
    SkillUpgradeConfig,
)
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
    eager_names = [name for name in globals() if not name.startswith("_")]
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
        GuardrailOutput,
        InputGuardrail,
        OutputGuardrail,
        InputGuardrailResult,
        OutputGuardrailResult,
        input_guardrail,
        output_guardrail,
        InputGuardrailTripwireTriggered,
        OutputGuardrailTripwireTriggered,
        ToolGuardrailFunctionOutput,
        ToolInputGuardrail,
        ToolOutputGuardrail,
        ToolInputGuardrailData,
        ToolOutputGuardrailData,
        ToolContext,
        tool_input_guardrail,
        tool_output_guardrail,
        ToolInputGuardrailTripwireTriggered,
        ToolOutputGuardrailTripwireTriggered,
        run_input_guardrails,
        run_output_guardrails,
        run_tool_input_guardrails,
        run_tool_output_guardrails,
    )
    from agentica.tools.search_serper_tool import SearchSerperTool  # noqa: F401
    from agentica.tools.dalle_tool import DalleTool  # noqa: F401
    from agentica.tools.shell_tool import ShellTool  # noqa: F401
    from agentica.tools.code_tool import CodeTool  # noqa: F401
    from agentica.tools.mcp_tool import McpTool, CompositeMultiMcpTool  # noqa: F401
    from agentica.mcp.config import MCPConfig  # noqa: F401


__all__ = _api_registry.PUBLIC_API_ALL
