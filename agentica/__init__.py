# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Lazy loading is used for tools and optional modules to improve import speed.
Core modules (Agent, Model, Memory, etc.) are imported eagerly.
"""
import importlib
import threading
from typing import TYPE_CHECKING

from agentica.version import __version__  # noqa, isort: skip
from agentica.config import (
    AGENTICA_HOME,
    AGENTICA_DOTENV_PATH,
    AGENTICA_LOG_LEVEL,
    AGENTICA_LOG_FILE,
    AGENTICA_WORKSPACE_DIR,
)  # noqa, isort: skip
from agentica.utils.log import set_log_level_to_debug, logger, set_log_level_to_info
from agentica.utils.io import write_audio_to_file

# model - core providers (fast import)
from agentica.model.openai.chat import OpenAIChat
from agentica.model.openai.like import OpenAILike
from agentica.model.azure.openai_chat import AzureOpenAIChat
from agentica.model.moonshot.chat import MoonshotChat
from agentica.model.moonshot.chat import MoonshotChat as Moonshot
from agentica.model.deepseek.chat import DeepSeekChat
from agentica.model.deepseek.chat import DeepSeekChat as DeepSeek
from agentica.model.doubao.chat import DoubaoChat
from agentica.model.doubao.chat import DoubaoChat as Doubao
from agentica.model.together.chat import TogetherChat
from agentica.model.together.chat import TogetherChat as Together
from agentica.model.xai.chat import GrokChat
from agentica.model.xai.chat import GrokChat as Grok
from agentica.model.yi.chat import YiChat
from agentica.model.yi.chat import YiChat as Yi
from agentica.model.qwen.chat import QwenChat
from agentica.model.qwen.chat import QwenChat as Qwen
from agentica.model.zhipuai.chat import ZhipuAIChat
from agentica.model.zhipuai.chat import ZhipuAIChat as ZhipuAI

# model base
from agentica.model.base import Model
from agentica.model.message import Message, MessageReferences, UserMessage, AssistantMessage, SystemMessage, ToolMessage
from agentica.model.content import Media, Video, Audio, Image
from agentica.model.response import ModelResponse, FileType

# memory
from agentica.memory import (
    AgentRun,
    SessionSummary,
    MemorySummarizer,
    WorkingMemory,
    WorkflowRun,
    WorkflowMemory,
    MemoryChunk,
    WorkspaceMemorySearch,
)

# database - base types only (fast import)
from agentica.db.base import BaseDb, SessionRow, MemoryRow, MetricsRow

# run response
from agentica.run_response import (
    RunResponse,
    RunEvent,
    RunResponseExtraData,
    ToolCallInfo,
    pprint_run_response,
)

# document
from agentica.document import Document

# tool base (core, fast import)
from agentica.tools.base import Tool, ModelTool, Function, FunctionCall

# compression
from agentica.compression import CompressionManager

# token counting
from agentica.utils.tokens import (
    count_tokens,
    count_text_tokens,
    count_image_tokens,
    count_message_tokens,
    count_tool_tokens,
)

# agent (core)
from agentica.agent import Agent, AgentCancelledError
from agentica.agent.config import PromptConfig, ToolConfig, WorkspaceMemoryConfig, TeamConfig
from agentica.run_config import RunConfig
from agentica.deep_agent import DeepAgent
from agentica.workflow import Workflow, WorkflowSession

# workspace
from agentica.workspace import Workspace, WorkspaceConfig

# ============================================================================
# Lazy imports - loaded on demand to improve startup time
# ============================================================================

# Mapping of lazy-loaded names to their module paths
_LAZY_IMPORTS = {
    # database (may have heavy dependencies like psycopg2, mysql-connector, redis)
    "SqliteDb": "agentica.db.sqlite",
    "PostgresDb": "agentica.db.postgres",
    "InMemoryDb": "agentica.db.memory",
    "JsonDb": "agentica.db.json",
    "MysqlDb": "agentica.db.mysql",
    "RedisDb": "agentica.db.redis",

    # litellm (heavy dependency)
    "LiteLLMChat": "agentica.model.litellm.chat",
    "LiteLLM": "agentica.model.litellm.chat",

    # knowledge (may have heavy dependencies like llama_index, langchain)
    "Knowledge": "agentica.knowledge.base",
    "LlamaIndexKnowledge": "agentica.knowledge.llamaindex_knowledge",
    "LangChainKnowledge": "agentica.knowledge.langchain_knowledge",

    # vectordb
    "SearchType": "agentica.vectordb.base",
    "Distance": "agentica.vectordb.base",
    "VectorDb": "agentica.vectordb.base",
    "InMemoryVectorDb": "agentica.vectordb.memory_vectordb",

    # embeddings
    "Embedding": "agentica.embedding.base",
    "OpenAIEmbedding": "agentica.embedding.openai",
    "AzureOpenAIEmbedding": "agentica.embedding.azure_openai",
    "HashEmbedding": "agentica.embedding.hash",
    "OllamaEmbedding": "agentica.embedding.ollama",
    "TogetherEmbedding": "agentica.embedding.together",
    "FireworksEmbedding": "agentica.embedding.fireworks",
    "ZhipuAIEmbedding": "agentica.embedding.zhipuai",
    "HttpEmbedding": "agentica.embedding.http",
    "JinaEmbedding": "agentica.embedding.jina",
    "GeminiEmbedding": "agentica.embedding.gemini",
    "HuggingfaceEmbedding": "agentica.embedding.huggingface",
    "MulanAIEmbedding": "agentica.embedding.mulanai",

    # rerank
    "Rerank": "agentica.rerank.base",
    "CohereRerank": "agentica.rerank.cohere",
    "JinaRerank": "agentica.rerank.jina",
    "ZhipuAIRerank": "agentica.rerank.zhipuai",

    # file
    "File": "agentica.file.base",
    "CsvFile": "agentica.file.csv",
    "TextFile": "agentica.file.txt",

    # skills
    "Skill": "agentica.skills",
    "SkillRegistry": "agentica.skills",
    "SkillLoader": "agentica.skills",
    "get_skill_registry": "agentica.skills",
    "reset_skill_registry": "agentica.skills",
    "load_skills": "agentica.skills",
    "get_available_skills": "agentica.skills",
    "register_skill": "agentica.skills",
    "register_skills": "agentica.skills",
    "list_skill_files": "agentica.skills",
    "read_skill_file": "agentica.skills",

    # tool guardrails (now in unified guardrails package)
    "ToolGuardrailFunctionOutput": "agentica.guardrails",
    "ToolInputGuardrail": "agentica.guardrails",
    "ToolOutputGuardrail": "agentica.guardrails",
    "ToolInputGuardrailData": "agentica.guardrails",
    "ToolOutputGuardrailData": "agentica.guardrails",
    "ToolContext": "agentica.guardrails",
    "tool_input_guardrail": "agentica.guardrails",
    "tool_output_guardrail": "agentica.guardrails",
    "ToolInputGuardrailTripwireTriggered": "agentica.guardrails",
    "ToolOutputGuardrailTripwireTriggered": "agentica.guardrails",
    "ToolGuardrailTripwireTriggered": "agentica.guardrails",
    "run_tool_input_guardrails": "agentica.guardrails",
    "run_tool_output_guardrails": "agentica.guardrails",

    # tools (each may have external dependencies)
    "SearchSerperTool": "agentica.tools.search_serper_tool",
    "BaiduSearchTool": "agentica.tools.baidu_search_tool",
    "ImageAnalysisTool": "agentica.tools.image_analysis_tool",
    "DalleTool": "agentica.tools.dalle_tool",
    "HackerNewsTool": "agentica.tools.hackernews_tool",
    "JinaTool": "agentica.tools.jina_tool",
    "ShellTool": "agentica.tools.shell_tool",
    "SkillTool": "agentica.tools.skill_tool",
    "WeatherTool": "agentica.tools.weather_tool",
    "CodeTool": "agentica.tools.code_tool",
    "PatchTool": "agentica.tools.patch_tool",

    # built-in tools for DeepAgent
    "BuiltinFileTool": "agentica.deep_tools",
    "BuiltinExecuteTool": "agentica.deep_tools",
    "BuiltinWebSearchTool": "agentica.deep_tools",
    "BuiltinFetchUrlTool": "agentica.deep_tools",
    "BuiltinTodoTool": "agentica.deep_tools",
    "BuiltinTaskTool": "agentica.deep_tools",
    "get_builtin_tools": "agentica.deep_tools",

    # subagent system
    "SubagentType": "agentica.subagent",
    "SubagentConfig": "agentica.subagent",
    "SubagentRun": "agentica.subagent",
    "SubagentRegistry": "agentica.subagent",
    "get_subagent_config": "agentica.subagent",
    "get_available_subagent_types": "agentica.subagent",
    "register_custom_subagent": "agentica.subagent",
    "unregister_custom_subagent": "agentica.subagent",
    "get_custom_subagent_configs": "agentica.subagent",
    
    # acp system
    "ACPServer": "agentica.acp",
    "ACPTool": "agentica.acp",
    "ACPToolCall": "agentica.acp",
    "ACPToolResult": "agentica.acp",
    "ACPRequest": "agentica.acp",
    "ACPResponse": "agentica.acp",
    "ACPErrorCode": "agentica.acp",
    "ACPMethod": "agentica.acp",
    "SessionManager": "agentica.acp",
    "ACPSession": "agentica.acp",
    "SessionStatus": "agentica.acp",

    # human-in-the-loop tool
    "UserInputTool": "agentica.tools.user_input_tool",
    "UserInputRequired": "agentica.tools.user_input_tool",

    # guardrails
    "GuardrailFunctionOutput": "agentica.guardrails",
    "InputGuardrail": "agentica.guardrails",
    "OutputGuardrail": "agentica.guardrails",
    "InputGuardrailResult": "agentica.guardrails",
    "OutputGuardrailResult": "agentica.guardrails",
    "input_guardrail": "agentica.guardrails",
    "output_guardrail": "agentica.guardrails",
    "InputGuardrailTripwireTriggered": "agentica.guardrails",
    "OutputGuardrailTripwireTriggered": "agentica.guardrails",

    # mcp
    "MCPConfig": "agentica.mcp.config",
    "McpTool": "agentica.tools.mcp_tool",
    "CompositeMultiMcpTool": "agentica.tools.mcp_tool",
}

# Cache for lazy-loaded modules
_LAZY_CACHE = {}
_LAZY_LOCK = threading.Lock()


def __getattr__(name: str):
    """Lazy import handler for optional modules."""
    if name in _LAZY_IMPORTS:
        if name not in _LAZY_CACHE:
            with _LAZY_LOCK:
                if name not in _LAZY_CACHE:  # double-checked locking
                    module_path = _LAZY_IMPORTS[name]
                    module = importlib.import_module(module_path)
                    _LAZY_CACHE[name] = getattr(module, name)
        return _LAZY_CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List all available names including lazy imports."""
    eager_names = [name for name in globals() if not name.startswith('_')]
    return sorted(set(eager_names) | set(_LAZY_IMPORTS.keys()))


# Type hints for IDE support (not executed at runtime)
if TYPE_CHECKING:  # noqa: F401
    # database
    from agentica.db.sqlite import SqliteDb  # noqa: F401
    from agentica.db.postgres import PostgresDb  # noqa: F401
    from agentica.db.memory import InMemoryDb  # noqa: F401
    from agentica.db.json import JsonDb  # noqa: F401
    from agentica.db.mysql import MysqlDb  # noqa: F401
    from agentica.db.redis import RedisDb  # noqa: F401

    # litellm
    from agentica.model.litellm.chat import LiteLLMChat  # noqa: F401
    from agentica.model.litellm.chat import LiteLLMChat as LiteLLM  # noqa: F401

    # knowledge
    from agentica.knowledge.base import Knowledge  # noqa: F401
    from agentica.knowledge.llamaindex_knowledge import LlamaIndexKnowledge  # noqa: F401
    from agentica.knowledge.langchain_knowledge import LangChainKnowledge  # noqa: F401

    # vectordb
    from agentica.vectordb.base import SearchType, Distance, VectorDb  # noqa: F401
    from agentica.vectordb.memory_vectordb import InMemoryVectorDb  # noqa: F401

    # embeddings
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

    # rerank
    from agentica.rerank.base import Rerank  # noqa: F401
    from agentica.rerank.cohere import CohereRerank  # noqa: F401
    from agentica.rerank.jina import JinaRerank  # noqa: F401
    from agentica.rerank.zhipuai import ZhipuAIRerank  # noqa: F401

    # file
    from agentica.file.base import File  # noqa: F401
    from agentica.file.csv import CsvFile  # noqa: F401
    from agentica.file.txt import TextFile  # noqa: F401

    # skills
    from agentica.skills import (  # noqa: F401
        Skill,
        SkillRegistry,
        SkillLoader,
        get_skill_registry,
        reset_skill_registry,
        load_skills,
        get_available_skills,
        register_skill,
        register_skills,
        list_skill_files,
        read_skill_file,
    )

    # guardrails (unified package)
    from agentica.guardrails import (  # noqa: F401
        # Agent-level guardrails
        GuardrailFunctionOutput,
        InputGuardrail,
        OutputGuardrail,
        InputGuardrailResult,
        OutputGuardrailResult,
        input_guardrail,
        output_guardrail,
        InputGuardrailTripwireTriggered,
        OutputGuardrailTripwireTriggered,
        GuardrailTripwireTriggered,
        run_input_guardrails,
        run_output_guardrails,
        # Tool-level guardrails
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
        run_tool_input_guardrails,
        run_tool_output_guardrails,
    )

    # tools
    from agentica.tools.search_serper_tool import SearchSerperTool  # noqa: F401
    from agentica.tools.baidu_search_tool import BaiduSearchTool  # noqa: F401
    from agentica.tools.image_analysis_tool import ImageAnalysisTool  # noqa: F401
    from agentica.tools.dalle_tool import DalleTool  # noqa: F401
    from agentica.tools.hackernews_tool import HackerNewsTool  # noqa: F401
    from agentica.tools.jina_tool import JinaTool  # noqa: F401
    from agentica.tools.shell_tool import ShellTool  # noqa: F401
    from agentica.tools.skill_tool import SkillTool  # noqa: F401
    from agentica.tools.weather_tool import WeatherTool  # noqa: F401
    from agentica.tools.code_tool import CodeTool  # noqa: F401
    from agentica.tools.patch_tool import PatchTool  # noqa: F401

    # built-in tools for DeepAgent
    from agentica.deep_tools import (  # noqa: F401
        BuiltinFileTool,
        BuiltinExecuteTool,
        BuiltinWebSearchTool,
        BuiltinFetchUrlTool,
        BuiltinTodoTool,
        BuiltinTaskTool,
        get_builtin_tools,
    )

    # subagent system
    from agentica.subagent import (  # noqa: F401
        SubagentType,
        SubagentConfig,
        SubagentRun,
        SubagentRegistry,
        get_subagent_config,
        get_available_subagent_types,
        register_custom_subagent,
        unregister_custom_subagent,
        get_custom_subagent_configs,
    )
    
    # acp system
    from agentica.acp import (  # noqa: F401
        ACPServer,
        ACPTool,
        ACPToolCall,
        ACPToolResult,
        ACPRequest,
        ACPResponse,
        ACPErrorCode,
        ACPMethod,
        SessionManager,
        ACPSession,
        SessionStatus,
    )

    # human-in-the-loop tool
    from agentica.tools.user_input_tool import UserInputTool, UserInputRequired  # noqa: F401

    # mcp
    from agentica.mcp.config import MCPConfig  # noqa: F401
    from agentica.tools.mcp_tool import McpTool, CompositeMultiMcpTool  # noqa: F401


# __all__ for explicit exports
__all__ = [
    # version
    "__version__",
    # config
    "AGENTICA_HOME",
    "AGENTICA_DOTENV_PATH",
    "AGENTICA_LOG_LEVEL",
    "AGENTICA_LOG_FILE",
    "AGENTICA_WORKSPACE_DIR",
    # logging
    "set_log_level_to_debug",
    "set_log_level_to_info",
    "logger",
    # utils
    "write_audio_to_file",
    # models
    "OpenAIChat",
    "OpenAILike",
    "AzureOpenAIChat",
    "MoonshotChat",
    "Moonshot",
    "DeepSeekChat",
    "DeepSeek",
    "DoubaoChat",
    "Doubao",
    "TogetherChat",
    "Together",
    "GrokChat",
    "Grok",
    "YiChat",
    "Yi",
    "QwenChat",
    "Qwen",
    "ZhipuAIChat",
    "ZhipuAI",
    "Model",
    "Message",
    "MessageReferences",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ToolMessage",
    "Media",
    "Video",
    "Audio",
    "Image",
    "ModelResponse",
    "FileType",
    # memory
    "AgentRun",
    "SessionSummary",
    "MemorySummarizer",
    "WorkingMemory",
    "WorkflowRun",
    "WorkflowMemory",
    "MemoryChunk",
    "WorkspaceMemorySearch",
    # database (base types eager, implementations lazy)
    "BaseDb",
    "SessionRow",
    "MemoryRow",
    "MetricsRow",
    # run response
    "RunResponse",
    "RunEvent",
    "RunResponseExtraData",
    "ToolCallInfo",
    "pprint_run_response",
    # document
    "Document",
    # tool base
    "Tool",
    "ModelTool",
    "Function",
    "FunctionCall",
    # compression
    "CompressionManager",
    # token counting
    "count_tokens",
    "count_text_tokens",
    "count_image_tokens",
    "count_message_tokens",
    "count_tool_tokens",
    # agent
    "Agent",
    "AgentCancelledError",
    "DeepAgent",
    "PromptConfig",
    "ToolConfig",
    "WorkspaceMemoryConfig",
    "TeamConfig",
    "RunConfig",
    "Workflow",
    "WorkflowSession",
    # workspace
    "Workspace",
    "WorkspaceConfig",
    # lazy imports (listed for __all__ but loaded on demand)
    *_LAZY_IMPORTS.keys(),
]
