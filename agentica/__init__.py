# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Lazy loading is used for tools and optional modules to improve import speed.
Core modules (Agent, Model, Memory, etc.) are imported eagerly.
"""
import importlib
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
from agentica.model.openai.chat import OpenAIChat as OpenAILLM
from agentica.model.openai.like import OpenAILike
from agentica.model.azure.openai_chat import AzureOpenAIChat
from agentica.model.azure.openai_chat import AzureOpenAIChat as AzureOpenAILLM
from agentica.model.moonshot import Moonshot
from agentica.model.moonshot import Moonshot as MoonshotChat
from agentica.model.moonshot import Moonshot as MoonshotLLM
from agentica.model.deepseek.chat import DeepSeek
from agentica.model.deepseek.chat import DeepSeek as DeepSeekChat
from agentica.model.deepseek.chat import DeepSeek as DeepSeekLLM
from agentica.model.doubao.chat import Doubao
from agentica.model.doubao.chat import Doubao as DoubaoChat
from agentica.model.together.together import Together
from agentica.model.together.together import Together as TogetherChat
from agentica.model.together.together import Together as TogetherLLM
from agentica.model.xai.grok import Grok
from agentica.model.xai.grok import Grok as GrokChat
from agentica.model.yi.chat import Yi
from agentica.model.yi.chat import Yi as YiChat
from agentica.model.yi.chat import Yi as YiLLM
from agentica.model.qwen.chat import Qwen
from agentica.model.zhipuai.chat import ZhipuAI
from agentica.model.zhipuai.chat import ZhipuAI as ZhipuAIChat
from agentica.model.zhipuai.chat import ZhipuAI as ZhipuAILLM

# model base
from agentica.model.base import Model
from agentica.model.message import Message, MessageReferences, UserMessage, AssistantMessage, SystemMessage, ToolMessage
from agentica.model.content import Media, Video, Audio, Image
from agentica.model.response import ModelResponse, FileType

# memory
from agentica.memory import (
    AgentRun,
    SessionSummary,
    Memory,
    MemorySearchResponse,
    MemoryManager,
    MemoryClassifier,
    MemoryRetrieval,
    MemorySummarizer,
    AgentMemory,
    WorkflowRun,
    WorkflowMemory,
    MemoryChunk,
    WorkspaceMemorySearch,
)

# database - base types only (fast import)
from agentica.db.base import BaseDb, SessionRow, MemoryRow, MetricsRow

from agentica.template import PromptTemplate

# run response
from agentica.run_response import (
    RunResponse,
    RunEvent,
    RunResponseExtraData,
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
from agentica.agent import Agent
from agentica.deep_agent import DeepAgent
from agentica.agent_session import AgentSession
from agentica.workflow import Workflow
from agentica.workflow_session import WorkflowSession

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
    "Emb": "agentica.emb.base",
    "OpenAIEmb": "agentica.emb.openai_emb",
    "AzureOpenAIEmb": "agentica.emb.azure_openai_emb",
    "HashEmb": "agentica.emb.hash_emb",
    "OllamaEmb": "agentica.emb.ollama_emb",
    "TogetherEmb": "agentica.emb.together_emb",
    "FireworksEmb": "agentica.emb.fireworks_emb",
    "ZhipuAIEmb": "agentica.emb.zhipuai_emb",

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

    # tool guardrails
    "ToolGuardrailFunctionOutput": "agentica.tools.guardrails",
    "ToolInputGuardrail": "agentica.tools.guardrails",
    "ToolOutputGuardrail": "agentica.tools.guardrails",
    "ToolInputGuardrailData": "agentica.tools.guardrails",
    "ToolOutputGuardrailData": "agentica.tools.guardrails",
    "ToolContext": "agentica.tools.guardrails",
    "tool_input_guardrail": "agentica.tools.guardrails",
    "tool_output_guardrail": "agentica.tools.guardrails",
    "ToolInputGuardrailTripwireTriggered": "agentica.tools.guardrails",
    "ToolOutputGuardrailTripwireTriggered": "agentica.tools.guardrails",

    # tools (each may have external dependencies)
    "SearchSerperTool": "agentica.tools.search_serper_tool",
    "BaiduSearchTool": "agentica.tools.baidu_search_tool",
    "RunPythonCodeTool": "agentica.tools.run_python_code_tool",
    "ImageAnalysisTool": "agentica.tools.image_analysis_tool",
    "CalculatorTool": "agentica.tools.calculator_tool",
    "DalleTool": "agentica.tools.dalle_tool",
    "FileTool": "agentica.tools.file_tool",
    "HackerNewsTool": "agentica.tools.hackernews_tool",
    "JinaTool": "agentica.tools.jina_tool",
    "ShellTool": "agentica.tools.shell_tool",
    "SkillTool": "agentica.tools.skill_tool",
    "TextAnalysisTool": "agentica.tools.text_analysis_tool",
    "WeatherTool": "agentica.tools.weather_tool",

    # built-in tools for DeepAgent
    "BuiltinFileTool": "agentica.deep_tools",
    "BuiltinExecuteTool": "agentica.deep_tools",
    "BuiltinWebSearchTool": "agentica.deep_tools",
    "BuiltinFetchUrlTool": "agentica.deep_tools",
    "BuiltinTodoTool": "agentica.deep_tools",
    "BuiltinTaskTool": "agentica.deep_tools",
    "get_builtin_tools": "agentica.deep_tools",

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


def __getattr__(name: str):
    """Lazy import handler for optional modules."""
    if name in _LAZY_IMPORTS:
        if name not in _LAZY_CACHE:
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
    from agentica.model.litellm.chat import LiteLLM  # noqa: F401

    # knowledge
    from agentica.knowledge.base import Knowledge  # noqa: F401
    from agentica.knowledge.llamaindex_knowledge import LlamaIndexKnowledge  # noqa: F401
    from agentica.knowledge.langchain_knowledge import LangChainKnowledge  # noqa: F401

    # vectordb
    from agentica.vectordb.base import SearchType, Distance, VectorDb  # noqa: F401
    from agentica.vectordb.memory_vectordb import InMemoryVectorDb  # noqa: F401

    # embeddings
    from agentica.emb.base import Emb  # noqa: F401
    from agentica.emb.openai_emb import OpenAIEmb  # noqa: F401
    from agentica.emb.azure_openai_emb import AzureOpenAIEmb  # noqa: F401
    from agentica.emb.hash_emb import HashEmb  # noqa: F401
    from agentica.emb.ollama_emb import OllamaEmb  # noqa: F401
    from agentica.emb.together_emb import TogetherEmb  # noqa: F401
    from agentica.emb.fireworks_emb import FireworksEmb  # noqa: F401
    from agentica.emb.zhipuai_emb import ZhipuAIEmb  # noqa: F401

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

    # tool guardrails
    from agentica.tools.guardrails import (  # noqa: F401
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
    )

    # tools
    from agentica.tools.search_serper_tool import SearchSerperTool  # noqa: F401
    from agentica.tools.baidu_search_tool import BaiduSearchTool  # noqa: F401
    from agentica.tools.run_python_code_tool import RunPythonCodeTool  # noqa: F401
    from agentica.tools.image_analysis_tool import ImageAnalysisTool  # noqa: F401
    from agentica.tools.calculator_tool import CalculatorTool  # noqa: F401
    from agentica.tools.dalle_tool import DalleTool  # noqa: F401
    from agentica.tools.file_tool import FileTool  # noqa: F401
    from agentica.tools.hackernews_tool import HackerNewsTool  # noqa: F401
    from agentica.tools.jina_tool import JinaTool  # noqa: F401
    from agentica.tools.shell_tool import ShellTool  # noqa: F401
    from agentica.tools.skill_tool import SkillTool  # noqa: F401
    from agentica.tools.text_analysis_tool import TextAnalysisTool  # noqa: F401
    from agentica.tools.weather_tool import WeatherTool  # noqa: F401

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

    # human-in-the-loop tool
    from agentica.tools.user_input_tool import UserInputTool, UserInputRequired  # noqa: F401

    # guardrails
    from agentica.guardrails import (  # noqa: F401
        GuardrailFunctionOutput,
        InputGuardrail,
        OutputGuardrail,
        InputGuardrailResult,
        OutputGuardrailResult,
        input_guardrail,
        output_guardrail,
        InputGuardrailTripwireTriggered,
        OutputGuardrailTripwireTriggered,
    )

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
    "OpenAILLM",
    "OpenAILike",
    "AzureOpenAIChat",
    "AzureOpenAILLM",
    "Moonshot",
    "MoonshotChat",
    "MoonshotLLM",
    "DeepSeek",
    "DeepSeekChat",
    "DeepSeekLLM",
    "Doubao",
    "DoubaoChat",
    "Together",
    "TogetherChat",
    "TogetherLLM",
    "Grok",
    "GrokChat",
    "Yi",
    "YiChat",
    "YiLLM",
    "Qwen",
    "ZhipuAI",
    "ZhipuAIChat",
    "ZhipuAILLM",
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
    "Memory",
    "MemorySearchResponse",
    "MemoryManager",
    "MemoryClassifier",
    "MemoryRetrieval",
    "MemorySummarizer",
    "AgentMemory",
    "WorkflowRun",
    "WorkflowMemory",
    "MemoryChunk",
    "WorkspaceMemorySearch",
    # database (base types eager, implementations lazy)
    "BaseDb",
    "SessionRow",
    "MemoryRow",
    "MetricsRow",
    # template
    "PromptTemplate",
    # run response
    "RunResponse",
    "RunEvent",
    "RunResponseExtraData",
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
    "DeepAgent",
    "AgentSession",
    "Workflow",
    "WorkflowSession",
    # workspace
    "Workspace",
    "WorkspaceConfig",
    # lazy imports (listed for __all__ but loaded on demand)
    *_LAZY_IMPORTS.keys(),
]
