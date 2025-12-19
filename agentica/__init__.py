# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from agentica.version import __version__  # noqa, isort: skip
from agentica.config import (
    AGENTICA_HOME,
    AGENTICA_DOTENV_PATH,
    AGENTICA_LOG_LEVEL,
    AGENTICA_LOG_FILE,
)  # noqa, isort: skip
from agentica.utils.log import set_log_level_to_debug, logger, set_log_level_to_info
from agentica.utils.io import write_audio_to_file
# model
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

# memory
from agentica.model.base import Model
from agentica.model.message import Message, MessageReferences
from agentica.model.content import Media, Video, Audio, Image
from agentica.model.response import ModelResponse, FileType
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
)

# database
from agentica.db import (
    BaseDb,
    SessionRow,
    MemoryRow,
    MetricsRow,
    SqliteDb,
    PostgresDb,
    InMemoryDb,
    JsonDb,
)

from agentica.template import PromptTemplate
# rag
from agentica.run_response import (
    RunResponse,
    RunEvent,
    RunResponseExtraData,
    pprint_run_response,
)
# knowledge
from agentica.knowledge.base import Knowledge
from agentica.knowledge.llamaindex_knowledge import LlamaIndexKnowledge
from agentica.knowledge.langchain_knowledge import LangChainKnowledge
# document
from agentica.document import Document
# vectordb
from agentica.vectordb.base import SearchType, Distance, VectorDb
from agentica.vectordb.memory_vectordb import InMemoryVectorDb
# emb
from agentica.emb.base import Emb
from agentica.emb.openai_emb import OpenAIEmb
from agentica.emb.azure_openai_emb import AzureOpenAIEmb
from agentica.emb.hash_emb import HashEmb
from agentica.emb.ollama_emb import OllamaEmb
from agentica.emb.together_emb import TogetherEmb
from agentica.emb.fireworks_emb import FireworksEmb
from agentica.emb.zhipuai_emb import ZhipuAIEmb

# file
from agentica.file.base import File
from agentica.file.csv import CsvFile
from agentica.file.txt import TextFile

# tool
from agentica.tools.base import Tool, ModelTool, Function, FunctionCall

# skills
from agentica.skills import (
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

from agentica.tools.guardrails import (
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
from agentica.tools.search_serper_tool import SearchSerperTool
from agentica.tools.baidu_search_tool import BaiduSearchTool
from agentica.tools.run_python_code_tool import RunPythonCodeTool
from agentica.tools.image_analysis_tool import ImageAnalysisTool
from agentica.tools.calculator_tool import CalculatorTool
from agentica.tools.dalle_tool import DalleTool
from agentica.tools.file_tool import FileTool
from agentica.tools.hackernews_tool import HackerNewsTool
from agentica.tools.jina_tool import JinaTool
from agentica.tools.shell_tool import ShellTool
from agentica.tools.skill_tool import SkillTool
from agentica.tools.text_analysis_tool import TextAnalysisTool
from agentica.tools.weather_tool import WeatherTool

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

# agent
from agentica.agent import Agent
from agentica.deep_agent import DeepAgent
from agentica.agent_session import AgentSession
from agentica.workflow import Workflow
from agentica.workflow_session import WorkflowSession

# built-in tools for DeepAgent
from agentica.deep_tools import (
    BuiltinFileTool,
    BuiltinExecuteTool,
    BuiltinWebSearchTool,
    BuiltinFetchUrlTool,
    BuiltinTodoTool,
    BuiltinTaskTool,
    get_builtin_tools,
)

# guardrails
from agentica.guardrails import (
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
from agentica.mcp.config import MCPConfig
