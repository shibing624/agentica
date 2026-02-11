# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: CLI configuration - constants, tool registry, argument parsing, model/agent creation
"""
import argparse
import importlib
import os
import sys
from typing import List, Optional, Any

from rich.console import Console

from agentica import DeepAgent, OpenAIChat, Moonshot, AzureOpenAIChat, Yi, ZhipuAI, DeepSeek, Doubao
from agentica.config import AGENTICA_HOME
from agentica.workspace import Workspace

console = Console()
history_file = os.path.join(AGENTICA_HOME, "cli_history.txt")

# Tool icons for CLI display
TOOL_ICONS = {
    # File tools
    "ls": "📁",
    "read_file": "📖",
    "write_file": "✏️",
    "edit_file": "✂️",
    "glob": "🔍",
    "grep": "🔎",
    # Execution
    "execute": "⚡",
    # Web tools
    "web_search": "🌐",
    "fetch_url": "🔗",
    # Task management
    "write_todos": "📋",
    "read_todos": "📋",
    # Subagent
    "task": "🤖",
    # Memory
    "save_memory": "💾",
    # Default
    "default": "🔧",
}

# Tool registry - maps tool names to (module_name, class_name)
# Format: 'tool_name': ('module_name', 'ClassName')
# Module path: agentica.tools.{module_name}_tool.{ClassName}
TOOL_REGISTRY = {
    # AI/ML Tools
    'cogvideo':        ('cogvideo',        'CogVideoTool'),
    'cogview':         ('cogview',         'CogViewTool'),
    'dalle':           ('dalle',           'DalleTool'),
    'image_analysis':  ('image_analysis',  'ImageAnalysisTool'),
    'ocr':             ('ocr',             'OcrTool'),
    'video_analysis':  ('video_analysis',  'VideoAnalysisTool'),
    'volc_tts':        ('volc_tts',        'VolcTtsTool'),

    # Search Tools
    'arxiv':           ('arxiv',           'ArxivTool'),
    'baidu_search':    ('baidu_search',    'BaiduSearchTool'),
    'dblp':            ('dblp',            'DblpTool'),
    'duckduckgo':      ('duckduckgo',      'DuckDuckGoTool'),
    'search_bocha':    ('search_bocha',    'SearchBochaTool'),
    'search_exa':      ('search_exa',      'SearchExaTool'),
    'search_serper':   ('search_serper',   'SearchSerperTool'),
    'web_search_pro':  ('web_search_pro',  'WebSearchProTool'),
    'wikipedia':       ('wikipedia',       'WikipediaTool'),

    # Web/Network Tools
    'browser':         ('browser',         'BrowserTool'),
    'jina':            ('jina',            'JinaTool'),
    'newspaper':       ('newspaper',       'NewspaperTool'),
    'url_crawler':     ('url_crawler',     'UrlCrawlerTool'),

    # File/Code Tools
    'calculator':      ('calculator',      'CalculatorTool'),
    'code':            ('code',            'CodeTool'),
    'edit':            ('edit',            'EditTool'),
    'file':            ('file',            'FileTool'),
    'run_nb_code':     ('run_nb_code',     'RunNbCodeTool'),
    'run_python_code': ('run_python_code', 'RunPythonCodeTool'),
    'shell':           ('shell',           'ShellTool'),
    'string':          ('string',          'StringTool'),
    'text_analysis':   ('text_analysis',   'TextAnalysisTool'),
    'workspace':       ('workspace',       'WorkspaceTool'),

    # Data Tools
    'hackernews':      ('hackernews',      'HackerNewsTool'),
    'sql':             ('sql',             'SqlTool'),
    'weather':         ('weather',         'WeatherTool'),
    'yfinance':        ('yfinance',        'YFinanceTool'),

    # Integration Tools
    'airflow':         ('airflow',         'AirflowTool'),
    'apify':           ('apify',           'ApifyTool'),
    'mcp':             ('mcp',             'MCPTool'),
    'memori':          ('memori',          'MemoriTool'),
    'skill':           ('skill',           'SkillTool'),
    'video_download':  ('video_download',  'VideoDownloadTool'),
}

# Model provider registry - maps provider name to model class
MODEL_REGISTRY = {
    'openai': OpenAIChat,
    'azure': AzureOpenAIChat,
    'moonshot': Moonshot,
    'zhipuai': ZhipuAI,
    'deepseek': DeepSeek,
    'yi': Yi,
    'doubao': Doubao,
}

# Example models for each provider (for /model command display)
EXAMPLE_MODELS = {
    'openai': ['gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5.2', 'o3-mini'],
    'azure': ['gpt-4o', 'gpt-4o-mini'],
    'moonshot': ['kimi-k2.5', 'moonshot-v1-128k'],
    'zhipuai': ['glm-4', 'glm-4-flash', 'glm-4.7-flash'],
    'deepseek': ['deepseek-chat', 'deepseek-reasoner'],
    'yi': ['yi-lightning', 'yi-large'],
    'doubao': ['doubao-1.5-pro-32k', 'doubao-1.5-lite-32k', 'doubao-1.5-vision-pro-32k'],
}


def _get_tool_import_path(tool_name: str) -> str:
    """Get full import path for a tool."""
    module_name, class_name = TOOL_REGISTRY[tool_name]
    return f"agentica.tools.{module_name}_tool.{class_name}"


def parse_args():
    parser = argparse.ArgumentParser(description='CLI for agentica')
    
    # Check if running in ACP mode (special handling)
    if len(sys.argv) > 1 and sys.argv[1] == 'acp':
        return None  # Signal to run in ACP mode
    
    parser.add_argument('--query', type=str, help='Question to ask the LLM', default=None)
    parser.add_argument('--model_provider', type=str,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='LLM model provider', default='zhipuai')
    parser.add_argument('--model_name', type=str,
                        help='LLM model name to use, can be gpt-5/glm-4.7-flash/deepseek-chat/yi-lightning/...',
                        default='glm-4.7-flash')
    parser.add_argument('--base_url', type=str, help='API base URL for the LLM')
    parser.add_argument('--api_key', type=str, help='API key for the LLM')
    parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens for the LLM')
    parser.add_argument('--temperature', type=float, help='Temperature for the LLM')
    parser.add_argument('--verbose', type=int, help='enable verbose mode', default=0)
    parser.add_argument('--work_dir', type=str, help='Working directory for file operations', default=None)
    parser.add_argument('--tools', nargs='*',
                        choices=list(TOOL_REGISTRY.keys()),
                        help='Additional tools to enable (on top of DeepAgent built-in tools)')
    parser.add_argument('--workspace', type=str, default=None,
                        help='Workspace directory path (default: ~/.agentica/workspace)')
    parser.add_argument('--no-workspace', action='store_true',
                        help='Disable workspace context injection')
    parser.add_argument('--no-skills', action='store_true',
                        help='Disable skills loading')
    parser.add_argument('command', nargs='?', choices=['acp'],
                        help='Run in ACP mode for IDE integration (agentica acp)')
    return parser.parse_args()


def configure_tools(tool_names: Optional[List[str]] = None) -> List[Any]:
    """Configure and instantiate tools based on their names."""
    if not tool_names:
        return []

    tools = []
    for name in tool_names:
        if name not in TOOL_REGISTRY:
            console.print(f"[yellow]Warning: Tool '{name}' not recognized. Skipping.[/yellow]")
            continue

        try:
            import_path = _get_tool_import_path(name)
            module_path, class_name = import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            tool_instance = tool_class()
            tools.append(tool_instance)
            console.print(f"[green]Loaded additional tool: {name}[/green]")
        except ImportError as e:
            console.print(f"[red]Error: Could not import tool '{name}'. Missing dependencies? {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]Error: Failed to initialize tool '{name}': {str(e)}[/red]")

    return tools


def get_model(model_provider, model_name, base_url=None, api_key=None, max_tokens=None, temperature=None):
    """Create a model instance based on the provider name.

    Uses MODEL_REGISTRY for provider lookup instead of if/elif chains.
    """
    params = {"id": model_name}
    if base_url is not None:
        params["base_url"] = base_url
    if api_key is not None:
        params["api_key"] = api_key
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature

    model_class = MODEL_REGISTRY.get(model_provider)
    if model_class is None:
        raise ValueError(f"Unsupported model provider: {model_provider}. Supported: {', '.join(MODEL_REGISTRY.keys())}")
    return model_class(**params)


def create_agent(agent_config: dict, extra_tools: Optional[List] = None,
                 workspace: Optional[Workspace] = None, skills_registry=None):
    """Helper to create or recreate a DeepAgent with current config."""
    model = get_model(
        model_provider=agent_config["model_provider"],
        model_name=agent_config["model_name"],
        base_url=agent_config.get("base_url"),
        api_key=agent_config.get("api_key"),
        max_tokens=agent_config.get("max_tokens"),
        temperature=agent_config.get("temperature"),
    )

    # Build instructions with skills
    instructions = []

    # Add skills summary if available
    if skills_registry and len(skills_registry) > 0:
        skills_summary = skills_registry.get_skills_summary()
        if skills_summary:
            instructions.append(f"\n# Available Skills\n{skills_summary}")

    # Build kwargs for DeepAgent
    deep_agent_kwargs = {
        "model": model,
        "work_dir": agent_config.get("work_dir"),
        "tools": extra_tools,
        "add_datetime_to_instructions": True,
        "add_history_to_messages": True,
        "debug_mode": agent_config["debug_mode"],
        "workspace": workspace,
    }

    # Add instructions if we have any
    if instructions:
        deep_agent_kwargs["instructions"] = instructions


    new_agent = DeepAgent(**deep_agent_kwargs)
    return new_agent
