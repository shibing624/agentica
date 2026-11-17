# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Global tool registry for managing named tools.

Usage:
    from agentica.tools.registry import register_tool, get_tool, list_tools

    register_tool("calculator", calculate_func)
    tool = get_tool("calculator")
    all_tools = list_tools()
"""
from typing import Dict, Union, Callable, List


_TOOL_REGISTRY: Dict[str, Union[Callable, "Tool"]] = {}

# Tool registry - maps tool names to (module_name, class_name, category, description)
# Module path: agentica.tools.{module_name}_tool.{ClassName}
TOOL_REGISTRY = {
    # AI/ML Tools
    "cogvideo": ("cogvideo", "CogVideoTool", "AI/ML", "Text-to-video generation with CogVideo"),
    "cogview": ("cogview", "CogViewTool", "AI/ML", "Text-to-image generation with CogView"),
    "dalle": ("dalle", "DalleTool", "AI/ML", "Image generation with DALL-E"),
    "video_analysis": ("video_analysis", "VideoAnalysisTool", "AI/ML", "Video content analysis"),
    "volc_tts": ("volc_tts", "VolcTtsTool", "AI/ML", "Text-to-speech with Volcengine"),
    # Search Tools
    "arxiv": ("arxiv", "ArxivTool", "Search", "Search academic papers on arXiv"),
    "baidu_search": ("baidu_search", "BaiduSearchTool", "Search", "Web search via Baidu"),
    "dblp": ("dblp", "DblpTool", "Search", "Search computer science papers on DBLP"),
    "duckduckgo": ("duckduckgo", "DuckDuckGoTool", "Search", "Web search via DuckDuckGo"),
    "search_bocha": ("search_bocha", "SearchBochaTool", "Search", "Web search via Bocha"),
    "search_exa": ("search_exa", "SearchExaTool", "Search", "Web search via Exa"),
    "search_serper": ("search_serper", "SearchSerperTool", "Search", "Web search via Serper (Google)"),
    "wikipedia": ("wikipedia", "WikipediaTool", "Search", "Search and read Wikipedia articles"),
    "zhipu_web_search": ("zhipu_web_search", "ZhipuWebSearchTool", "Search", "Web search via ZhipuAI"),
    # Web/Network Tools
    "jina": ("jina", "JinaTool", "Web", "Web content extraction via Jina Reader"),
    "newspaper": ("newspaper", "NewspaperTool", "Web", "Article extraction from news URLs"),
    "url_crawler": ("url_crawler", "UrlCrawlerTool", "Web", "Recursive URL crawling"),
    # File/Code Tools
    "calculator": ("calculator", "CalculatorTool", "Code & Files", "Mathematical expression evaluation"),
    "code": ("code", "CodeTool", "Code & Files", "Code generation and execution"),
    "edit": ("edit", "EditTool", "Code & Files", "File editing with diff patches"),
    "file": ("file", "FileTool", "Code & Files", "File system operations"),
    "run_nb_code": ("run_nb_code", "RunNbCodeTool", "Code & Files", "Execute Jupyter notebook code"),
    "run_python_code": ("run_python_code", "RunPythonCodeTool", "Code & Files", "Execute Python code snippets"),
    "string": ("string", "StringTool", "Code & Files", "String manipulation utilities"),
    "text_analysis": ("text_analysis", "TextAnalysisTool", "Code & Files", "Text analysis and NLP"),
    "workspace": ("workspace", "WorkspaceTool", "Code & Files", "Workspace file management"),
    # Data Tools
    "hackernews": ("hackernews", "HackerNewsTool", "Data", "Fetch Hacker News stories"),
    "sql": ("sql", "SqlTool", "Data", "SQL database queries"),
    "weather": ("weather", "WeatherTool", "Data", "Weather information"),
    "yfinance": ("yfinance", "YFinanceTool", "Data", "Financial data from Yahoo Finance"),
    # Integration Tools
    "airflow": ("airflow", "AirflowTool", "Integration", "Apache Airflow DAG management"),
    "apify": ("apify", "ApifyTool", "Integration", "Web scraping via Apify"),
    "mcp": ("mcp", "MCPTool", "Integration", "Model Context Protocol integration"),
    "memori": ("memori", "MemoriTool", "Integration", "Long-term memory management"),
    "skill": ("skill", "SkillTool", "Integration", "Skill document management"),
    "video_download": ("video_download", "VideoDownloadTool", "Integration", "Video download from URLs"),
}


def register_tool(name: str, tool_or_func: Union[Callable, "Tool"]) -> None:
    """Register a tool to the global registry.

    Args:
        name: Unique name for the tool.
        tool_or_func: A callable function or Tool instance.
    """
    _TOOL_REGISTRY[name] = tool_or_func


def get_tool(name: str) -> Union[Callable, "Tool"]:
    """Get a registered tool by name.

    Args:
        name: The tool name.

    Returns:
        The registered tool or function.

    Raises:
        KeyError: If the tool is not found.
    """
    if name not in _TOOL_REGISTRY:
        raise KeyError(
            f"Tool '{name}' not found. Available: {sorted(_TOOL_REGISTRY.keys())}"
        )
    return _TOOL_REGISTRY[name]


def list_tools() -> List[str]:
    """List all registered tool names.

    Returns:
        Sorted list of tool names.
    """
    return sorted(_TOOL_REGISTRY.keys())


def unregister_tool(name: str) -> None:
    """Remove a tool from the registry.

    Args:
        name: The tool name to remove.

    Raises:
        KeyError: If the tool is not found.
    """
    if name not in _TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not found.")
    del _TOOL_REGISTRY[name]


def clear_registry() -> None:
    """Remove all tools from the registry."""
    _TOOL_REGISTRY.clear()


# ── Re-exports from helpers for convenience ──
from agentica.tools.helpers import tool_error, tool_result  # noqa: F401, E402
