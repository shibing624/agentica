# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: command line interface for agentica
"""
import argparse
from rich.console import Console
from rich.text import Text
import importlib
from typing import List, Optional
from agentica import Agent, OpenAIChat, Moonshot, AzureOpenAIChat, Yi, ZhipuAI, DeepSeek, PythonAgent

console = Console()

# Tool mapping dictionary - maps tool names to their import paths
TOOL_MAP = {
    'airflow': 'agentica.tools.airflow_tool.AirflowTool',
    'analyze_image': 'agentica.tools.analyze_image_tool.AnalyzeImageTool',
    'arxiv': 'agentica.tools.arxiv_tool.ArxivTool',
    'baidu_search': 'agentica.tools.baidu_search_tool.BaiduSearchTool',
    'calculator': 'agentica.tools.calculator_tool.CalculatorTool',
    'code': 'agentica.tools.code_tool.CodeTool',
    'cogview': 'agentica.tools.cogview_tool.CogViewTool',
    'cogvideo': 'agentica.tools.cogvideo_tool.CogVideoTool',
    'dalle': 'agentica.tools.dalle_tool.DalleTool',
    'duckduckgo': 'agentica.tools.duckduckgo_tool.DuckDuckGoTool',
    'edit': 'agentica.tools.edit_tool.EditTool',
    'file': 'agentica.tools.file_tool.FileTool',
    'hackernews': 'agentica.tools.hackernews_tool.HackerNewsTool',
    'jina': 'agentica.tools.jina_tool.JinaTool',
    'mcp': 'agentica.tools.mcp_tool.MCPTool',
    'newspaper': 'agentica.tools.newspaper_tool.NewspaperTool',
    'ocr': 'agentica.tools.ocr_tool.OcrTool',
    'run_python_code': 'agentica.tools.run_python_code_tool.RunPythonCodeTool',
    'search_exa': 'agentica.tools.search_exa_tool.SearchExaTool',
    'search_serper': 'agentica.tools.search_serper_tool.SearchSerperTool',
    'shell': 'agentica.tools.shell_tool.ShellTool',
    'string': 'agentica.tools.string_tool.StringTool',
    'text_analysis': 'agentica.tools.text_analysis_tool.TextAnalysisTool',
    'url_crawler': 'agentica.tools.url_crawler_tool.UrlCrawlerTool',
    'volc_tts': 'agentica.tools.volc_tts_tool.VolcTtsTool',
    'weather': 'agentica.tools.weather_tool.WeatherTool',
    'web_search_pro': 'agentica.tools.web_search_pro_tool.WebSearchProTool',
    'wikipedia': 'agentica.tools.wikipedia_tool.WikipediaTool',
    'workspace': 'agentica.tools.workspace_tool.WorkspaceTool',
    'yfinance': 'agentica.tools.yfinance_tool.YFinanceTool',
}


def parse_args():
    parser = argparse.ArgumentParser(description='CLI for agentica')
    parser.add_argument('--query', type=str, help='Question to ask the LLM', default=None)
    parser.add_argument('--model_provider', type=str,
                        choices=['openai', 'azure', 'moonshot', 'zhipuai', 'deepseek', 'yi'],
                        help='LLM model provider', default='openai')
    parser.add_argument('--model_name', type=str,
                        help='LLM model name to use, can be gpt-4o/glm-4-flash/deepseek-chat/yi-lightning/...',
                        default='gpt-4o-mini')
    parser.add_argument('--api_base', type=str, help='API base URL for the LLM')
    parser.add_argument('--api_key', type=str, help='API key for the LLM')
    parser.add_argument('--max_tokens', type=int, help='Maximum number of tokens for the LLM')
    parser.add_argument('--temperature', type=float, help='Temperature for the LLM')
    parser.add_argument('--verbose', type=int, help='enable verbose mode', default=0)
    parser.add_argument('--tools', nargs='*',
                        choices=list(TOOL_MAP.keys()),
                        help='Tools to enable')
    return parser.parse_args()


def get_model(model_provider, model_name, api_base=None, api_key=None, max_tokens=None, temperature=None):
    params = {"id": model_name}
    if api_base is not None:
        params["api_base"] = api_base
    if api_key is not None:
        params["api_key"] = api_key
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if temperature is not None:
        params["temperature"] = temperature
    if model_provider == 'openai':
        model = OpenAIChat(**params)
    elif model_provider == 'azure':
        model = AzureOpenAIChat(**params)
    elif model_provider == 'moonshot':
        model = Moonshot(**params)
    elif model_provider == 'zhipuai':
        model = ZhipuAI(**params)
    elif model_provider == 'deepseek':
        model = DeepSeek(**params)
    elif model_provider == 'yi':
        model = Yi(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def configure_tools(tool_names: Optional[List[str]] = None) -> List:
    """
    Configure and instantiate tools based on their names.
    Uses dynamic import to load tools only when requested.

    Args:
        tool_names: List of tool names to enable. Must be keys in TOOL_MAP.

    Returns:
        List of instantiated tool objects.
    """
    if not tool_names:
        return []

    tools = []
    for name in tool_names:
        if name not in TOOL_MAP:
            console.print(f"[yellow]Warning: Tool '{name}' not recognized. Skipping.[/yellow]")
            continue

        try:
            # Dynamically import and instantiate the tool
            module_path, class_name = TOOL_MAP[name].rsplit('.', 1)
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            tool_instance = tool_class()
            tools.append(tool_instance)
            console.print(f"[green]Successfully loaded tool: {name}[/green]")
        except ImportError as e:
            console.print(f"[red]Error: Could not import tool '{name}'. Missing dependencies? {str(e)}[/red]")
        except Exception as e:
            console.print(f"[red]Error: Failed to initialize tool '{name}': {str(e)}[/red]")

    return tools


def run_interactive(agent):
    first_prompt = True
    while True:
        try:
            if first_prompt:
                console.print(Text("Enter your question (type 'exit' to quit):", style="green"))
                console.print(Text("> ", style="green"), end="")
                first_prompt = False
            else:
                console.print(Text("> ", style="green"), end="")

            line = console.input()
            query = line.strip()

            if query.lower() == 'exit':
                break
            if query:
                response = agent.run(query, stream=True)
                console.print(Text("\n", style="green"), end="")
                for chunk in response:
                    console.print(chunk.content, end="")
                console.print("\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(e)
            break


def main():
    args = parse_args()
    model = get_model(args.model_provider, args.model_name, api_base=args.api_base,
                      api_key=args.api_key, max_tokens=args.max_tokens)
    tools = configure_tools(args.tools) if args.tools else None
    debug_mode = args.verbose > 0
    if args.tools and 'python' in args.tools:
        agent = PythonAgent(model=model, add_datetime_to_instructions=True, add_history_to_messages=True,
                            tools=tools, debug_mode=debug_mode)
    else:
        agent = Agent(model=model, add_datetime_to_instructions=True, add_history_to_messages=True,
                      tools=tools, debug_mode=debug_mode)
    console.print(Text("Welcome to Agentica CLI!", style="bold green"))
    console.print(Text(f"Model provider: {args.model_provider}, Model name: {args.model_name}, "
                       f"tools: {args.tools}", style="red"))
    if args.query:
        response = agent.run(args.query, stream=True)
        for chunk in response:
            console.print(chunk.content, end="")
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
