# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: command line interface for agentica
"""
import argparse
from rich.console import Console
from rich.text import Text
from agentica import Agent, OpenAIChat, MoonshotChat, AzureOpenAIChat, YiChat, ZhipuAIChat, DeepSeekChat
from agentica.tools.search_serper_tool import SearchSerperTool
from agentica.tools.file_tool import FileTool
from agentica.tools.shell_tool import ShellTool
from agentica.tools.jina_tool import JinaTool
from agentica.tools.wikipedia_tool import WikipediaTool

console = Console()


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
                        choices=['search_serper', 'file_tool', 'shell_tool', 'yfinance_tool', 'web_search_pro',
                                 'cogview', 'cogvideo', 'jina', 'wikipedia'], help='Tools to enable')
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
        model = MoonshotChat(**params)
    elif model_provider == 'zhipuai':
        model = ZhipuAIChat(**params)
    elif model_provider == 'deepseek':
        model = DeepSeekChat(**params)
    elif model_provider == 'yi':
        model = YiChat(**params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def configure_tools(tool_names):
    """
    Configure tools to enable
    :param tool_names: list of tool names
    :return: list of tool instances
    """
    tools = []
    if 'search_serper' in tool_names:
        tools.append(SearchSerperTool())
    if 'file_tool' in tool_names:
        tools.append(FileTool())
    if 'shell_tool' in tool_names:
        tools.append(ShellTool())
    if 'yfinance_tool' in tool_names:
        from agentica.tools.yfinance_tool import YFinanceTool
        tools.append(YFinanceTool())
    if 'web_search_pro' in tool_names:
        from agentica.tools.web_search_pro_tool import WebSearchProTool
        tools.append(WebSearchProTool())
    if 'cogview' in tool_names:
        from agentica.tools.cogview_tool import CogViewTool
        tools.append(CogViewTool())
    if 'cogvideo' in tool_names:
        from agentica.tools.cogvideo_tool import CogVideoTool
        tools.append(CogVideoTool())
    if 'jina' in tool_names:
        tools.append(JinaTool())
    if 'wikipedia' in tool_names:
        tools.append(WikipediaTool())
    return tools


def run_interactive(agent):
    first_prompt = True
    while True:
        try:
            if first_prompt:
                console.print(Text("Enter your question (end with empty line, type 'exit' to quit):", style="green"))
                console.print(Text(">>> ", style="green"), end="")
                first_prompt = False
            else:
                console.print(Text(">>> ", style="green"), end="")
            multi_line_input = []
            while True:
                line = console.input()
                if line.strip() == "":
                    break
                multi_line_input.append(line)
            query = "\n".join(multi_line_input).strip()
            if query.lower() == 'exit':
                break
            if query:
                response = agent.run(query, stream=True)
                console.print(Text("> ", style="green"), end="")
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
    agent = Agent(model=model, add_datetime_to_instructions=True, add_history_to_messages=True,
                  tools=tools, debug_mode=debug_mode)
    if args.query:
        result = agent.run(args.query).content
        console.print(result)
    else:
        run_interactive(agent)


if __name__ == "__main__":
    main()
