# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Get the signatures of functions and classes in the agentica library.

Usage:
    python view_agentica_module.py --module agentica
    python view_agentica_module.py --module agentica.agent
    python view_agentica_module.py --module agentica.tools
"""
from typing import Literal, Callable, List

import inspect
from pydantic import BaseModel


def get_class_signature(cls: type) -> str:
    """Get the signature of a class.

    Args:
        cls: A class object.

    Returns:
        str: The signature of the class.
    """
    class_name = cls.__name__
    class_docstring = cls.__doc__ or ""

    class_str = f"class {class_name}:\n"
    if class_docstring:
        # Truncate long docstrings
        doc_lines = class_docstring.strip().split('\n')
        if len(doc_lines) > 5:
            class_docstring = '\n'.join(doc_lines[:5]) + '\n    ...'
        class_str += f'    """{class_docstring}"""\n'

    methods = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if method.__qualname__.split(".")[0] != class_name:
            continue

        if name.startswith("_") and name not in ["__init__", "__call__"]:
            continue

        try:
            sig = inspect.signature(method)
            method_str = f"    def {name}{sig}:\n"
            method_docstring = method.__doc__ or ""
            if method_docstring:
                # Truncate long docstrings
                doc_lines = method_docstring.strip().split('\n')
                if len(doc_lines) > 3:
                    method_docstring = '\n'.join(doc_lines[:3]) + '\n        ...'
                method_str += f'        """{method_docstring}"""\n'
            methods.append(method_str)
        except (ValueError, TypeError):
            continue

    class_str += "\n".join(methods)
    return class_str


def get_function_signature(func: Callable) -> str:
    """Get the signature of a function."""
    try:
        sig = inspect.signature(func)
        method_str = f"def {func.__name__}{sig}:\n"
    except (ValueError, TypeError):
        method_str = f"def {func.__name__}(...):\n"

    method_docstring = func.__doc__ or ""
    if method_docstring:
        doc_lines = method_docstring.strip().split('\n')
        if len(doc_lines) > 3:
            method_docstring = '\n'.join(doc_lines[:3]) + '\n   ...'
        method_str += f'   """{method_docstring}"""\n'

    return method_str


class FuncOrCls(BaseModel):
    """The class records the module, signature, docstring, reference, and type"""

    module: str
    signature: str
    docstring: str
    reference: str
    type: Literal["function", "class"]

    def __init__(
        self,
        module: str,
        signature: str,
        docstring: str,
        reference: str,
        type: Literal["function", "class"],
    ) -> None:
        super().__init__(
            module=module,
            signature=signature.strip(),
            docstring=docstring.strip(),
            reference=reference,
            type=type,
        )


def _truncate_docstring(docstring: str, max_length: int = 200) -> str:
    """Truncate the docstring to a maximum length."""
    if len(docstring) > max_length:
        return docstring[:max_length] + "..."
    return docstring


def get_agentica_module_signatures() -> List[FuncOrCls]:
    """Get the signatures of functions and classes in the agentica library.

    Returns:
        A list of FuncOrCls instances representing the functions and
        classes in the agentica library.
    """
    try:
        import agentica
    except ImportError:
        return []

    signatures = []

    # Get all exported names from agentica
    all_names = getattr(agentica, '__all__', dir(agentica))

    for name in all_names:
        if name.startswith('_'):
            continue

        try:
            obj = getattr(agentica, name)
            path_module = f"agentica.{name}"

            if inspect.isfunction(obj):
                try:
                    file = inspect.getfile(obj)
                    source_lines, start_line = inspect.getsourcelines(obj)
                    signatures.append(
                        FuncOrCls(
                            module=path_module,
                            signature=get_function_signature(obj),
                            docstring=_truncate_docstring(obj.__doc__ or ""),
                            reference=f"{file}: {start_line}-{start_line + len(source_lines)}",
                            type="function",
                        ),
                    )
                except (OSError, TypeError):
                    continue

            elif inspect.isclass(obj):
                try:
                    file = inspect.getfile(obj)
                    source_lines, start_line = inspect.getsourcelines(obj)
                    signatures.append(
                        FuncOrCls(
                            module=path_module,
                            signature=get_class_signature(obj),
                            docstring=_truncate_docstring(obj.__doc__ or ""),
                            reference=f"{file}: {start_line}-{start_line + len(source_lines)}",
                            type="class",
                        ),
                    )
                except (OSError, TypeError):
                    continue

            elif inspect.ismodule(obj):
                # Handle submodules
                sub_all = getattr(obj, '__all__', [])
                for sub_name in sub_all:
                    if sub_name.startswith('_'):
                        continue
                    try:
                        sub_obj = getattr(obj, sub_name)
                        sub_path = f"{path_module}.{sub_name}"

                        if inspect.isclass(sub_obj):
                            file = inspect.getfile(sub_obj)
                            source_lines, start_line = inspect.getsourcelines(sub_obj)
                            signatures.append(
                                FuncOrCls(
                                    module=sub_path,
                                    signature=get_class_signature(sub_obj),
                                    docstring=_truncate_docstring(sub_obj.__doc__ or ""),
                                    reference=f"{file}: {start_line}-{start_line + len(source_lines)}",
                                    type="class",
                                ),
                            )
                        elif inspect.isfunction(sub_obj):
                            file = inspect.getfile(sub_obj)
                            source_lines, start_line = inspect.getsourcelines(sub_obj)
                            signatures.append(
                                FuncOrCls(
                                    module=sub_path,
                                    signature=get_function_signature(sub_obj),
                                    docstring=_truncate_docstring(sub_obj.__doc__ or ""),
                                    reference=f"{file}: {start_line}-{start_line + len(source_lines)}",
                                    type="function",
                                ),
                            )
                    except (AttributeError, OSError, TypeError):
                        continue

        except (AttributeError, TypeError):
            continue

    return signatures


def view_agentica_library(module: str) -> str:
    """View Agentica's Python library by given a module name
    (e.g. agentica), and return the module's submodules, classes, and
    functions. Given a class name, return the class's documentation, methods,
    and their signatures. Given a function name, return the function's
    documentation and signature. If you don't have any information about
    Agentica library, try to use "agentica" to view the available top
    modules.

    Note this function only provides the module's brief information.
    For more information, you should view the source code.

    Args:
        module: The module name to view, which should be a module path separated
            by dots (e.g. "agentica.agent"). It can refer to a module,
            a class, or a function.

    Returns:
        str: Information about the module, class, or function.
    """
    if not module.startswith("agentica"):
        return (
            f"Module '{module}' is invalid. The input module should be "
            f"'agentica' or submodule of 'agentica.xxx.xxx' "
            f"(separated by dots)."
        )

    try:
        import agentica
    except ImportError:
        return "Error: agentica library is not installed. Please install it with: pip install agentica"

    # Top-level modules description
    agentica_top_modules = {
        "agent": "Core Agent class for building AI agents with tools and memory",
        "model": "LLM model implementations (OpenAI, DeepSeek, Qwen, ZhipuAI, etc.)",
        "tools": "Built-in tools for agents (FileTool, ShellTool, SearchTool, etc.)",
        "memory": "Memory management for agents (AgentMemory, MemoryManager)",
        "knowledge": "Knowledge base and RAG implementations",
        "workflow": "Workflow orchestration for multi-agent systems",
        "mcp": "MCP (Model Context Protocol) support",
        "db": "Database backends for persistence (SqliteDb, PostgresDb)",
        "vectordb": "Vector database implementations for RAG",
        "emb": "Embedding model implementations",
        "document": "Document processing utilities",
        "compression": "Token compression utilities",
    }

    # Top modules
    if module == "agentica":
        top_modules_description = [
            "The top-level modules in Agentica library:",
        ] + [
            f"- agentica.{k}: {v}"
            for k, v in agentica_top_modules.items()
        ] + [
            "",
            "You can further view the classes/functions within above "
            "modules by calling this function with the module name.",
            "",
            "Example: view_agentica_library('agentica.agent')",
        ]
        return "\n".join(top_modules_description)

    # Get all module signatures
    modules = get_agentica_module_signatures()

    # Check for exact match
    for as_module in modules:
        if as_module.module == module:
            return f"""- The signature of '{module}':
```python
{as_module.signature}
```

- Source code reference: {as_module.reference}"""

    # Check for submodules
    collected_modules = []
    for as_module in modules:
        if as_module.module.startswith(module):
            collected_modules.append(as_module)

    if len(collected_modules) > 0:
        collected_modules_content = [
            f"The classes/functions and their truncated docstring in "
            f"'{module}' module:",
        ] + [
            f"- {_.module}: {repr(_.docstring)}" for _ in collected_modules
        ] + [
            "",
            "The docstring is truncated for limited context. For detailed "
            "signature and methods, call this function with the above "
            "module name",
        ]
        return "\n".join(collected_modules_content)

    # Try to import and inspect the module directly
    try:
        parts = module.split('.')
        obj = agentica
        for part in parts[1:]:  # Skip 'agentica'
            obj = getattr(obj, part)

        if inspect.isclass(obj):
            return f"""- The signature of '{module}':
```python
{get_class_signature(obj)}
```"""
        elif inspect.isfunction(obj):
            return f"""- The signature of '{module}':
```python
{get_function_signature(obj)}
```"""
        elif inspect.ismodule(obj):
            # List contents of the module
            contents = []
            for name in dir(obj):
                if name.startswith('_'):
                    continue
                sub_obj = getattr(obj, name, None)
                if sub_obj is None:
                    continue
                if inspect.isclass(sub_obj):
                    contents.append(f"- {module}.{name} (class): {_truncate_docstring(sub_obj.__doc__ or '', 100)}")
                elif inspect.isfunction(sub_obj):
                    contents.append(f"- {module}.{name} (function): {_truncate_docstring(sub_obj.__doc__ or '', 100)}")

            if contents:
                return f"Contents of '{module}':\n" + "\n".join(contents[:20])

    except (AttributeError, ImportError):
        pass

    return (
        f"Module '{module}' not found. Use 'agentica' to view the "
        f"top-level modules to ensure the given module is valid."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="View Agentica library modules, classes, and functions."
    )
    parser.add_argument(
        "--module",
        type=str,
        default="agentica",
        help="The module name to view, e.g. 'agentica' or 'agentica.agent'",
    )
    args = parser.parse_args()

    res = view_agentica_library(module=args.module)
    print(res)
