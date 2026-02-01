# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Universal Python library analyzer - explore any library's modules, classes, and functions.

Usage:
    python view_python_module.py --module requests
    python view_python_module.py --module numpy.linalg
    python view_python_module.py --module pandas.DataFrame
"""
import importlib
import inspect
from typing import Literal, Callable, List, Optional, Any

from pydantic import BaseModel


def get_class_signature(cls: type) -> str:
    """Get the signature of a class including its methods.

    Args:
        cls: A class object.

    Returns:
        str: The formatted signature of the class.
    """
    class_name = cls.__name__

    # Get base classes
    bases = [b.__name__ for b in cls.__bases__ if b.__name__ != 'object']
    bases_str = f"({', '.join(bases)})" if bases else ""

    class_str = f"class {class_name}{bases_str}:\n"

    # Add docstring
    class_docstring = cls.__doc__ or ""
    if class_docstring:
        doc_lines = class_docstring.strip().split('\n')
        if len(doc_lines) > 5:
            class_docstring = '\n'.join(doc_lines[:5]) + '\n    ...'
        class_str += f'    """{class_docstring}"""\n'

    # Get methods
    methods = []
    for name, method in inspect.getmembers(cls):
        # Skip private methods except __init__ and __call__
        if name.startswith("_") and name not in ["__init__", "__call__"]:
            continue

        if not (inspect.isfunction(method) or inspect.ismethod(method)):
            continue

        # Check if method belongs to this class
        if hasattr(method, '__qualname__'):
            if method.__qualname__.split(".")[0] != class_name:
                continue

        try:
            sig = inspect.signature(method)
            method_str = f"    def {name}{sig}:\n"
            method_docstring = method.__doc__ or ""
            if method_docstring:
                doc_lines = method_docstring.strip().split('\n')
                if len(doc_lines) > 3:
                    method_docstring = '\n'.join(doc_lines[:3]) + '\n        ...'
                method_str += f'        """{method_docstring}"""\n'
            methods.append(method_str)
        except (ValueError, TypeError):
            methods.append(f"    def {name}(...):\n        pass\n")

    if methods:
        class_str += "\n".join(methods[:10])  # Limit to 10 methods
        if len(methods) > 10:
            class_str += f"\n    # ... and {len(methods) - 10} more methods"

    return class_str


def get_function_signature(func: Callable) -> str:
    """Get the signature of a function.

    Args:
        func: A function object.

    Returns:
        str: The formatted signature of the function.
    """
    try:
        sig = inspect.signature(func)
        func_str = f"def {func.__name__}{sig}:\n"
    except (ValueError, TypeError):
        func_str = f"def {func.__name__}(...):\n"

    docstring = func.__doc__ or ""
    if docstring:
        doc_lines = docstring.strip().split('\n')
        if len(doc_lines) > 5:
            docstring = '\n'.join(doc_lines[:5]) + '\n    ...'
        func_str += f'    """{docstring}"""\n'

    return func_str


class ModuleInfo(BaseModel):
    """Information about a module member (class, function, or submodule)."""

    name: str
    full_path: str
    type: Literal["class", "function", "module", "other"]
    docstring: str
    signature: Optional[str] = None
    source_file: Optional[str] = None

    model_config = {"arbitrary_types_allowed": True}


def _truncate(text: str, max_length: int = 150) -> str:
    """Truncate text to a maximum length."""
    if not text:
        return ""
    text = text.strip().replace('\n', ' ')
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def get_module_members(module: Any, module_path: str) -> List[ModuleInfo]:
    """Get all public members of a module.

    Args:
        module: The module object.
        module_path: The full path of the module (e.g., 'requests.api').

    Returns:
        List of ModuleInfo objects describing each member.
    """
    members = []

    # Prefer __all__ if available, otherwise use dir()
    names = getattr(module, '__all__', None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith('_')]

    for name in names:
        try:
            obj = getattr(module, name)
            full_path = f"{module_path}.{name}"

            if inspect.isclass(obj):
                members.append(ModuleInfo(
                    name=name,
                    full_path=full_path,
                    type="class",
                    docstring=_truncate(obj.__doc__ or ""),
                    signature=get_class_signature(obj),
                ))
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                members.append(ModuleInfo(
                    name=name,
                    full_path=full_path,
                    type="function",
                    docstring=_truncate(obj.__doc__ or ""),
                    signature=get_function_signature(obj) if inspect.isfunction(obj) else None,
                ))
            elif inspect.ismodule(obj):
                members.append(ModuleInfo(
                    name=name,
                    full_path=full_path,
                    type="module",
                    docstring=_truncate(obj.__doc__ or ""),
                ))
        except (AttributeError, TypeError):
            continue

    return members


def view_python_library(module_path: str) -> str:
    """View any Python library's structure by module path.

    Explore modules, classes, functions, and their documentation for any
    installed Python library.

    Args:
        module_path: The module path to explore, e.g., 'requests', 'numpy.linalg',
                    or 'pandas.DataFrame'. Use dot notation for submodules.

    Returns:
        str: Formatted information about the module, class, or function.

    Examples:
        >>> view_python_library('requests')
        # Shows top-level structure of requests library

        >>> view_python_library('requests.get')
        # Shows signature and docs for requests.get function

        >>> view_python_library('pandas.DataFrame')
        # Shows DataFrame class signature and methods
    """
    if not module_path:
        return "Error: Please provide a module path (e.g., 'requests', 'numpy.linalg')"

    parts = module_path.split('.')
    library_name = parts[0]

    # Try to import the base library
    try:
        base_module = importlib.import_module(library_name)
    except ImportError as e:
        return f"Error: Cannot import '{library_name}'. Is it installed?\n\nInstall with: pip install {library_name}\n\nOriginal error: {e}"

    # Navigate to the target
    current = base_module
    current_path = library_name

    for part in parts[1:]:
        # First try as attribute (for classes/functions)
        if hasattr(current, part):
            current = getattr(current, part)
            current_path = f"{current_path}.{part}"
        else:
            # Try as submodule
            try:
                current = importlib.import_module(f"{current_path}.{part}")
                current_path = f"{current_path}.{part}"
            except ImportError:
                return f"Error: Cannot find '{part}' in '{current_path}'.\n\nTry: view_python_library('{current_path}') to see available members."

    # Format output based on what we found
    if inspect.ismodule(current):
        return _format_module(current, current_path)
    elif inspect.isclass(current):
        return _format_class(current, current_path)
    elif inspect.isfunction(current) or inspect.isbuiltin(current):
        return _format_function(current, current_path)
    else:
        # It's a value or other object
        return f"'{current_path}' is a {type(current).__name__}:\n\nValue: {repr(current)[:500]}"


def _format_module(module: Any, path: str) -> str:
    """Format module information for display."""
    lines = [f"# Module: {path}\n"]

    if module.__doc__:
        doc = _truncate(module.__doc__, 300)
        lines.append(f"{doc}\n")

    members = get_module_members(module, path)

    # Group by type
    classes = [m for m in members if m.type == "class"]
    functions = [m for m in members if m.type == "function"]
    submodules = [m for m in members if m.type == "module"]

    if submodules:
        lines.append("\n## Submodules")
        for m in submodules[:15]:
            lines.append(f"- {m.full_path}: {m.docstring or '(no description)'}")
        if len(submodules) > 15:
            lines.append(f"  ... and {len(submodules) - 15} more")

    if classes:
        lines.append("\n## Classes")
        for m in classes[:15]:
            lines.append(f"- {m.name}: {m.docstring or '(no description)'}")
        if len(classes) > 15:
            lines.append(f"  ... and {len(classes) - 15} more")

    if functions:
        lines.append("\n## Functions")
        for m in functions[:20]:
            lines.append(f"- {m.name}: {m.docstring or '(no description)'}")
        if len(functions) > 20:
            lines.append(f"  ... and {len(functions) - 20} more")

    lines.append(f"\n---\nUse view_python_library('{path}.<name>') for details on specific items.")

    return "\n".join(lines)


def _format_class(cls: type, path: str) -> str:
    """Format class information for display."""
    lines = [f"# Class: {path}\n"]

    # Get source file if available
    try:
        source_file = inspect.getfile(cls)
        lines.append(f"Source: {source_file}\n")
    except (TypeError, OSError):
        pass

    lines.append("```python")
    lines.append(get_class_signature(cls))
    lines.append("```")

    return "\n".join(lines)


def _format_function(func: Callable, path: str) -> str:
    """Format function information for display."""
    lines = [f"# Function: {path}\n"]

    # Get source file if available
    try:
        source_file = inspect.getfile(func)
        source_lines, start_line = inspect.getsourcelines(func)
        lines.append(f"Source: {source_file}:{start_line}\n")
    except (TypeError, OSError):
        pass

    lines.append("```python")
    lines.append(get_function_signature(func))
    lines.append("```")

    # Full docstring
    if func.__doc__:
        lines.append("\n## Documentation\n")
        lines.append(func.__doc__.strip())

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Explore any Python library's structure, classes, and functions.",
        epilog="""
Examples:
  python view_python_module.py --module requests
  python view_python_module.py --module numpy.linalg
  python view_python_module.py --module pandas.DataFrame
  python view_python_module.py --module agentica.Agent
        """
    )
    parser.add_argument(
        "--module", "-m",
        type=str,
        required=True,
        help="Module path to explore (e.g., 'requests', 'numpy.linalg', 'pandas.DataFrame')"
    )

    args = parser.parse_args()
    result = view_python_library(args.module)
    print(result)
