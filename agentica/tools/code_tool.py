# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Code analysis tool - provides code understanding and quality analysis.

This tool focuses on code comprehension and quality checking.

Key features:
- Code structure analysis (AST-based)
- Code formatting (via external formatters like black, prettier)
- Code linting (via external linters like pylint, eslint)
- Symbol search and code outline generation
"""
import ast
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from agentica.tools.base import Tool
from agentica.utils.log import logger


class CodeTool(Tool):
    """
    Code analysis tool - provides code understanding, formatting, and quality checking.

    Difference from built-in tools:
    - BuiltinFileTool (buildin_tools.py): File read/write, string-based edit
    - CodeTool: Code semantic analysis, formatting, linting (code-specific features)

    This tool does NOT edit files directly - it analyzes code and optionally
    calls external formatters/linters to improve code quality.

    Example usage:
        code_tool = CodeTool()

        # Analyze code structure
        result = code_tool.analyze_code("src/main.py")

        # Format code using black
        result = code_tool.format_code("src/main.py", formatter="black")

        # Check code quality
        result = code_tool.lint_code("src/main.py", linter="pylint")
    """

    def __init__(
            self,
            work_dir: Optional[str] = None,
            enable_analysis: bool = True,
            enable_format: bool = True,
            enable_lint: bool = True,
            enable_symbols: bool = True,
            enable_outline: bool = True,
    ):
        """
        Initialize the CodeTool.

        Args:
            work_dir: The working directory for code operations. Defaults to current directory.
            enable_analysis: Whether to include the analyze_code function.
            enable_format: Whether to include the format_code function.
            enable_lint: Whether to include the lint_code function.
            enable_symbols: Whether to include the find_symbols function.
            enable_outline: Whether to include the get_code_outline function.
        """
        super().__init__(name="code_tool")
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()

        if enable_analysis:
            self.register(self.analyze_code)
        if enable_format:
            self.register(self.format_code)
        if enable_lint:
            self.register(self.lint_code)
        if enable_symbols:
            self.register(self.find_symbols)
        if enable_outline:
            self.register(self.get_code_outline)

    def _resolve_path(self, file_path: str) -> Path:
        """Resolves a file path, making it absolute if it's relative."""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.work_dir.joinpath(path)
        return path

    def _run_command(self, cmd: List[str], cwd: str = None, timeout: int = 60) -> Tuple[str, str, int]:
        """Run a shell command and return stdout, stderr, and return code."""
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd or str(self.work_dir)
            )
            stdout, stderr = process.communicate(timeout=timeout)
            return stdout, stderr, process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            return "", "Command timed out", 1
        except Exception as e:
            return "", f"Error executing command: {str(e)}", 1

    def analyze_code(self, file_path: str) -> str:
        """Analyze Python code structure using AST.

        Provides detailed information about:
        - Import statements
        - Function definitions (name, args, docstring, line number)
        - Class definitions (name, methods, bases, docstring)
        - Global variables
        - Code complexity estimate

        Args:
            file_path: Path to the Python file to analyze.

        Returns:
            JSON-formatted analysis of the code structure.

        Example:
            result = code_tool.analyze_code("src/main.py")
            # Returns: {"imports": [...], "functions": [...], "classes": [...], ...}
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return f"Error: File not found: {file_path}"

            if not file_path.endswith('.py'):
                return f"Error: Only Python files are supported. File: {file_path}"

            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()

            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return f"Error: Syntax error in {file_path}: {str(e)}"

            analysis = {
                'imports': [],
                'functions': [],
                'classes': [],
                'global_variables': [],
                'total_lines': len(code.splitlines()),
                'complexity_estimate': 'low'
            }

            # Analyze imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        analysis['imports'].append({
                            'name': name.name,
                            'alias': name.asname
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for name in node.names:
                        analysis['imports'].append({
                            'name': f"{module}.{name.name}" if module else name.name,
                            'alias': name.asname,
                            'from_import': True
                        })

            # Analyze functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    args = [arg.arg for arg in node.args.args]
                    docstring = ast.get_docstring(node) or ''
                    analysis['functions'].append({
                        'name': node.name,
                        'args': args,
                        'line_number': node.lineno,
                        'docstring': docstring[:100] if docstring else '',
                        'decorator_list': [d.id if isinstance(d, ast.Name) else "complex_decorator" for d in node.decorator_list]
                    })

            # Analyze classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [child.name for child in node.body if isinstance(child, ast.FunctionDef)]
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else "complex_base")

                    docstring = ast.get_docstring(node) or ''
                    analysis['classes'].append({
                        'name': node.name,
                        'methods': methods,
                        'bases': bases,
                        'line_number': node.lineno,
                        'docstring': docstring[:100] if docstring else ''
                    })

            # Analyze global variables
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
                    value = "complex_value"
                    if isinstance(node.value, ast.Constant):
                        value = node.value.value if hasattr(node.value, 'value') else "complex_value"

                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis['global_variables'].append({
                                'name': target.id,
                                'value': str(value)[:50],
                                'line_number': node.lineno
                            })

            # Estimate complexity
            func_count = len(analysis['functions'])
            class_count = len(analysis['classes'])
            if func_count + class_count > 20 or analysis['total_lines'] > 500:
                analysis['complexity_estimate'] = 'high'
            elif func_count + class_count > 10:
                analysis['complexity_estimate'] = 'medium'

            return json.dumps(analysis, indent=2)

        except Exception as e:
            error_msg = f"Error analyzing code in {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def format_code(self, file_path: str, formatter: str = "auto") -> str:
        """Format a code file using external formatters.

        Supported formatters:
        - Python: black, autopep8, yapf
        - JavaScript/TypeScript/JSON/HTML/CSS: prettier

        Args:
            file_path: Path to the file to format.
            formatter: Formatter to use. "auto" selects based on file extension.

        Returns:
            Result of the formatting operation.

        Example:
            result = code_tool.format_code("src/main.py")  # Uses black for Python
            result = code_tool.format_code("src/app.js", formatter="prettier")
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return f"Error: File not found: {file_path}"

            file_ext = path.suffix.lower()

            # Auto-select formatter
            if formatter == "auto":
                if file_ext == ".py":
                    formatter = "black"
                elif file_ext in [".js", ".jsx", ".ts", ".tsx", ".json", ".html", ".css"]:
                    formatter = "prettier"
                else:
                    return f"Error: Cannot auto-determine formatter for {file_ext} files"

            cmd = []
            if formatter == "black":
                cmd = ["black", str(path)]
            elif formatter == "autopep8":
                cmd = ["autopep8", "--in-place", str(path)]
            elif formatter == "yapf":
                cmd = ["yapf", "--in-place", str(path)]
            elif formatter == "prettier":
                cmd = ["prettier", "--write", str(path)]
            else:
                return f"Error: Unsupported formatter: {formatter}"

            # Check if formatter is installed
            try:
                subprocess.run([cmd[0], "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except FileNotFoundError:
                return f"Error: {formatter} is not installed. Please install it (e.g., pip install {formatter})"

            stdout, stderr, returncode = self._run_command(cmd)

            if returncode == 0:
                return f"Successfully formatted {file_path} with {formatter}"
            else:
                return f"Error formatting {file_path} with {formatter}: {stderr or stdout}"

        except Exception as e:
            error_msg = f"Error formatting {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def lint_code(self, file_path: str, linter: str = "auto") -> str:
        """Lint a code file and return issues found.

        Supported linters:
        - Python: pylint, flake8
        - JavaScript/TypeScript: eslint

        Args:
            file_path: Path to the file to lint.
            linter: Linter to use. "auto" selects based on file extension.

        Returns:
            Linting results with any issues found.

        Example:
            result = code_tool.lint_code("src/main.py")  # Uses pylint for Python
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return f"Error: File not found: {file_path}"

            file_ext = path.suffix.lower()

            if linter == "auto":
                if file_ext == ".py":
                    linter = "pylint"
                elif file_ext in [".js", ".jsx", ".ts", ".tsx"]:
                    linter = "eslint"
                else:
                    return f"Error: Cannot auto-determine linter for {file_ext} files"

            cmd = []
            if linter == "pylint":
                cmd = ["pylint", str(path)]
            elif linter == "flake8":
                cmd = ["flake8", str(path)]
            elif linter == "eslint":
                cmd = ["eslint", str(path)]
            else:
                return f"Error: Unsupported linter: {linter}"

            # Check if linter is installed
            try:
                subprocess.run([cmd[0], "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except FileNotFoundError:
                return f"Error: {linter} is not installed. Please install it first."

            stdout, stderr, returncode = self._run_command(cmd)

            if stdout or stderr:
                return f"Linting results for {file_path} with {linter}:\n\n{stdout}\n{stderr}"
            elif returncode == 0:
                return f"No issues found in {file_path} with {linter}"
            else:
                return f"Error linting {file_path} with {linter}: Return code {returncode}"

        except Exception as e:
            error_msg = f"Error linting {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def find_symbols(self, file_path: str, symbol_type: str = "all", pattern: str = "") -> str:
        """Find and list symbols (functions, classes, variables) in a code file.

        Args:
            file_path: Path to the file to analyze.
            symbol_type: Type of symbols to find ("all", "function", "class", "variable").
            pattern: Optional regex pattern to filter symbols by name.

        Returns:
            JSON-formatted list of found symbols.

        Example:
            result = code_tool.find_symbols("src/main.py", symbol_type="function", pattern="test_.*")
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return f"Error: File not found: {file_path}"

            if not file_path.endswith('.py'):
                return f"Error: Only Python files are supported. File: {file_path}"

            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()

            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return f"Error: Syntax error in {file_path}: {str(e)}"

            symbols = []

            def matches_pattern(name):
                if not pattern:
                    return True
                try:
                    return bool(re.search(pattern, name))
                except re.error:
                    return pattern in name

            # Find functions
            if symbol_type in ["all", "function"]:
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and matches_pattern(node.name):
                        args = [arg.arg for arg in node.args.args]
                        symbols.append({
                            'type': 'function',
                            'name': node.name,
                            'line': node.lineno,
                            'args': args,
                            'docstring': (ast.get_docstring(node) or '')[:100]
                        })

            # Find classes
            if symbol_type in ["all", "class"]:
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and matches_pattern(node.name):
                        symbols.append({
                            'type': 'class',
                            'name': node.name,
                            'line': node.lineno,
                            'docstring': (ast.get_docstring(node) or '')[:100],
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        })

            # Find variables
            if symbol_type in ["all", "variable"]:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and matches_pattern(target.id):
                                symbols.append({
                                    'type': 'variable',
                                    'name': target.id,
                                    'line': node.lineno
                                })

            # Sort by line number
            symbols.sort(key=lambda x: x['line'])

            return json.dumps(symbols, indent=2)

        except Exception as e:
            error_msg = f"Error finding symbols in {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def get_code_outline(self, file_path: str) -> str:
        """Generate a hierarchical outline of a code file's structure.

        Returns a Markdown-formatted outline showing:
        - Imports
        - Classes with methods
        - Standalone functions
        - Global variables

        Args:
            file_path: Path to the file to outline.

        Returns:
            Markdown-formatted outline of the code structure.

        Example:
            print(code_tool.get_code_outline("src/main.py"))
            # Output:
            # # Code Outline: main.py
            #
            # ## Imports
            # - `import os`
            #
            # ## Classes
            # ### MyClass - Line 10
            # - **method1()** - Line 15
            #
            # ## Functions
            # ### standalone_func() - Line 30
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return f"Error: File not found: {file_path}"

            if not file_path.endswith('.py'):
                return f"Error: Only Python files are supported. File: {file_path}"

            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()

            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return f"Error: Syntax error in {file_path}: {str(e)}"

            outline = [f"# Code Outline: {Path(file_path).name}", ""]

            # Process imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.asname:
                            imports.append(f"import {name.name} as {name.asname}")
                        else:
                            imports.append(f"import {name.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for name in node.names:
                        if name.asname:
                            imports.append(f"from {module} import {name.name} as {name.asname}")
                        else:
                            imports.append(f"from {module} import {name.name}")

            if imports:
                outline.append("## Imports")
                for imp in sorted(set(imports)):
                    outline.append(f"- `{imp}`")
                outline.append("")

            # Process classes
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': (ast.get_docstring(node) or '').split('.')[0][:80],
                        'methods': []
                    }
                    if node.bases:
                        bases = []
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                bases.append(base.id)
                            else:
                                bases.append("...")
                        class_info['bases'] = bases

                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            method_info = {
                                'name': child.name,
                                'line': child.lineno,
                                'args': [arg.arg for arg in child.args.args]
                            }
                            class_info['methods'].append(method_info)

                    classes.append(class_info)

            if classes:
                outline.append("## Classes")
                for cls in sorted(classes, key=lambda x: x['line']):
                    if 'bases' in cls and cls['bases']:
                        outline.append(f"### {cls['name']}({', '.join(cls['bases'])}) - Line {cls['line']}")
                    else:
                        outline.append(f"### {cls['name']} - Line {cls['line']}")

                    if cls['docstring']:
                        outline.append(f"> {cls['docstring']}")

                    for method in sorted(cls['methods'], key=lambda x: x['line']):
                        args_str = ', '.join(method['args'])
                        outline.append(f"- **{method['name']}({args_str})** - Line {method['line']}")

                outline.append("")

            # Process standalone functions
            functions = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': (ast.get_docstring(node) or '').split('.')[0][:80],
                        'args': [arg.arg for arg in node.args.args]
                    }
                    functions.append(func_info)

            if functions:
                outline.append("## Functions")
                for func in sorted(functions, key=lambda x: x['line']):
                    args_str = ', '.join(func['args'])
                    outline.append(f"### {func['name']}({args_str}) - Line {func['line']}")
                    if func['docstring']:
                        outline.append(f"> {func['docstring']}")
                outline.append("")

            return "\n".join(outline)

        except Exception as e:
            error_msg = f"Error generating outline for {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg


if __name__ == '__main__':
    # Simple test
    tool = CodeTool()

    # Create a sample Python file
    sample_code = '''
import os
import sys
from datetime import datetime

GLOBAL_VAR = "This is a global variable"

def sample_function(arg1, arg2):
    """Sample function that does something."""
    return arg1 + arg2

class SampleClass:
    """Sample class demonstrating structure."""
    def __init__(self, value):
        self.value = value

    def get_value(self):
        """Return the stored value."""
        return self.value
    '''

    with open("sample_code.py", "w") as f:
        f.write(sample_code)

    print("Code Outline:")
    print(tool.get_code_outline("sample_code.py"))

    print("\nCode Analysis:")
    print(tool.analyze_code("sample_code.py"))

    print("\nFind Symbols:")
    print(tool.find_symbols("sample_code.py", symbol_type="function"))

    # Clean up
    import os
    if os.path.exists("sample_code.py"):
        os.remove("sample_code.py")
