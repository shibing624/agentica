# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tools for code analysis, formatting, linting, refactoring, and more.
"""
import ast
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

from agentica.tools.base import Tool
from agentica.utils.log import logger


class CodeTool(Tool):
    """
    A toolkit for code operations, including analysis, formatting, and linting.
    Essential for building code editor functionality.
    """

    def __init__(
            self,
            work_dir: Optional[str] = None,
            analyze_code: bool = True,
            format_code: bool = True,
            run_code: bool = True,
            lint_code: bool = True,
            find_symbols: bool = True,
            get_code_outline: bool = True,
    ):
        """
        Initialize the CodeTool.

        Args:
            work_dir: The working directory for code operations. Defaults to current directory.
            analyze_code: Whether to include the analyze_code function.
            format_code: Whether to include the format_code function.
            run_code: Whether to include the run_code function.
            lint_code: Whether to include the lint_code function.
            find_symbols: Whether to include the find_symbols function.
            get_code_outline: Whether to include the get_code_outline function.
        """
        super().__init__(name="code_tool")
        self.work_dir = Path(work_dir) if work_dir else Path.cwd()
        
        if analyze_code:
            self.register(self.analyze_code)
        if format_code:
            self.register(self.format_code)
        if run_code:
            self.register(self.run_code)
        if lint_code:
            self.register(self.lint_code)
        if find_symbols:
            self.register(self.find_symbols)
        if get_code_outline:
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
        """Analyze Python code and return useful information about it.

        Args:
            file_path (str): Path to the Python file to analyze.

        Example:
            from agentica.tools.code_tool import CodeTool
            analyzer = CodeTool()
            result = analyzer.analyze_code("example.py")
            print(result)

        Returns:
            str: JSON-formatted analysis of the code including functions, classes, imports, etc.
        """
        try:
            path = self._resolve_path(file_path)
            
            if not path.exists():
                return f"Error: File not found: {file_path}"
            
            if not file_path.endswith('.py'):
                return f"Error: Only Python files are supported for analysis. File: {file_path}"
            
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
                'complexity_estimate': 'low'  # Default value
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
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    
                    # Extract docstring if it exists
                    docstring = ast.get_docstring(node) or ''
                    
                    analysis['functions'].append({
                        'name': node.name,
                        'args': args,
                        'line_number': node.lineno,
                        'docstring': docstring,
                        'decorator_list': [d.id if isinstance(d, ast.Name) else "complex_decorator" for d in node.decorator_list]
                    })
            
            # Analyze classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            methods.append(child.name)
                    
                    # Get base classes
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(f"{base.value.id}.{base.attr}" if hasattr(base.value, 'id') else "complex_base")
                    
                    # Extract docstring if it exists
                    docstring = ast.get_docstring(node) or ''
                    
                    analysis['classes'].append({
                        'name': node.name,
                        'methods': methods,
                        'bases': bases,
                        'line_number': node.lineno,
                        'docstring': docstring
                    })
            
            # Analyze global variables
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
                    if isinstance(node.value, ast.Constant):
                        value = node.value.value if hasattr(node.value, 'value') else "complex_value"
                    else:
                        value = "complex_value"
                    
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            analysis['global_variables'].append({
                                'name': target.id,
                                'value': str(value),
                                'line_number': node.lineno
                            })
            
            # Estimate complexity
            complexity = 'low'
            if len(analysis['functions']) + len(analysis['classes']) > 10:
                complexity = 'medium'
            if len(analysis['functions']) + len(analysis['classes']) > 20:
                complexity = 'high'
            if analysis['total_lines'] > 500:
                complexity = 'high'
            analysis['complexity_estimate'] = complexity
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            error_msg = f"Error analyzing code in {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def format_code(self, file_path: str, formatter: str = "black") -> str:
        """Format a code file using the specified formatter.

        Args:
            file_path (str): Path to the file to format.
            formatter (str): Formatter to use (black, autopep8, yapf, prettier). Defaults to black.

        Example:
            from agentica.tools.code_tool import CodeTool
            formatter = CodeTool()
            result = formatter.format_code("example.py", formatter="black")
            print(result)

        Returns:
            str: Result of the formatting operation.
        """
        try:
            path = self._resolve_path(file_path)
            
            if not path.exists():
                return f"Error: File not found: {file_path}"
            
            # Determine file type and appropriate formatter
            file_ext = path.suffix.lower()
            
            # Override formatter based on file extension if needed
            if formatter == "black" and file_ext != ".py":
                if file_ext in [".js", ".jsx", ".ts", ".tsx", ".json", ".html", ".css"]:
                    formatter = "prettier"
            
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
                return f"Error: {formatter} is not installed. Please install it first."
            
            # Run formatter
            stdout, stderr, returncode = self._run_command(cmd)
            
            if returncode == 0:
                return f"Successfully formatted {file_path} with {formatter}"
            else:
                return f"Error formatting {file_path} with {formatter}: {stderr}"
            
        except Exception as e:
            error_msg = f"Error formatting {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def run_code(self, file_path: str, args: str = "", timeout: int = 30) -> str:
        """Run a code file and return the output.

        Args:
            file_path (str): Path to the file to run.
            args (str): Command-line arguments to pass to the program.
            timeout (int): Maximum execution time in seconds.

        Example:
            from agentica.tools.code_tool import CodeTool
            runner = CodeTool()
            result = runner.run_code("example.py", args="--verbose")
            print(result)

        Returns:
            str: Output from running the code.
        """
        try:
            path = self._resolve_path(file_path)
            
            if not path.exists():
                return f"Error: File not found: {file_path}"
            
            # Determine how to run the file based on extension
            file_ext = path.suffix.lower()
            
            cmd = []
            if file_ext == ".py":
                cmd = ["python", str(path)]
            elif file_ext in [".js", ".mjs"]:
                cmd = ["node", str(path)]
            elif file_ext == ".sh":
                cmd = ["bash", str(path)]
            elif file_ext == ".rb":
                cmd = ["ruby", str(path)]
            elif file_ext == ".pl":
                cmd = ["perl", str(path)]
            elif file_ext in [".go"]:
                cmd = ["go", "run", str(path)]
            elif path.is_file() and os.access(path, os.X_OK):
                # File is executable
                cmd = [str(path)]
            else:
                return f"Error: Unsupported file type or file is not executable: {file_path}"
            
            # Add arguments if provided
            if args:
                cmd.extend(args.split())
            
            # Run the command
            stdout, stderr, returncode = self._run_command(cmd, timeout=timeout)
            
            result = f"Exit code: {returncode}\n\nStandard Output:\n{stdout}\n\nStandard Error:\n{stderr}"
            return result
            
        except Exception as e:
            error_msg = f"Error running {file_path}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def lint_code(self, file_path: str, linter: str = "auto") -> str:
        """Lint a code file and return issues found.

        Args:
            file_path (str): Path to the file to lint.
            linter (str): Linter to use (auto, pylint, flake8, eslint). Defaults to auto.

        Example:
            from agentica.tools.code_tool import CodeTool
            linter = CodeTool()
            result = linter.lint_code("example.py")
            print(result)

        Returns:
            str: Result of the linting operation with any issues found.
        """
        try:
            path = self._resolve_path(file_path)
            
            if not path.exists():
                return f"Error: File not found: {file_path}"
            
            # Determine file type and appropriate linter
            file_ext = path.suffix.lower()
            
            if linter == "auto":
                if file_ext == ".py":
                    linter = "pylint"
                elif file_ext in [".js", ".jsx", ".ts", ".tsx"]:
                    linter = "eslint"
                else:
                    return f"Error: Could not determine appropriate linter for {file_path}. Please specify a linter."
            
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
            
            # Run linter
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
            file_path (str): Path to the file to analyze.
            symbol_type (str): Type of symbols to find (all, function, class, variable).
            pattern (str): Regular expression pattern to filter symbols by name.

        Example:
            from agentica.tools.code_tool import CodeTool
            finder = CodeTool()
            result = finder.find_symbols("example.py", symbol_type="function", pattern="test_")
            print(result)

        Returns:
            str: JSON-formatted list of found symbols.
        """
        try:
            path = self._resolve_path(file_path)
            
            if not path.exists():
                return f"Error: File not found: {file_path}"
            
            if not file_path.endswith('.py'):
                return f"Error: Only Python files are supported for symbol finding. File: {file_path}"
            
            with open(path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return f"Error: Syntax error in {file_path}: {str(e)}"
            
            symbols = []
            
            # Helper to check pattern
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
                            'docstring': ast.get_docstring(node) or ''
                        })
            
            # Find classes
            if symbol_type in ["all", "class"]:
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and matches_pattern(node.name):
                        symbols.append({
                            'type': 'class',
                            'name': node.name,
                            'line': node.lineno,
                            'docstring': ast.get_docstring(node) or '',
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

        Args:
            file_path (str): Path to the file to outline.

        Example:
            from agentica.tools.code_tool import CodeTool
            outliner = CodeTool()
            result = outliner.get_code_outline("example.py")
            print(result)

        Returns:
            str: Markdown-formatted outline of the code structure.
        """
        try:
            path = self._resolve_path(file_path)
            
            if not path.exists():
                return f"Error: File not found: {file_path}"
            
            if not file_path.endswith('.py'):
                return f"Error: Only Python files are supported for outlining. File: {file_path}"
            
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
                        'docstring': ast.get_docstring(node) or '',
                        'methods': []
                    }
                    
                    # Get base classes
                    if node.bases:
                        bases = []
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                bases.append(base.id)
                            else:
                                bases.append("...")
                        class_info['bases'] = bases
                    
                    # Get methods
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            method_info = {
                                'name': child.name,
                                'line': child.lineno,
                                'docstring': ast.get_docstring(child) or '',
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
                        outline.append(f"> {cls['docstring'].split('.')[0]}.")
                    
                    for method in sorted(cls['methods'], key=lambda x: x['line']):
                        args_str = ', '.join(method['args'])
                        outline.append(f"- **{method['name']}({args_str})** - Line {method['line']}")
                        if method['docstring']:
                            outline.append(f"  > {method['docstring'].split('.')[0]}.")
                
                outline.append("")
            
            # Process standalone functions
            functions = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node) or '',
                        'args': [arg.arg for arg in node.args.args]
                    }
                    functions.append(func_info)
            
            if functions:
                outline.append("## Functions")
                for func in sorted(functions, key=lambda x: x['line']):
                    args_str = ', '.join(func['args'])
                    outline.append(f"### {func['name']}({args_str}) - Line {func['line']}")
                    if func['docstring']:
                        outline.append(f"> {func['docstring'].split('.')[0]}.")
                outline.append("")
            
            # Process global variables
            variables = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign) and all(isinstance(target, ast.Name) for target in node.targets):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            var_info = {
                                'name': target.id,
                                'line': node.lineno
                            }
                            variables.append(var_info)
            
            if variables:
                outline.append("## Global Variables")
                for var in sorted(variables, key=lambda x: x['line']):
                    outline.append(f"- **{var['name']}** - Line {var['line']}")
            
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
    
    # Test different functions
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