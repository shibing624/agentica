---
name: python-lib-analyzer
description: Analyze any Python library structure, explore modules, classes, and functions with signatures and documentation.
trigger: /pylib
requires:
  - python
allowed-tools:
  - shell
  - python
metadata:
  emoji: "🐍"
  version: "1.0"
---

# Python Library Analyzer

A skill for exploring and analyzing any Python library's structure, including modules, classes, functions, and their documentation.

## When to Use

Use this skill when you need to:
- Explore an unfamiliar Python library's structure
- Find available classes and functions in a module
- Get function/method signatures and documentation
- Understand library architecture before using it

## Quick Start

### 1. View Top-Level Modules

```bash
python view_python_module.py --module <library_name>
```

Example:
```bash
python view_python_module.py --module requests
python view_python_module.py --module numpy
python view_python_module.py --module pandas
```

### 2. Explore Submodules

```bash
python view_python_module.py --module <library>.<submodule>
```

Example:
```bash
python view_python_module.py --module requests.api
python view_python_module.py --module numpy.linalg
python view_python_module.py --module pandas.DataFrame
```

### 3. Get Class/Function Details

```bash
python view_python_module.py --module <library>.<Class>
python view_python_module.py --module <library>.<function>
```

## Script Location

The `view_python_module.py` script is located in the same directory as this SKILL.md file.

## Workflow

1. **Check if library is installed**:
   ```bash
   pip list | grep <library_name>
   ```

2. **Install if needed** (ask user permission first):
   ```bash
   pip install <library_name>
   ```

3. **Start exploration from top level**:
   ```bash
   python view_python_module.py --module <library_name>
   ```

4. **Drill down into specific modules/classes**:
   ```bash
   python view_python_module.py --module <library>.<module>.<Class>
   ```

## Example Session

```bash
# Explore requests library
$ python view_python_module.py --module requests
Available submodules in 'requests':
- requests.api: Convenience functions (get, post, put, delete, etc.)
- requests.models: Request and Response classes
- requests.sessions: Session management
- requests.auth: Authentication handlers
...

# Get details on the get function
$ python view_python_module.py --module requests.get
def get(url, params=None, **kwargs):
    """Sends a GET request.

    Args:
        url: URL for the request.
        params: Dictionary of query parameters.
        **kwargs: Optional arguments for request.

    Returns:
        Response object
    """
```

## Tips

- Start with the library name to see the overall structure
- Use tab completion in shell for module paths
- For large libraries, explore section by section
- Check `__all__` exports for public API
