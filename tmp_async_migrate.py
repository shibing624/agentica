#!/usr/bin/env python3
"""Async-first migration script for model files.

Transforms sync model methods to async by:
1. Adding `async` keyword to method signatures
2. Adding `await` to self.invoke() and self.response() calls
3. Converting `for _ in self.run_function_calls(` to `async for`
4. Converting `yield from self.response_stream(` to `async for ... yield`
5. Converting `yield from self.handle_stream_tool_calls(` to `async for ... yield`
6. Converting `yield from self.handle_post_tool_call_messages_stream(` to `async for ... yield`
7. Converting `for response in self.invoke_stream(` to `async for`
8. Converting Iterator to AsyncIterator in type hints
9. Adding asyncio + functools imports
"""
import re
import sys
from pathlib import Path

BASE = Path("/Users/xuming/Documents/Codes/agentica-async-first/agentica/model")

# Files that need migration (not openai/chat.py, groq, litellm, ollama/chat, ollama/hermes, hf - already done)
FILES_TO_MIGRATE = [
    BASE / "google" / "gemini.py",
    BASE / "anthropic" / "claude.py",
    BASE / "aws" / "bedrock.py",
    BASE / "cohere" / "chat.py",
    BASE / "vertexai" / "gemini.py",
    BASE / "ollama" / "tools.py",
    BASE / "together" / "together.py",
]


def migrate_file(filepath: Path):
    content = filepath.read_text()
    original = content
    
    # 1. Add imports if needed
    if "import asyncio" not in content:
        # Add after the first import block
        content = content.replace(
            "from agentica.model.base import Model",
            "import asyncio\nimport functools\n\nfrom agentica.model.base import Model",
            1
        )
    if "import functools" not in content and "functools" not in content:
        content = content.replace(
            "import asyncio\n\nfrom agentica.model.base import Model",
            "import asyncio\nimport functools\n\nfrom agentica.model.base import Model",
            1
        )
    
    # For ollama/tools.py which imports from ollama/chat.py (already async)
    if "from agentica.model.ollama.chat import Ollama" in content:
        if "import asyncio" not in content:
            content = "import asyncio\nimport functools\n\n" + content
    
    # For together.py which imports from openai/like.py  
    if "from agentica.model.openai.like import OpenAILike" in content:
        if "import asyncio" not in content:
            content = "import asyncio\nimport functools\n\n" + content
    
    # 2. Replace Iterator with AsyncIterator in imports
    content = content.replace(
        "from typing import Optional, List, Iterator,",
        "from typing import Optional, List, AsyncIterator,"
    )
    # Handle various import formats
    content = re.sub(
        r'(from typing import[^)]*)\bIterator\b',
        lambda m: m.group(0).replace('Iterator', 'AsyncIterator') if 'AsyncIterator' not in m.group(0) else m.group(0),
        content
    )
    
    # 3. Convert sync def to async def for key methods
    # invoke
    content = re.sub(
        r'(\s+)def invoke\(self,',
        r'\1async def invoke(self,',
        content
    )
    # invoke_stream 
    content = re.sub(
        r'(\s+)def invoke_stream\(self,',
        r'\1async def invoke_stream(self,',
        content
    )
    # response (but not response_stream or response_content)
    content = re.sub(
        r'(\s+)def response\(self,',
        r'\1async def response(self,',
        content
    )
    # response_stream
    content = re.sub(
        r'(\s+)def response_stream\(self,',
        r'\1async def response_stream(self,',
        content
    )
    # handle_tool_calls
    content = re.sub(
        r'(\s+)def handle_tool_calls\(self,',
        r'\1async def handle_tool_calls(self,',
        content
    )
    # _handle_tool_calls
    content = re.sub(
        r'(\s+)def _handle_tool_calls\(self,',
        r'\1async def _handle_tool_calls(self,',
        content
    )
    # handle_stream_tool_calls
    content = re.sub(
        r'(\s+)def handle_stream_tool_calls\(self,',
        r'\1async def handle_stream_tool_calls(self,',
        content
    )
    # _handle_stream_tool_calls
    content = re.sub(
        r'(\s+)def _handle_stream_tool_calls\(self,',
        r'\1async def _handle_stream_tool_calls(self,',
        content
    )
    
    # 4. Convert sync SDK calls in invoke to run_in_executor
    # For invoke methods that call self.get_client() or self.client synchronously
    # This needs to be handled per-file, but we can do common patterns
    
    # 5. Add await to self.invoke() calls inside response()
    content = re.sub(
        r'(\s+)(response\s*(?::\s*\w+)?\s*=\s*)self\.invoke\(',
        r'\1\2await self.invoke(',
        content
    )
    
    # Add await to self.invoke_stream() calls
    # Pattern: for response in self.invoke_stream(...)
    content = re.sub(
        r'(\s+)for (\w+) in self\.invoke_stream\(',
        r'\1async for \2 in self.invoke_stream(',
        content
    )
    
    # 6. Convert `for _ in self.run_function_calls(` to `async for`
    content = re.sub(
        r'(\s+)for (\w+) in self\.run_function_calls\(',
        r'\1async for \2 in self.run_function_calls(',
        content
    )
    
    # 7. Convert `yield from self.response_stream(` to `async for resp in self.response_stream(...): yield resp`
    content = re.sub(
        r'(\s+)yield from self\.response_stream\((.*?)\)',
        r'\1async for _resp in self.response_stream(\2):\n\1    yield _resp',
        content
    )
    
    # Convert `yield from self.handle_stream_tool_calls(`
    content = re.sub(
        r'(\s+)yield from self\.handle_stream_tool_calls\((.*?)\)',
        r'\1async for _resp in self.handle_stream_tool_calls(\2):\n\1    yield _resp',
        content
    )
    
    # Convert `yield from self._handle_stream_tool_calls(`
    content = re.sub(
        r'(\s+)yield from self\._handle_stream_tool_calls\((.*?)\)',
        r'\1async for _resp in self._handle_stream_tool_calls(\2):\n\1    yield _resp',
        content
    )
    
    # Convert `yield from self.handle_post_tool_call_messages_stream(`
    content = re.sub(
        r'(\s+)yield from self\.handle_post_tool_call_messages_stream\((.*?)\)',
        r'\1async for _resp in self.handle_post_tool_call_messages_stream(\2):\n\1    yield _resp',
        content
    )
    
    # 8. Add await to recursive self.response() calls
    content = re.sub(
        r'(\s+)(response_after_tool_calls\s*=\s*)self\.response\(',
        r'\1\2await self.response(',
        content
    )
    
    # Add await to self.handle_tool_calls() calls
    content = re.sub(
        r'if self\.handle_tool_calls\(',
        'if await self.handle_tool_calls(',
        content
    )
    content = re.sub(
        r'if self\._handle_tool_calls\(',
        'if await self._handle_tool_calls(',
        content
    )
    
    # 9. Replace Iterator[ModelResponse] return types with AsyncIterator[ModelResponse]
    content = content.replace(
        ") -> Iterator[ModelResponse]:",
        ") -> AsyncIterator[ModelResponse]:"
    )
    content = content.replace(
        "Iterator[ModelResponse]:",
        "AsyncIterator[ModelResponse]:"
    )
    
    # 10. For invoke methods using sync SDK - wrap with run_in_executor
    # Gemini: self.get_client().generate_content(...)
    content = re.sub(
        r'(\s+)return self\.get_client\(\)\.generate_content\(contents=self\.format_messages\(messages\)\)',
        r'''\1loop = asyncio.get_running_loop()
\1return await loop.run_in_executor(
\1    None, functools.partial(self.get_client().generate_content, contents=self.format_messages(messages))
\1)''',
        content
    )
    
    # Gemini stream: yield from self.get_client().generate_content(..., stream=True)
    content = re.sub(
        r'(\s+)yield from self\.get_client\(\)\.generate_content\(\s*\n\s*contents=self\.format_messages\(messages\),\s*\n\s*stream=True,\s*\n\s*\)',
        r'''\1loop = asyncio.get_running_loop()
\1result = await loop.run_in_executor(
\1    None, functools.partial(
\1        self.get_client().generate_content,
\1        contents=self.format_messages(messages), stream=True,
\1    )
\1)
\1for item in result:
\1    yield item''',
        content
    )
    
    # Anthropic: self.get_client().messages.create(...)
    content = re.sub(
        r'(\s+)return self\.get_client\(\)\.messages\.create\(\s*\n\s*model=self\.id,\s*\n\s*messages=chat_messages,.*?\n\s*\*\*request_kwargs,\s*\n\s*\)',
        r'''\1loop = asyncio.get_running_loop()
\1return await loop.run_in_executor(
\1    None, functools.partial(
\1        self.get_client().messages.create,
\1        model=self.id, messages=chat_messages, **request_kwargs,
\1    )
\1)''',
        content,
        flags=re.DOTALL
    )
    
    # Anthropic stream: self.get_client().messages.stream(...)
    content = re.sub(
        r'(\s+)return self\.get_client\(\)\.messages\.stream\(\s*\n\s*model=self\.id,\s*\n\s*messages=chat_messages,.*?\n\s*\*\*request_kwargs,\s*\n\s*\)',
        r'''\1loop = asyncio.get_running_loop()
\1return await loop.run_in_executor(
\1    None, functools.partial(
\1        self.get_client().messages.stream,
\1        model=self.id, messages=chat_messages, **request_kwargs,
\1    )
\1)''',
        content,
        flags=re.DOTALL
    )
    
    # Bedrock: self.bedrock_runtime_client.converse(**body)
    content = re.sub(
        r'(\s+)return self\.bedrock_runtime_client\.converse\(\*\*body\)',
        r'''\1loop = asyncio.get_running_loop()
\1return await loop.run_in_executor(
\1    None, functools.partial(self.bedrock_runtime_client.converse, **body)
\1)''',
        content
    )
    
    # Cohere: self.client.chat(...)
    content = re.sub(
        r'(\s+)return self\.client\.chat\(message=',
        r'''\1loop = asyncio.get_running_loop()
\1return await loop.run_in_executor(
\1    None, functools.partial(self.client.chat, message=''',
        content
    )
    # Fix the closing of the cohere chat call
    content = re.sub(
        r'(functools\.partial\(self\.client\.chat, message=chat_message or "", model=self\.id, \*\*api_kwargs)\)',
        r'\1)\n        )',
        content
    )
    
    # Cohere stream: self.client.chat_stream(...)
    content = re.sub(
        r'(\s+)return self\.client\.chat_stream\(message=',
        r'''\1loop = asyncio.get_running_loop()
\1return await loop.run_in_executor(
\1    None, functools.partial(self.client.chat_stream, message=''',
        content
    )
    content = re.sub(
        r'(functools\.partial\(self\.client\.chat_stream, message=chat_message or "", model=self\.id, \*\*api_kwargs)\)',
        r'\1)\n        )',
        content
    )
    
    # Bedrock stream invoke
    content = re.sub(
        r'(\s+)(response = self\.bedrock_runtime_client\.converse_stream\(\*\*body\))',
        r'''\1loop = asyncio.get_running_loop()
\1response = await loop.run_in_executor(
\1    None, functools.partial(self.bedrock_runtime_client.converse_stream, **body)
\1)''',
        content
    )
    
    # Mistral client calls (already handled in manual edit, but for safety)
    content = re.sub(
        r'(\s+)(response = self\.client\.chat\.complete\()',
        r'\1loop = asyncio.get_running_loop()\n\1response = await loop.run_in_executor(\n\1    None, functools.partial(self.client.chat.complete,',
        content
    )
    
    # 11. Fix `for response in self.invoke_stream(messages=messages):` where invoke_stream is now async
    # The `async for` was already handled above, but we need to handle the Anthropic streaming pattern
    # which uses `with response as stream:` - this is sync context manager from sync SDK
    # For Anthropic: `response = self.invoke_stream(...)` then `with response as stream:`
    # Since invoke_stream now returns via run_in_executor, the result is still a sync context manager
    # We need to add await to the invoke_stream call
    content = re.sub(
        r'(\s+)(response = )self\.invoke_stream\(messages=messages\)',
        r'\1\2await self.invoke_stream(messages=messages)',
        content
    )
    
    # 12. For together.py - `yield from super().response_stream(messages)`
    content = re.sub(
        r'(\s+)yield from super\(\)\.response_stream\(messages\)',
        r'''\1async for _resp in super().response_stream(messages):
\1    yield _resp''',
        content
    )
    
    # For together.py: for response in self.invoke_stream(messages=messages):
    # Already handled by step 5

    # Write back
    if content != original:
        filepath.write_text(content)
        print(f"Migrated: {filepath}")
    else:
        print(f"No changes: {filepath}")


if __name__ == "__main__":
    for f in FILES_TO_MIGRATE:
        if f.exists():
            migrate_file(f)
        else:
            print(f"NOT FOUND: {f}")
