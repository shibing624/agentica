# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Workspace management for Agentica agents.
Inspired by OpenClaw's workspace concept.
"""
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import date

from agentica.config import AGENTICA_WORKSPACE_DIR

# Default workspace path from config
DEFAULT_WORKSPACE_PATH = AGENTICA_WORKSPACE_DIR


@dataclass
class WorkspaceConfig:
    """工作空间配置

    Attributes:
        agent_md: Agent 操作指南文件名
        persona_md: 人格设定文件名
        tools_md: 工具说明文件名
        user_md: 用户信息文件名
        memory_md: 长期记忆文件名
        memory_dir: 日记忆目录名
        skills_dir: 技能目录名
    """
    agent_md: str = "AGENT.md"
    persona_md: str = "PERSONA.md"
    tools_md: str = "TOOLS.md"
    user_md: str = "USER.md"
    memory_md: str = "MEMORY.md"
    memory_dir: str = "memory"
    skills_dir: str = "skills"


class Workspace:
    """Agent 工作空间

    工作空间是 Agent 的配置和记忆存储目录，包含：
    - AGENT.md: Agent 操作指南和约束
    - PERSONA.md: Agent 人格设定
    - TOOLS.md: 工具使用说明
    - USER.md: 用户信息
    - MEMORY.md: 长期记忆
    - memory/: 日记忆目录
    - skills/: 自定义技能目录

    Example:
        >>> workspace = Workspace("~/.agentica/workspace")
        >>> workspace.initialize()
        >>> context = workspace.get_context_prompt()
        >>> memory = workspace.get_memory_prompt(days=2)
    """

    DEFAULT_FILES = {
        "AGENT.md": """# Agent Instructions

You are a helpful AI assistant.

## Guidelines
1. Be concise and accurate
2. Use tools when needed
3. Store important information in memory
4. Follow user preferences in USER.md
""",
        "PERSONA.md": """# Persona

## Personality
- Friendly and professional
- Direct and honest
- Proactive in helping

## Communication Style
- Clear and concise
- Use examples when explaining
- Ask clarifying questions when needed
""",
        "TOOLS.md": """# Tool Usage Guidelines

## File Operations
- Always use absolute paths
- Read files before editing
- Create backups for important changes

## Shell Commands
- Prefer safe, non-destructive commands
- Explain what commands will do
""",
        "USER.md": """# User Profile

<!-- Add user preferences and information here -->

## Preferences
- Language: 中文
- Style: Concise

## Context
<!-- User's background, projects, etc. -->
""",
    }

    def __init__(
        self,
        path: Optional[str | Path] = None,
        config: Optional[WorkspaceConfig] = None
    ):
        """初始化工作空间

        Args:
            path: 工作空间路径，默认使用 AGENTICA_WORKSPACE_DIR (~/.agentica/workspace)
            config: 工作空间配置，默认使用 WorkspaceConfig 默认值
        """
        if path is None:
            path = DEFAULT_WORKSPACE_PATH
        self.path = Path(path).expanduser().resolve()
        self.config = config or WorkspaceConfig()

    def initialize(self, force: bool = False) -> bool:
        """初始化工作空间

        创建工作空间目录和默认配置文件。

        Args:
            force: 是否覆盖已存在的文件

        Returns:
            是否成功初始化
        """
        self.path.mkdir(parents=True, exist_ok=True)

        # 创建默认文件
        for filename, content in self.DEFAULT_FILES.items():
            filepath = self.path / filename
            if not filepath.exists() or force:
                filepath.write_text(content, encoding="utf-8")

        # 创建目录
        (self.path / self.config.memory_dir).mkdir(exist_ok=True)
        (self.path / self.config.skills_dir).mkdir(exist_ok=True)

        return True

    def exists(self) -> bool:
        """检查工作空间是否存在

        Returns:
            工作空间目录和 AGENT.md 文件是否都存在
        """
        return self.path.exists() and (self.path / self.config.agent_md).exists()

    def read_file(self, filename: str) -> Optional[str]:
        """读取工作空间文件

        Args:
            filename: 文件名（相对于工作空间路径）

        Returns:
            文件内容，如果文件不存在或为空则返回 None
        """
        filepath = self.path / filename
        if filepath.exists() and filepath.is_file():
            content = filepath.read_text(encoding="utf-8").strip()
            return content if content else None
        return None

    def write_file(self, filename: str, content: str):
        """写入工作空间文件

        Args:
            filename: 文件名（相对于工作空间路径）
            content: 要写入的内容
        """
        filepath = self.path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")

    def append_file(self, filename: str, content: str):
        """追加内容到工作空间文件

        Args:
            filename: 文件名（相对于工作空间路径）
            content: 要追加的内容
        """
        filepath = self.path / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if filepath.exists():
            existing = filepath.read_text(encoding="utf-8").strip()

        new_content = f"{existing}\n\n{content}".strip() if existing else content
        filepath.write_text(new_content, encoding="utf-8")

    def get_context_prompt(self) -> str:
        """获取工作空间上下文（用于注入 System Prompt）

        读取 AGENT.md, PERSONA.md, TOOLS.md, USER.md 文件内容，
        合并为一个上下文字符串。

        Returns:
            合并后的上下文字符串
        """
        files = [
            self.config.agent_md,
            self.config.persona_md,
            self.config.tools_md,
            self.config.user_md,
        ]

        contents = []
        for f in files:
            content = self.read_file(f)
            if content:
                contents.append(f"<!-- {f} -->\n{content}")

        return "\n\n---\n\n".join(contents) if contents else ""

    def get_memory_prompt(self, days: int = 2) -> str:
        """获取最近记忆（用于注入上下文）

        读取 MEMORY.md 长期记忆和最近几天的日记忆。

        Args:
            days: 读取最近几天的记忆

        Returns:
            记忆内容字符串
        """
        contents = []

        # 读取 MEMORY.md
        long_term = self.read_file(self.config.memory_md)
        if long_term:
            contents.append(f"## Long-term Memory\n\n{long_term}")

        # 读取日记忆
        memory_dir = self.path / self.config.memory_dir
        if memory_dir.exists():
            files = sorted(memory_dir.glob("*.md"), reverse=True)[:days]
            for f in files:
                content = f.read_text(encoding="utf-8").strip()
                if content:
                    contents.append(f"## {f.stem}\n\n{content}")

        return "\n\n".join(contents) if contents else ""

    def write_memory(self, content: str, to_daily: bool = True):
        """写入记忆

        Args:
            content: 记忆内容
            to_daily: True 写入日记忆，False 写入长期记忆
        """
        if to_daily:
            today = date.today().isoformat()
            filepath = self.path / self.config.memory_dir / f"{today}.md"
            self.append_file(str(filepath.relative_to(self.path)), content)
        else:
            self.append_file(self.config.memory_md, content)

    def get_skills_dir(self) -> Path:
        """获取技能目录路径

        Returns:
            技能目录的绝对路径
        """
        return self.path / self.config.skills_dir

    def list_files(self) -> Dict[str, bool]:
        """列出工作空间文件状态

        Returns:
            字典，键为文件名，值为文件是否存在
        """
        files = [
            self.config.agent_md,
            self.config.persona_md,
            self.config.tools_md,
            self.config.user_md,
            self.config.memory_md,
        ]
        return {f: (self.path / f).exists() for f in files}

    def get_all_memory_files(self) -> List[Path]:
        """获取所有记忆文件路径

        Returns:
            所有记忆文件的路径列表
        """
        files = []

        # 长期记忆
        memory_md = self.path / self.config.memory_md
        if memory_md.exists():
            files.append(memory_md)

        # 日记忆
        memory_dir = self.path / self.config.memory_dir
        if memory_dir.exists():
            files.extend(sorted(memory_dir.glob("*.md"), reverse=True))

        return files

    def search_memory(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict]:
        """搜索记忆（简单关键词搜索）

        Args:
            query: 搜索查询
            limit: 返回数量限制
            min_score: 最小匹配分数

        Returns:
            匹配的记忆列表，每项包含 content, file_path, score
        """
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        for file_path in self.get_all_memory_files():
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            content_lower = content.lower()

            # 计算简单匹配分数
            score = 0.0
            for word in query_words:
                if word in content_lower:
                    score += 1.0 / len(query_words)

            if score >= min_score:
                results.append({
                    "content": content,
                    "file_path": str(file_path.relative_to(self.path)),
                    "score": score,
                })

        # 按分数排序
        results.sort(key=lambda x: -x["score"])
        return results[:limit]

    def clear_daily_memory(self, keep_days: int = 7):
        """清理旧的日记忆

        Args:
            keep_days: 保留最近几天的记忆
        """
        memory_dir = self.path / self.config.memory_dir
        if not memory_dir.exists():
            return

        files = sorted(memory_dir.glob("*.md"), reverse=True)
        for f in files[keep_days:]:
            f.unlink()

    def __repr__(self) -> str:
        return f"Workspace(path={self.path}, exists={self.exists()})"

    def __str__(self) -> str:
        return str(self.path)
