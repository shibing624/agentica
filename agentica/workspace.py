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
        users_dir: 用户数据目录名（用于多用户隔离）
    """
    agent_md: str = "AGENT.md"
    persona_md: str = "PERSONA.md"
    tools_md: str = "TOOLS.md"
    user_md: str = "USER.md"
    memory_md: str = "MEMORY.md"
    memory_dir: str = "memory"
    skills_dir: str = "skills"
    users_dir: str = "users"


class Workspace:
    """Agent 工作空间

    工作空间是 Agent 的配置和记忆存储目录，支持多用户隔离。

    目录结构：
    - AGENT.md: Agent 操作指南和约束（全局共享）
    - PERSONA.md: Agent 人格设定（全局共享）
    - TOOLS.md: 工具使用说明（全局共享）
    - USER.md: 默认用户信息（无 user_id 时使用）
    - MEMORY.md: 默认长期记忆（无 user_id 时使用）
    - memory/: 默认日记忆目录（无 user_id 时使用）
    - skills/: 自定义技能目录（全局共享）
    - users/: 用户数据目录（多用户隔离）
        - {user_id}/
            - USER.md: 用户信息
            - MEMORY.md: 长期记忆
            - memory/: 日记忆目录

    单用户模式（不指定 user_id）：
        >>> workspace = Workspace("~/.agentica/workspace")
        >>> workspace.initialize()
        >>> workspace.save_memory("User prefers concise responses")

    多用户模式（指定 user_id）：
        >>> workspace = Workspace("~/.agentica/workspace", user_id="alice@example.com")
        >>> workspace.initialize()
        >>> workspace.save_memory("Alice likes Python")  # 存储到 users/alice@example.com/

    切换用户：
        >>> workspace.set_user("bob@example.com")
        >>> workspace.save_memory("Bob prefers detailed explanations")
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
        config: Optional[WorkspaceConfig] = None,
        user_id: Optional[str] = None,
    ):
        """初始化工作空间

        Args:
            path: 工作空间路径，默认使用 AGENTICA_WORKSPACE_DIR (~/.agentica/workspace)
            config: 工作空间配置，默认使用 WorkspaceConfig 默认值
            user_id: 用户 ID，用于多用户隔离。不指定则使用默认用户目录
        """
        if path is None:
            path = DEFAULT_WORKSPACE_PATH
        self.path = Path(path).expanduser().resolve()
        self.config = config or WorkspaceConfig()
        self._user_id = user_id

    @property
    def user_id(self) -> Optional[str]:
        """获取当前用户 ID"""
        return self._user_id

    def set_user(self, user_id: Optional[str]):
        """设置当前用户 ID

        Args:
            user_id: 用户 ID，设为 None 则使用默认用户
        """
        self._user_id = user_id

    def _get_user_path(self) -> Path:
        """获取当前用户的数据目录路径

        Returns:
            如果指定了 user_id，返回 users/{user_id}/ 目录路径
            否则返回工作空间根目录
        """
        if self._user_id:
            # 对 user_id 进行安全处理，替换不安全字符
            safe_user_id = self._user_id.replace("/", "_").replace("\\", "_").replace("..", "_")
            return self.path / self.config.users_dir / safe_user_id
        return self.path

    def _get_user_memory_dir(self) -> Path:
        """获取当前用户的日记忆目录"""
        return self._get_user_path() / self.config.memory_dir

    def _get_user_memory_md(self) -> Path:
        """获取当前用户的长期记忆文件路径"""
        return self._get_user_path() / self.config.memory_md

    def _get_user_md(self) -> Path:
        """获取当前用户的 USER.md 文件路径"""
        return self._get_user_path() / self.config.user_md

    def initialize(self, force: bool = False) -> bool:
        """初始化工作空间

        创建工作空间目录和默认配置文件。
        如果指定了 user_id，会同时创建用户数据目录。

        Args:
            force: 是否覆盖已存在的文件

        Returns:
            是否成功初始化
        """
        self.path.mkdir(parents=True, exist_ok=True)

        # 创建全局共享的默认文件
        for filename, content in self.DEFAULT_FILES.items():
            filepath = self.path / filename
            if not filepath.exists() or force:
                filepath.write_text(content, encoding="utf-8")

        # 创建全局目录
        (self.path / self.config.memory_dir).mkdir(exist_ok=True)
        (self.path / self.config.skills_dir).mkdir(exist_ok=True)
        (self.path / self.config.users_dir).mkdir(exist_ok=True)

        # 如果指定了 user_id，创建用户目录
        if self._user_id:
            self._initialize_user_dir()

        return True

    def _initialize_user_dir(self):
        """初始化当前用户的数据目录"""
        user_path = self._get_user_path()
        user_path.mkdir(parents=True, exist_ok=True)

        # 创建用户的 USER.md
        user_md = user_path / self.config.user_md
        if not user_md.exists():
            user_md.write_text(f"""# User Profile

## User ID
{self._user_id}

## Preferences
- Language: 中文
- Style: Concise

## Context
<!-- User's background, projects, etc. -->
""", encoding="utf-8")

        # 创建用户的记忆目录
        (user_path / self.config.memory_dir).mkdir(exist_ok=True)

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

        读取 AGENT.md, PERSONA.md, TOOLS.md 文件内容（全局共享），
        以及用户特定的 USER.md 文件内容。

        Returns:
            合并后的上下文字符串
        """
        contents = []

        # 读取全局共享文件
        global_files = [
            self.config.agent_md,
            self.config.persona_md,
            self.config.tools_md,
        ]
        for f in global_files:
            content = self.read_file(f)
            if content:
                contents.append(f"<!-- {f} -->\n{content}")

        # 读取用户特定的 USER.md
        user_md_path = self._get_user_md()
        if user_md_path.exists():
            content = user_md_path.read_text(encoding="utf-8").strip()
            if content:
                user_label = f"USER.md (user: {self._user_id})" if self._user_id else "USER.md"
                contents.append(f"<!-- {user_label} -->\n{content}")
        else:
            # 回退到全局 USER.md
            content = self.read_file(self.config.user_md)
            if content:
                contents.append(f"<!-- {self.config.user_md} -->\n{content}")

        return "\n\n---\n\n".join(contents) if contents else ""

    def get_memory_prompt(self, days: int = 2) -> str:
        """获取最近记忆（用于注入上下文）

        读取用户特定的 MEMORY.md 长期记忆和最近几天的日记忆。
        如果未指定 user_id，则读取默认记忆目录。

        Args:
            days: 读取最近几天的记忆

        Returns:
            记忆内容字符串
        """
        contents = []

        # 读取长期记忆（用户特定）
        long_term_path = self._get_user_memory_md()
        if long_term_path.exists():
            long_term = long_term_path.read_text(encoding="utf-8").strip()
            if long_term:
                user_label = f" (user: {self._user_id})" if self._user_id else ""
                contents.append(f"## Long-term Memory{user_label}\n\n{long_term}")

        # 读取日记忆（用户特定）
        memory_dir = self._get_user_memory_dir()
        if memory_dir.exists():
            files = sorted(memory_dir.glob("*.md"), reverse=True)[:days]
            for f in files:
                content = f.read_text(encoding="utf-8").strip()
                if content:
                    contents.append(f"## {f.stem}\n\n{content}")

        return "\n\n".join(contents) if contents else ""

    def write_memory(self, content: str, to_daily: bool = True):
        """写入记忆

        将内容写入当前用户的记忆文件。
        如果未指定 user_id，则写入默认记忆目录。

        Args:
            content: 记忆内容
            to_daily: True 写入日记忆，False 写入长期记忆
        """
        # 确保用户目录存在
        if self._user_id:
            self._initialize_user_dir()

        if to_daily:
            today = date.today().isoformat()
            memory_dir = self._get_user_memory_dir()
            memory_dir.mkdir(parents=True, exist_ok=True)
            filepath = memory_dir / f"{today}.md"

            # 追加内容
            existing = ""
            if filepath.exists():
                existing = filepath.read_text(encoding="utf-8").strip()
            new_content = f"{existing}\n\n{content}".strip() if existing else content
            filepath.write_text(new_content, encoding="utf-8")
        else:
            memory_md = self._get_user_memory_md()
            memory_md.parent.mkdir(parents=True, exist_ok=True)

            # 追加内容
            existing = ""
            if memory_md.exists():
                existing = memory_md.read_text(encoding="utf-8").strip()
            new_content = f"{existing}\n\n{content}".strip() if existing else content
            memory_md.write_text(new_content, encoding="utf-8")

    def save_memory(self, content: str, long_term: bool = False):
        """保存记忆（write_memory 的别名，更符合语义）

        Args:
            content: 记忆内容
            long_term: True 写入长期记忆，False 写入日记忆（默认）
        """
        self.write_memory(content, to_daily=not long_term)

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
        """获取当前用户的所有记忆文件路径

        Returns:
            所有记忆文件的路径列表
        """
        files = []

        # 长期记忆
        memory_md = self._get_user_memory_md()
        if memory_md.exists():
            files.append(memory_md)

        # 日记忆
        memory_dir = self._get_user_memory_dir()
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

        清理当前用户的旧日记忆文件。

        Args:
            keep_days: 保留最近几天的记忆
        """
        memory_dir = self._get_user_memory_dir()
        if not memory_dir.exists():
            return

        files = sorted(memory_dir.glob("*.md"), reverse=True)
        for f in files[keep_days:]:
            f.unlink()

    def create_memory_search(self):
        """创建工作空间记忆搜索实例

        Returns:
            WorkspaceMemorySearch 实例，可用于向量搜索和关键词搜索

        Example:
            >>> workspace = Workspace()
            >>> search = workspace.create_memory_search()
            >>> search.index()
            >>> results = search.search("Python programming")
        """
        from agentica.memory import WorkspaceMemorySearch
        return WorkspaceMemorySearch(workspace_path=str(self.path))

    def search_memory_hybrid(
        self,
        query: str,
        limit: int = 5,
        embedder=None,
    ) -> List[Dict]:
        """混合搜索记忆（向量 + 关键词）

        使用向量相似度和关键词匹配的组合来搜索记忆。

        Args:
            query: 搜索查询
            limit: 返回数量限制
            embedder: 可选的嵌入模型实例，如果未提供会尝试使用 OpenAIEmb

        Returns:
            匹配的记忆列表

        Example:
            >>> workspace = Workspace()
            >>> workspace.initialize()
            >>> workspace.write_memory("Python is great for AI")
            >>> results = workspace.search_memory_hybrid("artificial intelligence")
        """
        from agentica.memory import WorkspaceMemorySearch

        search = WorkspaceMemorySearch(workspace_path=str(self.path))
        search.index()

        results = search.search_hybrid(query, limit=limit, embedder=embedder)

        # Convert MemoryChunk to dict for consistency with search_memory
        return [
            {
                "content": r.content,
                "file_path": r.file_path,
                "score": r.score,
            }
            for r in results
        ]

    def __repr__(self) -> str:
        user_info = f", user_id={self._user_id}" if self._user_id else ""
        return f"Workspace(path={self.path}, exists={self.exists()}{user_info})"

    def __str__(self) -> str:
        return str(self.path)

    def list_users(self) -> List[str]:
        """列出所有已注册的用户 ID

        Returns:
            用户 ID 列表
        """
        users_dir = self.path / self.config.users_dir
        if not users_dir.exists():
            return []

        users = []
        for user_dir in users_dir.iterdir():
            if user_dir.is_dir():
                users.append(user_dir.name)
        return sorted(users)

    def get_user_info(self, user_id: Optional[str] = None) -> Dict:
        """获取用户信息摘要

        Args:
            user_id: 用户 ID，如果不指定则使用当前用户

        Returns:
            用户信息字典，包含 user_id, memory_count, last_activity 等
        """
        target_user = user_id or self._user_id
        old_user = self._user_id

        try:
            if target_user:
                self._user_id = target_user

            memory_files = self.get_all_memory_files()
            memory_count = len(memory_files)

            last_activity = None
            if memory_files:
                # 获取最新记忆文件的修改时间
                latest_file = memory_files[0]
                if latest_file.exists():
                    import datetime
                    mtime = latest_file.stat().st_mtime
                    last_activity = datetime.datetime.fromtimestamp(mtime).isoformat()

            return {
                "user_id": target_user or "default",
                "memory_count": memory_count,
                "last_activity": last_activity,
                "user_path": str(self._get_user_path()),
            }
        finally:
            self._user_id = old_user

    def delete_user(self, user_id: str, confirm: bool = False) -> bool:
        """删除用户数据

        Args:
            user_id: 要删除的用户 ID
            confirm: 必须设为 True 才能执行删除

        Returns:
            是否成功删除
        """
        if not confirm:
            raise ValueError("Must set confirm=True to delete user data")

        if not user_id:
            raise ValueError("user_id cannot be empty")

        safe_user_id = user_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        user_path = self.path / self.config.users_dir / safe_user_id

        if not user_path.exists():
            return False

        import shutil
        shutil.rmtree(user_path)
        return True
