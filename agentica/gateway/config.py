# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Configuration management for the gateway service.

Gateway-specific settings only. Model/path/workspace config is in agentica/config.py.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

load_dotenv()

from agentica.config import AGENTICA_WORKSPACE_DIR


@dataclass
class Settings:
    """Gateway service configuration.

    All fields are mutable at runtime so that routes (e.g. /api/model,
    /api/config/base_dir) can update them without property-override hacks.
    """

    # Server
    host: str = "0.0.0.0"
    port: int = 8789
    debug: bool = False
    gateway_token: Optional[str] = None

    # Default user ID (single-user scenario)
    default_user_id: str = "default"

    # Agent cache: max number of concurrent Agent instances kept in LRU cache
    agent_max_sessions: int = 50

    # File upload limits
    upload_max_size_mb: int = 50
    upload_allowed_extensions: str = (
        ".txt,.md,.py,.js,.ts,.jsx,.tsx,.json,.yaml,.yml,.toml,.csv,"
        ".pdf,.png,.jpg,.jpeg,.gif,.webp,.svg,.zip,.tar,.gz"
    )

    # Run history retention (days); runs older than this are pruned on startup
    job_runs_retention_days: int = 30

    # Feishu
    feishu_app_id: Optional[str] = None
    feishu_app_secret: Optional[str] = None
    feishu_allowed_users: List[str] = field(default_factory=list)
    feishu_allowed_groups: List[str] = field(default_factory=list)

    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_allowed_users: List[str] = field(default_factory=list)

    # Discord
    discord_bot_token: Optional[str] = None
    discord_allowed_users: List[str] = field(default_factory=list)
    discord_allowed_guilds: List[str] = field(default_factory=list)

    # QQ (qq-botpy)
    qq_app_id: Optional[str] = None
    qq_app_secret: Optional[str] = None
    qq_allowed_users: List[str] = field(default_factory=list)

    # WeCom (Enterprise WeChat, wecom_aibot_sdk)
    wecom_bot_id: Optional[str] = None
    wecom_secret: Optional[str] = None
    wecom_allowed_users: List[str] = field(default_factory=list)

    # DingTalk (dingtalk-stream)
    dingtalk_client_id: Optional[str] = None
    dingtalk_client_secret: Optional[str] = None
    dingtalk_allowed_users: List[str] = field(default_factory=list)

    # WeChat (personal, ilinkai inline client)
    wechat_token_file: Optional[str] = None
    wechat_allowed_users: List[str] = field(default_factory=list)

    # Model / path settings — mutable fields with env-var defaults.
    # Routes update these at runtime (e.g. model switch, base_dir change).
    model_provider: str = ""
    model_name: str = ""
    model_thinking: str = ""
    model_reasoning_effort: str = ""

    # Optional sibling models for DeepAgent. None / empty means reuse the
    # main model (same provider, base_url, api_key).
    aux_model_provider: str = ""
    aux_model_name: str = ""
    aux_base_url: str = ""
    aux_api_key: str = ""

    task_model_provider: str = ""
    task_model_name: str = ""
    task_base_url: str = ""
    task_api_key: str = ""

    _base_dir: str = ""

    @property
    def workspace_path(self) -> Path:
        return Path(AGENTICA_WORKSPACE_DIR)

    @property
    def base_dir(self) -> Path:
        return Path(self._base_dir) if self._base_dir else Path(os.getenv("AGENTICA_BASE_DIR", str(Path.home())))

    @base_dir.setter
    def base_dir(self, value):
        """Accept both str and Path for convenience."""
        self._base_dir = str(value)

    @property
    def upload_allowed_ext_set(self) -> set[str]:
        """Return upload_allowed_extensions as a lowercase set for fast lookup."""
        return {
            e.strip().lower()
            for e in self.upload_allowed_extensions.split(",")
            if e.strip()
        }

    @classmethod
    def from_env(cls) -> "Settings":
        """Load configuration from environment variables."""
        allowed_users = os.getenv("FEISHU_ALLOWED_USERS", "")
        allowed_groups = os.getenv("FEISHU_ALLOWED_GROUPS", "")

        return cls(
            # Server
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8789")),
            debug=os.getenv("DEBUG", "").lower() in ("1", "true"),
            gateway_token=os.getenv("GATEWAY_TOKEN") or None,

            # Default user
            default_user_id=os.getenv("DEFAULT_USER_ID", "default"),

            # Agent cache
            agent_max_sessions=int(os.getenv("AGENT_MAX_SESSIONS", "50")),

            # Upload limits
            upload_max_size_mb=int(os.getenv("UPLOAD_MAX_SIZE_MB", "50")),
            upload_allowed_extensions=os.getenv(
                "UPLOAD_ALLOWED_EXTENSIONS",
                ".txt,.md,.py,.js,.ts,.jsx,.tsx,.json,.yaml,.yml,.toml,.csv,"
                ".pdf,.png,.jpg,.jpeg,.gif,.webp,.svg,.zip,.tar,.gz",
            ),

            # Run history retention
            job_runs_retention_days=int(os.getenv("JOB_RUNS_RETENTION_DAYS", "30")),

            # Feishu
            feishu_app_id=os.getenv("FEISHU_APP_ID"),
            feishu_app_secret=os.getenv("FEISHU_APP_SECRET"),
            feishu_allowed_users=[
                u.strip() for u in allowed_users.split(",") if u.strip()
            ],
            feishu_allowed_groups=[
                g.strip() for g in allowed_groups.split(",") if g.strip()
            ],

            # Telegram
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
            telegram_allowed_users=[
                u.strip() for u in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") if u.strip()
            ],

            # Discord
            discord_bot_token=os.getenv("DISCORD_BOT_TOKEN"),
            discord_allowed_users=[
                u.strip() for u in os.getenv("DISCORD_ALLOWED_USERS", "").split(",") if u.strip()
            ],
            discord_allowed_guilds=[
                g.strip() for g in os.getenv("DISCORD_ALLOWED_GUILDS", "").split(",") if g.strip()
            ],

            # QQ
            qq_app_id=os.getenv("QQ_APP_ID"),
            qq_app_secret=os.getenv("QQ_APP_SECRET"),
            qq_allowed_users=[
                u.strip() for u in os.getenv("QQ_ALLOWED_USERS", "").split(",") if u.strip()
            ],

            # WeCom (Enterprise WeChat)
            wecom_bot_id=os.getenv("WECOM_BOT_ID"),
            wecom_secret=os.getenv("WECOM_SECRET"),
            wecom_allowed_users=[
                u.strip() for u in os.getenv("WECOM_ALLOWED_USERS", "").split(",") if u.strip()
            ],

            # DingTalk
            dingtalk_client_id=os.getenv("DINGTALK_CLIENT_ID"),
            dingtalk_client_secret=os.getenv("DINGTALK_CLIENT_SECRET"),
            dingtalk_allowed_users=[
                u.strip() for u in os.getenv("DINGTALK_ALLOWED_USERS", "").split(",") if u.strip()
            ],

            # WeChat (personal)
            wechat_token_file=os.getenv("WECHAT_TOKEN_FILE") or None,
            wechat_allowed_users=[
                u.strip() for u in os.getenv("WECHAT_ALLOWED_USERS", "").split(",") if u.strip()
            ],

            # Model / path
            model_provider=os.getenv("AGENTICA_MODEL_PROVIDER", "deepseek"),
            model_name=os.getenv("AGENTICA_MODEL_NAME", "deepseek-v4-flash"),
            model_thinking=os.getenv("AGENTICA_MODEL_THINKING", ""),
            model_reasoning_effort=os.getenv("AGENTICA_REASONING_EFFORT", ""),

            # Auxiliary model (leave empty to reuse main model)
            aux_model_provider=os.getenv("AGENTICA_AUX_MODEL_PROVIDER", ""),
            aux_model_name=os.getenv("AGENTICA_AUX_MODEL_NAME", ""),
            aux_base_url=os.getenv("AGENTICA_AUX_BASE_URL", ""),
            aux_api_key=os.getenv("AGENTICA_AUX_API_KEY", ""),

            # Task-subagent model (leave empty to reuse main model)
            task_model_provider=os.getenv("AGENTICA_TASK_MODEL_PROVIDER", ""),
            task_model_name=os.getenv("AGENTICA_TASK_MODEL_NAME", ""),
            task_base_url=os.getenv("AGENTICA_TASK_BASE_URL", ""),
            task_api_key=os.getenv("AGENTICA_TASK_API_KEY", ""),
        )


# Global settings instance
settings = Settings.from_env()
