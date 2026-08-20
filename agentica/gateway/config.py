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

from agentica.config import AGENTICA_WORKSPACE_DIR, AGENTICA_NUM_HISTORY_TURNS
from agentica.global_config import apply_global_config, get_profile


@dataclass
class Settings:
    """Gateway service configuration.

    All fields are mutable at runtime so that routes (e.g. /api/model,
    /api/config/base_dir) can update them without property-override hacks.
    """

    # Server. Loopback by default: this API can run tools as you, so listening
    # on every interface is something to ask for (HOST=0.0.0.0), not something
    # to get by not thinking about it.
    host: str = "127.0.0.1"
    port: int = 8881
    debug: bool = False

    # When set, the gateway exits once this pid is gone — the desktop shell
    # passes its own so killing the shell cannot orphan the server.
    parent_pid: int = 0

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

    # Conversation history window (in turns/runs) fed into the prompt on every
    # request — the single source of truth for _build_agent(), so a rebuilt
    # agent (e.g. after a model/profile switch invalidates the cache) uses the
    # exact same window as the interactive agent it replaced. Defaults to the
    # SDK-wide AGENTICA_NUM_HISTORY_TURNS (same default the CLI/SDK use), so
    # web/CLI/SDK stay aligned unless a user overrides it explicitly.
    num_history_turns: int = AGENTICA_NUM_HISTORY_TURNS

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

    # Slack (Socket Mode)
    slack_bot_token: Optional[str] = None
    slack_app_token: Optional[str] = None
    slack_allowed_users: List[str] = field(default_factory=list)
    slack_allowed_channels: List[str] = field(default_factory=list)

    # Peer bridge: let an IM user address this machine's live CLI sessions
    # (`@list`, `@<session> <text>`). On by default — this is a personal-
    # assistant gateway; the bridge idles until an IM user types `@`, and a
    # channel's `allowed_users` (when set) filters every inbound message
    # before it can reach the bridge. Set PEER_BRIDGE=false to turn it off.
    peer_bridge_enabled: bool = True

    # Model / path settings — loaded from the active config.yaml profile at
    # startup (see Settings.from_env), with env vars as fallback. Routes
    # update these at runtime (profile switch, base_dir change).
    model_provider: str = ""
    model_name: str = ""
    model_base_url: str = ""
    model_api_key: str = ""
    model_wire_api: str = ""
    model_thinking: str = ""
    model_reasoning: str = ""
    model_reasoning_effort: str = ""
    max_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 0.0
    context_window: int = 0

    # Auxiliary model — the cheap/fast model for all background LLM work
    # (memory extraction, context compression, goal judging, skill upgrade)
    # AND the `task` subagent tool. Empty model_name = reuse the main model.
    # This replaces the old separate task_model_* block (CLI unified them).
    auxiliary_model_provider: str = ""
    auxiliary_model_name: str = ""
    auxiliary_base_url: str = ""
    auxiliary_api_key: str = ""
    auxiliary_wire_api: str = ""
    auxiliary_reasoning: str = ""
    auxiliary_reasoning_effort: str = ""

    _base_dir: str = ""

    @property
    def workspace_path(self) -> Path:
        return Path(AGENTICA_WORKSPACE_DIR)

    @property
    def base_dir(self) -> Path:
        # Default project work_dir = the directory `agentica-gateway` was
        # launched from (os.getcwd()), mirroring the CLI. Falls back to that
        # cwd rather than $HOME so the agent operates on the user's current
        # project by default instead of dumping files into the home dir.
        return Path(self._base_dir) if self._base_dir else Path(os.getenv("AGENTICA_BASE_DIR", os.getcwd()))

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

        # Load the active config.yaml profile — single source of truth for the
        # main model + auxiliary model. apply_global_config() also projects the
        # profile's api_key and the free-form env block into os.environ (with
        # setdefault semantics, so shell env still wins).
        profile = apply_global_config() or {}
        aux_profile = profile.get("auxiliary_model") or {}
        if not isinstance(aux_profile, dict):
            aux_profile = {}
        model_wire_api = profile.get("wire_api") or os.getenv("AGENTICA_WIRE_API", "")
        model_reasoning = profile.get("reasoning") or os.getenv("AGENTICA_REASONING", "")
        model_reasoning_effort = "" if model_wire_api == "responses" else (
            profile.get("reasoning_effort") or os.getenv("AGENTICA_REASONING_EFFORT", "")
        )
        auxiliary_wire_api = aux_profile.get("wire_api") or os.getenv("AGENTICA_AUXILIARY_WIRE_API", "")
        auxiliary_reasoning = aux_profile.get("reasoning") or os.getenv("AGENTICA_AUXILIARY_REASONING", "")
        auxiliary_reasoning_effort = "" if auxiliary_wire_api == "responses" else (
            aux_profile.get("reasoning_effort") or os.getenv("AGENTICA_AUXILIARY_REASONING_EFFORT", "")
        )

        return cls(
            # Server
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8881")),
            debug=os.getenv("DEBUG", "").lower() in ("1", "true"),
            parent_pid=int(os.getenv("AGENTICA_GATEWAY_PARENT_PID", "0") or 0),

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

            # Conversation history window — inherits the SDK-wide default
            # (env var / config.yaml `settings.num_history_turns`); see
            # agentica.config.AGENTICA_NUM_HISTORY_TURNS.
            num_history_turns=AGENTICA_NUM_HISTORY_TURNS,

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

            # Slack (Socket Mode)
            slack_bot_token=os.getenv("SLACK_BOT_TOKEN"),
            slack_app_token=os.getenv("SLACK_APP_TOKEN"),
            slack_allowed_users=[
                u.strip() for u in os.getenv("SLACK_ALLOWED_USERS", "").split(",") if u.strip()
            ],
            slack_allowed_channels=[
                u.strip() for u in os.getenv("SLACK_ALLOWED_CHANNELS", "").split(",") if u.strip()
            ],

            # Peer bridge (IM -> this machine's CLI sessions); on unless
            # explicitly disabled.
            peer_bridge_enabled=os.getenv("PEER_BRIDGE", "true").strip().lower()
                not in ("0", "false", "no", "off"),

            # Model / path — profile first, env fallback, built-in default last.
            # When a config.yaml profile is present it wins (gateway is a
            # profile-driven service); env vars keep env-only setups working.
            model_provider=(profile.get("model_provider")
                or os.getenv("AGENTICA_MODEL_PROVIDER", "deepseek")),
            model_name=(profile.get("model_name")
                or os.getenv("AGENTICA_MODEL_NAME", "deepseek-v4-flash")),
            model_base_url=(profile.get("base_url")
                or os.getenv("AGENTICA_BASE_URL", "")),
            model_api_key=(profile.get("api_key")
                or os.getenv("AGENTICA_API_KEY", "")),
            model_wire_api=model_wire_api,
            model_thinking=(profile.get("thinking")
                or os.getenv("AGENTICA_MODEL_THINKING", "")),
            model_reasoning=model_reasoning,
            model_reasoning_effort=model_reasoning_effort,
            max_tokens=int(profile.get("max_tokens")
                or os.getenv("AGENTICA_MAX_TOKENS", "0") or 0),
            temperature=float(profile.get("temperature")
                or os.getenv("AGENTICA_TEMPERATURE", "0") or 0),
            top_p=float(profile.get("top_p")
                or os.getenv("AGENTICA_TOP_P", "0") or 0),
            context_window=int(profile.get("context_window")
                or os.getenv("AGENTICA_CONTEXT_WINDOW", "0") or 0),

            # Auxiliary model (leave empty to reuse main model)
            auxiliary_model_provider=(aux_profile.get("model_provider")
                or os.getenv("AGENTICA_AUXILIARY_MODEL_PROVIDER", "")),
            auxiliary_model_name=(aux_profile.get("model_name")
                or os.getenv("AGENTICA_AUXILIARY_MODEL_NAME", "")),
            auxiliary_base_url=(aux_profile.get("base_url")
                or os.getenv("AGENTICA_AUXILIARY_BASE_URL", "")),
            auxiliary_api_key=(aux_profile.get("api_key")
                or os.getenv("AGENTICA_AUXILIARY_API_KEY", "")),
            auxiliary_wire_api=auxiliary_wire_api,
            auxiliary_reasoning=auxiliary_reasoning,
            auxiliary_reasoning_effort=auxiliary_reasoning_effort,
        )


# Global settings instance
settings = Settings.from_env()
