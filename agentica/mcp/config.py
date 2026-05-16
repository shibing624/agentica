# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
from agentica.config import AGENTICA_HOME
from agentica.utils.log import logger


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list] = None
    env: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: float = 15.0
    read_timeout: float = 300.0
    enable: bool = True  # Whether to load this MCP server


class MCPConfig:
    """Manager for MCP server configurations"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.servers: Dict[str, MCPServerConfig] = {}
        self._load_config()

    def _find_config_file(self) -> str:
        """Find MCP configuration file in current or parent directories.
        Supports JSON (.json) and YAML (.yaml, .yml) formats.
        """
        current_dir = os.getcwd()
        # List of supported config file names
        config_files = ['mcp_config.json', 'mcp_config.yaml', 'mcp_config.yml']

        while current_dir != os.path.dirname(current_dir):
            for config_file in config_files:
                config_path = os.path.join(current_dir, config_file)
                if os.path.exists(config_path):
                    return config_path
            current_dir = os.path.dirname(current_dir)

        # Last-resort: check the user-global AGENTICA_HOME directory.
        # In multi-tenant SDK deployments this is a footgun — all tenants
        # in the process would share whatever MCP servers (and their
        # credentials) the operator dropped in ~/.agentica/. Pass an
        # explicit ``config_path`` (per workspace or per request) to opt
        # out; the WARN below makes the unsafe path observable.
        for config_file in config_files:
            config_path = os.path.join(AGENTICA_HOME, config_file)
            if os.path.exists(config_path):
                logger.warning(
                    "MCP config falling back to user-global %s — all agents "
                    "in this process will share these servers and credentials. "
                    "Pass MCPConfig(config_path=...) explicitly for multi-tenant safety.",
                    config_path,
                )
                return config_path

        return ""

    def _load_config(self) -> None:
        """Load configuration from MCP config file.
        Supports both JSON and YAML formats.
        Empty files are silently skipped (treated as no servers configured).
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # Empty file → no servers configured, nothing to do
            if not content:
                return

            # Determine file format based on extension
            if self.config_path.lower().endswith(('.yaml', '.yml')):
                config = yaml.safe_load(content)
            else:  # Default to JSON
                config = json.loads(content)

            # yaml.safe_load on whitespace-only content returns None
            if not config:
                return

            for name, server_config in config.get('mcpServers', {}).items():
                # Check enable field, default to True if not specified
                enable = server_config.get('enable', True)
                if not enable:
                    continue  # Skip disabled MCP servers

                self.servers[name] = MCPServerConfig(
                    name=name,
                    url=server_config.get('url'),
                    command=server_config.get('command'),
                    args=server_config.get('args', []),
                    env=server_config.get('env'),
                    headers=server_config.get('headers'),
                    timeout=server_config.get('timeout', 15.0),
                    read_timeout=server_config.get('read_timeout', 300.0),
                    enable=enable
                )
        except ImportError as e:
            raise e
        except Exception as e:
            raise ValueError(f"Error loading MCP config from {self.config_path}: {e}")

    def get_server_config(self, name: str) -> Optional[MCPServerConfig]:
        """Get configuration for a named server"""
        return self.servers.get(name)

    def list_servers(self) -> Dict[str, MCPServerConfig]:
        """List all configured servers"""
        return self.servers
