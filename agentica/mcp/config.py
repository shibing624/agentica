# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from agentica.config import AGENTICA_HOME


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list] = None
    env: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: float = 5.0
    read_timeout: float = 300.0


class MCPConfig:
    """Manager for MCP server configurations"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.servers: Dict[str, MCPServerConfig] = {}
        self._load_config()

    def _find_config_file(self) -> str:
        """Find mcp_config.json in current or parent directories"""
        current_dir = os.getcwd()
        while current_dir != os.path.dirname(current_dir):
            config_path = os.path.join(current_dir, 'mcp_config.json')
            if os.path.exists(config_path):
                return config_path
            current_dir = os.path.dirname(current_dir)
        config_path = os.path.join(AGENTICA_HOME, 'mcp_config.json')
        if os.path.exists(config_path):
            return config_path
        else:
            return ""

    def _load_config(self) -> None:
        """Load configuration from mcp_config.json"""
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            for name, server_config in config.get('mcpServers', {}).items():
                self.servers[name] = MCPServerConfig(
                    name=name,
                    url=server_config.get('url'),
                    command=server_config.get('command'),
                    args=server_config.get('args', []),
                    env=server_config.get('env'),
                    headers=server_config.get('headers'),
                    timeout=server_config.get('timeout', 5.0),
                    read_timeout=server_config.get('read_timeout', 300.0)
                )
        except Exception as e:
            raise ValueError(f"Error loading MCP config from {self.config_path}: {e}")

    def get_server_config(self, name: str) -> Optional[MCPServerConfig]:
        """Get configuration for a named server"""
        return self.servers.get(name)

    def list_servers(self) -> Dict[str, MCPServerConfig]:
        """List all configured servers"""
        return self.servers
