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

try:
    import yaml
except ImportError:
    yaml = None


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

        # Check in AGENTICA_HOME directory
        for config_file in config_files:
            config_path = os.path.join(AGENTICA_HOME, config_file)
            if os.path.exists(config_path):
                return config_path

        return ""

    def _load_config(self) -> None:
        """Load configuration from MCP config file.
        Supports both JSON and YAML formats.
        """
        if not self.config_path or not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                # Determine file format based on extension
                if self.config_path.lower().endswith(('.yaml', '.yml')):
                    if yaml is None:
                        raise ImportError("YAML support requires PyYAML. Install with 'pip install pyyaml'")
                    config = yaml.safe_load(f)
                else:  # Default to JSON
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
