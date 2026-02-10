# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ACP Protocol Handler

Handles JSON-RPC message parsing, validation, and serialization.
"""


import json
import sys
from typing import Any, Dict, Optional, Union

from agentica.acp.types import ACPRequest, ACPResponse, ACPErrorCode
from agentica.utils.log import logger


class ACPProtocolHandler:
    """Handles JSON-RPC protocol communication over stdio"""
    
    def __init__(self):
        self._buffer = ""
        
    def read_message(self) -> Optional[ACPRequest]:
        """
        Read a JSON-RPC message from stdin.
        
        Format: Content-Length: <length>\r\n\r\n<json_body>
        
        Returns:
            ACPRequest if valid message received, None if EOF
        """
        try:
            # Read headers
            while True:
                line = sys.stdin.readline()
                if not line:
                    return None  # EOF
                    
                line = line.strip()
                if not line:
                    break  # End of headers
                    
                if line.startswith("Content-Length: "):
                    content_length = int(line.split(": ")[1])
                elif line.startswith("Content-Type: "):
                    # Ignore content type for now
                    pass
            
            # Read body
            if 'content_length' not in locals():
                logger.error("Missing Content-Length header")
                return None
                
            body = sys.stdin.read(content_length)
            if not body:
                return None
                
            # Parse JSON
            data = json.loads(body)
            
            # Validate it's a request (has method)
            if "method" not in data:
                logger.warning(f"Received non-request message: {data}")
                return None
                
            return ACPRequest.from_dict(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading message: {e}")
            return None
    
    def write_message(self, response: ACPResponse) -> None:
        """
        Write a JSON-RPC message to stdout.
        
        Format: Content-Length: <length>\r\n\r\n<json_body>
        """
        try:
            body = json.dumps(response.to_dict(), ensure_ascii=False)
            content_length = len(body.encode('utf-8'))
            
            message = f"Content-Length: {content_length}\r\n\r\n{body}"
            
            sys.stdout.write(message)
            sys.stdout.flush()
            
        except Exception as e:
            logger.error(f"Error writing message: {e}")
    
    def send_error(self, request_id: Union[str, int, None], code: int, message: str, data: Any = None) -> None:
        """Send an error response"""
        if request_id is None:
            request_id = 0
        response = ACPResponse.create_error(request_id, code, message, data)
        self.write_message(response)
    
    def send_success(self, request_id: Union[str, int], result: Dict[str, Any]) -> None:
        """Send a success response"""
        response = ACPResponse.create_success(request_id, result)
        self.write_message(response)
    
    def send_notification(self, method: str, params: Dict[str, Any]) -> None:
        """
        Send a notification to the client.
        
        Notifications are one-way messages that don't require a response.
        They have id=null according to JSON-RPC spec.
        """
        try:
            body = json.dumps({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }, ensure_ascii=False)
            content_length = len(body.encode('utf-8'))
            
            message = f"Content-Length: {content_length}\r\n\r\n{body}"
            
            sys.stdout.write(message)
            sys.stdout.flush()
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")


def create_error_response(request_id: Union[str, int, None], code: int, message: str) -> Dict[str, Any]:
    """Create a JSON-RPC error response dict"""
    response = {
        "jsonrpc": "2.0",
        "id": request_id if request_id is not None else 0,
        "error": {
            "code": code,
            "message": message,
        }
    }
    return response


def create_success_response(request_id: Union[str, int], result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a JSON-RPC success response dict"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }
