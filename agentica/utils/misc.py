# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union

from agentica.model.message import Message


def get_text_from_message(message: Union[List, Dict, str, Message]) -> str:
    """Return the user texts from the message"""

    if isinstance(message, str):
        return message
    if isinstance(message, list):
        text_messages = []
        if len(message) == 0:
            return ""

        if "type" in message[0]:
            for m in message:
                m_type = m.get("type")
                if m_type is not None and isinstance(m_type, str):
                    m_value = m.get(m_type)
                    if m_value is not None and isinstance(m_value, str):
                        if m_type == "text":
                            text_messages.append(m_value)
                        # if m_type == "image_url":
                        #     text_messages.append(f"Image: {m_value}")
                        # else:
                        #     text_messages.append(f"{m_type}: {m_value}")
        elif "role" in message[0]:
            for m in message:
                m_role = m.get("role")
                if m_role is not None and isinstance(m_role, str):
                    m_content = m.get("content")
                    if m_content is not None and isinstance(m_content, str):
                        if m_role == "user":
                            text_messages.append(m_content)
        if len(text_messages) > 0:
            return "\n".join(text_messages)
    if isinstance(message, dict):
        if "content" in message:
            return get_text_from_message(message["content"])
    if isinstance(message, Message) and message.content is not None:
        return get_text_from_message(message.content)
    return ""


def remove_indent(s: Optional[str]) -> Optional[str]:
    """
    Remove the indent from a string.

    Args:
        s (str): String to remove indent from

    Returns:
        str: String with indent removed
    """
    if s is not None and isinstance(s, str):
        return "\n".join([line.strip() for line in s.split("\n")])
    return None


def merge_dictionaries(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    """
    Recursively merges two dictionaries.
    If there are conflicting keys, values from 'b' will take precedence.

    @params:
    a (Dict[str, Any]): The first dictionary to be merged.
    b (Dict[str, Any]): The second dictionary, whose values will take precedence.

    Returns:
    None: The function modifies the first dictionary in place.
    """
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            merge_dictionaries(a[key], b[key])
        else:
            a[key] = b[key]


def current_datetime() -> datetime:
    return datetime.now()


def current_datetime_utc() -> datetime:
    return datetime.now(timezone.utc)


def current_datetime_utc_str() -> str:
    return current_datetime_utc().strftime("%Y-%m-%dT%H:%M:%S")


def calculate_sha256(file):
    sha256 = hashlib.sha256()
    # Read the file in chunks to efficiently handle large files
    for chunk in iter(lambda: file.read(8192), b""):
        sha256.update(chunk)
    return sha256.hexdigest()


def calculate_sha256_string(string):
    # Create a new SHA-256 hash object
    sha256_hash = hashlib.sha256()
    # Update the hash object with the bytes of the input string
    sha256_hash.update(string.encode("utf-8"))
    # Get the hexadecimal representation of the hash
    hashed_string = sha256_hash.hexdigest()
    return hashed_string


def literal_similarity(text1, text2):
    """
    判断两个文本的字面相似度

    参数:
    text1: 第一个文本字符串
    text2: 第二个文本字符串

    返回值:
    相似度分数，介于0和1之间，其中1表示完全相同
    """
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, text1, text2)
    return matcher.ratio()
