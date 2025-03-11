# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict


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


def dataclass_to_dict(dataclass_object, exclude: Optional[set[str]] = None, exclude_none: bool = False) -> dict:
    final_dict = asdict(dataclass_object)
    if exclude:
        for key in exclude:
            final_dict.pop(key)
    if exclude_none:
        final_dict = {k: v for k, v in final_dict.items() if v is not None}
    return final_dict
