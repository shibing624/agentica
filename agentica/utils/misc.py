# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import hashlib
import os
import time
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


def browser_toolkit_save_auth_cookie(
        cookie_json_path: str, url: str, wait_time: int = 60
):
    r"""Saves authentication cookies and browser storage state to a JSON file.

    This function launches a browser window and navigates to the specified URL,
    allowing the user to manually authenticate (log in) during a 60-second
    wait period.After authentication, it saves all cookies, localStorage, and
    sessionStorage data to the specified JSON file path, which can be used
    later to maintain authenticated sessions without requiring manual login.

    Args:
        cookie_json_path (str): Path where the authentication cookies and
            storage state will be saved as a JSON file. If the file already
            exists, it will be loaded first and then overwritten with updated
            state. The function checks if this file exists before attempting
            to use it.
        url (str): The URL to navigate to for authentication (e.g., a login
            page).
        wait_time (int): The time in seconds to wait for the user to manually
            authenticate.

    Usage:
        1. The function opens a browser window and navigates to the specified
            URL
        2. User manually logs in during the wait_time wait period
        3. Browser storage state (including auth cookies) is saved to the
           specified file
        4. The saved state can be used in subsequent browser sessions to
           maintain authentication

    Note:
        The wait_time sleep is intentional to give the user enough time to
        complete the manual authentication process before the storage state
        is captured.
    """
    from playwright.sync_api import sync_playwright

    playwright = sync_playwright().start()

    # Launch visible browser window using Chromium
    browser = playwright.chromium.launch(headless=False, channel="chromium")

    # Check if cookie file exists before using it
    storage_state = (
        cookie_json_path if os.path.exists(cookie_json_path) else None
    )

    # Create browser context with proper typing
    context = browser.new_context(
        accept_downloads=True, storage_state=storage_state
    )
    page = context.new_page()
    page.goto(url)  # Navigate to the authentication URL
    # Wait for page to fully load
    page.wait_for_load_state("load", timeout=1000)
    time.sleep(wait_time)  # Wait 60 seconds for user to manually authenticate
    # Save browser storage state (cookies, localStorage, etc.) to JSON file
    context.storage_state(path=cookie_json_path)

    browser.close()  # Close the browser when finished
