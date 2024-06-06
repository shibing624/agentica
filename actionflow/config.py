# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

from loguru import logger

pwd_path = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.getenv("DOTENV_PATH", os.path.join(pwd_path, "../.env"))
try:
    from dotenv import load_dotenv  # noqa

    if load_dotenv(dotenv_path, override=True):
        logger.info(f"Loaded environment variables from {dotenv_path}")
except ImportError:
    logger.debug("dotenv not installed, skipping...")

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
default_model = os.getenv("OPENAI_DEFAULT_MODEL")
