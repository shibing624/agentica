# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os

from loguru import logger

pwd_path = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.environ.get("DOTENV_PATH", os.path.join(pwd_path, "../.env"))
try:
    from dotenv import load_dotenv  # noqa

    if load_dotenv(dotenv_path, override=True):
        logger.info(f"Loaded environment variables from {dotenv_path}")
except ImportError:
    logger.debug("dotenv not installed, skipping...")

api_key = os.environ.get("API_KEY")  # "your-api-key"
# OpenAI API Base URL; "https://api.moonshot.cn/v1" for Moonshot API
base_url = os.environ.get("API_BASE", "https://api.openai.com/v1")
default_model = os.environ.get("DEFAULT_MODEL", "gpt-3.5-turbo")  # "gpt-3.5-turbo" or "moonshot-v1-8k" and so on

# get url api key
JINA_API_KEY = os.environ.get("JINA_API_KEY")

# Search engine api key
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
# SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

# Code-interpreter E2B api key
E2B_API_KEY = os.environ.get("E2B_API_KEY")
