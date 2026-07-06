# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Pydantic request/response models shared across routes.
"""
from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str
    session_id: str = "default"
    user_id: str = "default"
    agent_id: str = "main"
    work_dir: Optional[str] = None
    goal: str = ""
    skill: str = ""
    tool: str = ""
    approval_mode: str = "ask"


class ChatResponse(BaseModel):
    """Chat response payload."""
    content: str
    session_id: str
    user_id: str = "default"
    tool_calls: int = 0


class MemoryRequest(BaseModel):
    content: str
    user_id: str = "default"
    long_term: bool = False


class SendRequest(BaseModel):
    channel: str
    channel_id: str
    message: str


class CronJobCreateRequest(BaseModel):
    prompt: str
    schedule: str
    name: Optional[str] = None
    user_id: str = "default"
    timezone: str = "Asia/Shanghai"
    deliver: str = "local"
    timeout_seconds: float = 0.0
    max_retries: int = 0
    retry_delay_ms: int = 60000
    permissions: Optional[dict] = None


class CronJobUpdateRequest(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    schedule: Optional[str] = None
    deliver: Optional[str] = None
    timeout_seconds: Optional[float] = None
    max_retries: Optional[int] = None
    retry_delay_ms: Optional[int] = None
    permissions: Optional[dict] = None


class ModelSwitchRequest(BaseModel):
    model_provider: str
    model_name: str


class ProfileSwitchRequest(BaseModel):
    name: str


class ProfileUpsertRequest(BaseModel):
    name: str
    model_provider: str
    model_name: str
    base_url: str = ""
    api_key: Optional[str] = None
    reasoning_effort: Optional[str] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    auxiliary_model: Optional[dict] = None
    env: Optional[dict] = None


class RenameRequest(BaseModel):
    name: str


class ThinkingToggleRequest(BaseModel):
    enabled: bool


class BaseDirRequest(BaseModel):
    base_dir: str


class OpenRequest(BaseModel):
    path: str
    app: str = "finder"  # "finder" or "terminal"
