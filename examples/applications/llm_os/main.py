# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: LLM OS - AI助手系统，基于 Agent + 内置工具实现。

亮点：
- auxiliary_model + fallback_models：副模型（便宜，用于记忆提取等副任务）+ 回退模型（主模型失败时兜底）
- Workspace + 长期记忆：会话归档 + 后台异步抽取记忆，不阻塞响应
- SensitiveWordHook：回答返回后扫描用户输入，命中违规词类（辱骂 / 投诉 / 极端情绪）就告警

依赖：
    pip install streamlit agentica sqlalchemy lancedb pyarrow

run:
    cd examples/applications/llm_os
    streamlit run main.py
"""
import asyncio
import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from textwrap import dedent
from typing import Awaitable, Callable, Dict, List, Optional, Pattern, Union

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentica import (
    Agent,
    AgentHooks,
    OpenAIChat,
    Workspace,
    WorkspaceMemoryConfig,
)
from agentica.agent.config import PromptConfig, ToolConfig
from agentica.document import Document
from agentica.embedding.zhipuai import ZhipuAIEmbedding
from agentica.knowledge.base import Knowledge
from agentica.tools.buildin_tools import get_builtin_tools
from agentica.utils.log import logger
from agentica.vectordb.lancedb_vectordb import LanceDb

PAGE_TITLE = "LLM OS"
PAGE_ICON = ":orange_heart:"
DEFAULT_MODEL = "gpt-4o"
MODEL_OPTIONS = ["gpt-4o", "gpt-5.5", "gpt-4o-mini"]
AUXILIARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4o-mini"

SYSTEM_DESCRIPTION = dedent("""\
    You are the most advanced AI system in the world called `LLM-OS`.
    You have access to a set of powerful tools to assist the user.
    Your goal is to assist the user in the best way possible.
""")

SYSTEM_INSTRUCTIONS = [
    "When the user sends a message, first **think** and determine if:\n"
    " - You can answer by using a tool available to you\n"
    " - You need to search the knowledge base\n"
    " - You need to search the internet\n"
    " - You need to ask a clarifying question",
    "If the user asks about a topic, first ALWAYS search your knowledge base using the `search_knowledge` tool.",
    "If you don't find relevant information in your knowledge base, use the `web_search` tool to search the internet.",
    "If the user asks to summarize the conversation or if you need to reference your chat history, use the `get_chat_history` tool.",
    "If the user's message is unclear, ask clarifying questions to get more information.",
    "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
    "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
]

INTRODUCTION = dedent("""\
    Hi, I'm your LLM OS.
    I have access to powerful tools to assist you:
    - File operations (read, write, edit)
    - Code execution
    - Web search
    - URL content fetching
    - Task management
    Let's solve some problems together!
""")


# ──────────────────────────────────────────────────────────────────────────────
# SensitiveWordHook：回答返回后扫描用户输入，命中违规词类就告警
#
# 生产化要点：
# 1. 词典统一过滤：硬性丢弃 <2 字的词（"死"/"滚"/"+"），并去重
# 2. 预编译正则：每类一个 alternation 模式，扫描 O(L) 而不是 O(L*N)
# 3. ASCII 词加 \b 边界：避免 "sb" 误匹配 "absurd"、"bb" 误匹配 "abby"
# 4. 短输入跳过：用户输入 < min_input_length 直接 return
# 5. 回调线程化：同步回调跑在线程池，不阻塞事件循环；async 回调直接 await
# 6. 回调异常隔离：任何 callback 异常只 log，不影响主响应
# ──────────────────────────────────────────────────────────────────────────────

# 词典按"类别 → 词列表"组织，命中时输出类别，方便下游分级处理。
SENSITIVE_WORDS: Dict[str, List[str]] = {
    "辱骂/人身攻击": [
        "去死", "有病", "垃圾", "骗子", "脑残", "你妈的", "死全家", "bb",
        "狗叫", "脑子有坑", "脑子有病", "滚蛋", "草你", "我草", "卧槽",
        "我操", "操你", "sb", "傻逼", "坑钱",
    ],
    "投诉威胁": [
        "投诉", "起诉", "举报", "曝光", "法院", "找律师", "律师函", "315",
        "媒体", "工商局", "市长热线",
    ],
    "极端情绪": [
        "恨你们", "恶心", "倒闭吧", "再也不信了",
    ],
    "质疑AI/要求人工": [
        "你是AI", "机器人", "是真人吗", "人工客服", "转人工", "找真人",
        "答非所问", "不是人", "智障", "模板回复", "套话",
    ],
}

MIN_WORD_LEN = 2  # 硬性下限：单字误报率太高，全部丢弃
DEFAULT_MIN_INPUT_LEN = 2  # 用户输入短于这个不扫描

# 回调签名：同步或异步皆可
HitCallback = Callable[["SensitiveHit"], Union[None, Awaitable[None]]]


@dataclass
class SensitiveHit:
    """单次命中记录，方便下游邮件/监控消费。"""
    session_id: Optional[str]
    user_id: Optional[str]
    user_input: str
    matches: Dict[str, List[str]]  # 类别 -> 命中的词（去重）
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def categories(self) -> List[str]:
        return list(self.matches.keys())

    def to_alert_text(self) -> str:
        parts = [f"[{c}] {'/'.join(words)}" for c, words in self.matches.items()]
        return "命中：" + "；".join(parts)


def _compile_vocab(vocab: Dict[str, List[str]]) -> Dict[str, Pattern[str]]:
    """把词典编译成 {类别: 正则}。

    - 丢弃 <``MIN_WORD_LEN`` 字的词
    - 去重 + 按长度倒序排（保证 "脑子有坑" 优先于 "脑残" 这类不会发生，但长词优先在 alternation 里更稳）
    - ASCII 词补 ``\b`` 词边界
    - 大小写不敏感
    """
    compiled: Dict[str, Pattern[str]] = {}
    for category, words in vocab.items():
        clean = sorted(
            {w.strip() for w in words if len(w.strip()) >= MIN_WORD_LEN},
            key=lambda w: (-len(w), w),
        )
        if not clean:
            continue
        parts: List[str] = []
        for w in clean:
            esc = re.escape(w)
            # ASCII-only token：加自定义边界，避免 "sb" 匹配 "absurd"。
            # 不能用 \b：Python re 把中文也算 \w，"去315投" 里 "去/3" 之间
            # 没有 \b 边界，会漏报。用 lookaround 只针对 [A-Za-z0-9] 邻居。
            if w.isascii() and re.fullmatch(r"[A-Za-z0-9]+", w):
                parts.append(rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])")
            else:
                parts.append(esc)
        compiled[category] = re.compile("|".join(parts), re.IGNORECASE)
    return compiled


# 模块加载时编译一次，扫描时直接用
_COMPILED_VOCAB: Dict[str, Pattern[str]] = _compile_vocab(SENSITIVE_WORDS)


def scan_sensitive(
    text: str,
    compiled: Optional[Dict[str, Pattern[str]]] = None,
) -> Dict[str, List[str]]:
    """扫描文本，返回 ``{类别: [命中词去重保序,...]}``，无命中返回空 dict。"""
    if not text:
        return {}
    table = compiled if compiled is not None else _COMPILED_VOCAB
    hits: Dict[str, List[str]] = {}
    for category, pattern in table.items():
        seen: List[str] = []
        seen_set: set = set()
        for m in pattern.finditer(text):
            tok = m.group(0).lower()
            if tok not in seen_set:
                seen_set.add(tok)
                seen.append(tok)
        if seen:
            hits[category] = seen
    return hits


class SensitiveWordHook(AgentHooks):
    """回答后扫描用户输入，命中违规词时调用 ``on_hit`` 回调。

    线上可在 ``on_hit`` 里发邮件 / 推飞书 / 写监控指标。
    回调可以是同步函数（会跑在线程池里不阻塞事件循环）或 ``async`` 协程。
    """

    def __init__(
        self,
        on_hit: Optional[HitCallback] = None,
        vocab: Optional[Dict[str, List[str]]] = None,
        min_input_length: int = DEFAULT_MIN_INPUT_LEN,
    ):
        self._on_hit = on_hit
        self._compiled = _compile_vocab(vocab) if vocab is not None else _COMPILED_VOCAB
        self._min_input_length = max(MIN_WORD_LEN, min_input_length)

    async def on_end(self, agent, output, **kwargs) -> None:
        run_input = agent.run_input
        if not isinstance(run_input, str):
            return
        stripped = run_input.strip()
        if len(stripped) < self._min_input_length:
            return

        matches = scan_sensitive(stripped, self._compiled)
        if not matches:
            return

        hit = SensitiveHit(
            session_id=agent.session_id,
            user_id=getattr(agent.workspace, "user_id", None) if agent.workspace else None,
            user_input=stripped,
            matches=matches,
        )
        logger.warning(
            f"[SensitiveWordHook] agent={agent.name} session={hit.session_id} "
            f"user={hit.user_id} {hit.to_alert_text()} | input={stripped!r}"
        )

        if self._on_hit is None:
            return
        try:
            if inspect.iscoroutinefunction(self._on_hit):
                await self._on_hit(hit)
            else:
                # 同步回调可能做阻塞 I/O（发邮件 / HTTP），扔进线程池避免卡事件循环
                await asyncio.to_thread(self._on_hit, hit)
        except Exception as e:
            logger.error(f"[SensitiveWordHook] on_hit callback failed: {e}", exc_info=True)


def _streamlit_alert_sink(hit: SensitiveHit) -> None:
    """默认回调：把命中信息塞进 st.session_state，供 UI 展示。

    生产环境替换为发邮件 / 推飞书 / 上报监控即可，签名保持 ``(SensitiveHit) -> None``
    或 ``async (SensitiveHit) -> None``。
    """
    bucket = st.session_state.setdefault("sensitive_alerts", [])
    bucket.append(hit)


def create_llm_os(
        model_id: str = DEFAULT_MODEL,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        debug: bool = False,
) -> Agent:
    """创建 LLM OS Agent。

    Args:
        model_id: 主模型 id
        session_id: 会话 id，用于 Workspace 隔离
        user_id: 用户 id，用于 Workspace 隔离
        debug: 是否开启调试模式
    """
    logger.info(f"Creating LLM OS with model: {model_id}")

    embedder = ZhipuAIEmbedding()
    vector_db = LanceDb(
        uri="outputs/llm_os_lancedb",
        table_name="llm_os_documents",
        embedding=embedder,
    )
    knowledge = Knowledge(vector_db=vector_db)

    model = OpenAIChat(id=model_id)
    auxiliary_model = OpenAIChat(id=AUXILIARY_MODEL)
    fallback_model = OpenAIChat(id=FALLBACK_MODEL)

    llm_os = Agent(
        name="llm_os",
        model=model,
        auxiliary_model=auxiliary_model,
        fallback_models=[fallback_model],
        description=SYSTEM_DESCRIPTION,
        instructions=SYSTEM_INSTRUCTIONS,
        knowledge=knowledge,
        tools=get_builtin_tools(),
        tool_config=ToolConfig(search_knowledge=True),
        add_history_to_context=True,
        num_history_turns=6,
        prompt_config=PromptConfig(markdown=True, introduction=INTRODUCTION, enable_agentic_prompt=True),
        # 工作区 + 长期记忆：归档 + 后台抽取，不阻塞响应
        workspace=Workspace(
            os.path.expanduser("~/.agentica/workspace"),
            user_id=user_id,
        ),
        session_id=session_id,
        enable_long_term_memory=True,
        long_term_memory_config=WorkspaceMemoryConfig(
            auto_archive=True,
            auto_extract_memory=True,
            auto_extract_memory_background=True,
            load_workspace_context=True,
            load_workspace_memory=True,
            max_memory_entries=10,
            sync_memories_to_global_agent_md=False,
        ),
        # 敏感词监控：命中后 _streamlit_alert_sink 把告警塞进 session_state；
        # 线上换成发邮件 / 推飞书 / 上报监控即可。
        hooks=SensitiveWordHook(on_hit=_streamlit_alert_sink),
        debug=debug,
    )

    return llm_os


def init_session_state():
    defaults = {
        "llm_id": DEFAULT_MODEL,
        "llm_os": None,
        "llm_os_session_id": None,
        "messages": [{"role": "assistant", "content": "Ask me questions..."}],
        "debug": False,
        "url_scrape_key": 0,
        "file_uploader_key": 100,
        "sensitive_alerts": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def restart_agent():
    logger.debug("Restarting agent...")
    st.session_state["llm_os"] = None
    st.session_state["llm_os_session_id"] = None
    st.session_state["url_scrape_key"] += 1
    st.session_state["file_uploader_key"] += 1
    st.session_state["sensitive_alerts"] = []
    st.rerun()


def render_sidebar():
    st.sidebar.markdown("### Settings")

    llm_id = st.sidebar.selectbox("Select Model", options=MODEL_OPTIONS, index=0)
    if st.session_state["llm_id"] != llm_id:
        st.session_state["llm_id"] = llm_id
        restart_agent()

    debug = st.sidebar.checkbox(
        "Debug Mode",
        value=st.session_state["debug"],
        help="Enable debug mode to see detailed logs"
    )
    if st.session_state["debug"] != debug:
        st.session_state["debug"] = debug
        restart_agent()

    return llm_id, debug


def render_knowledge_base_controls(llm_os: Agent):
    if not llm_os.knowledge:
        return

    st.sidebar.markdown("### Knowledge Base")

    input_url = st.sidebar.text_input(
        "Add URL to Knowledge Base",
        key=st.session_state["url_scrape_key"]
    )
    if st.sidebar.button("Add URL") and input_url:
        input_url = input_url.strip()
        cache_key = f"{input_url}_scraped"
        if cache_key not in st.session_state:
            with st.sidebar.status("Processing URL...", expanded=True):
                web_documents = llm_os.knowledge.read_url(input_url)
                if web_documents:
                    llm_os.knowledge.load_documents(web_documents, upsert=True)
                    st.session_state[cache_key] = True
                    st.sidebar.success("URL added successfully")
                else:
                    st.sidebar.error("Could not read URL")

    file_types = ["pdf", "txt", "md", "docx", "xlsx", "json", "jsonl", "csv", "tsv", "html"]
    uploaded_file = st.sidebar.file_uploader(
        "Upload Document",
        type=file_types,
        key=st.session_state["file_uploader_key"]
    )
    if uploaded_file:
        save_dir = "outputs/uploads/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded_file.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cache_key = f"{uploaded_file.name}_uploaded"
        if cache_key not in st.session_state:
            with st.sidebar.status("Processing document...", expanded=True):
                documents: List[Document] = llm_os.knowledge.read_file(save_path)
                if documents:
                    llm_os.knowledge.load_documents(documents, upsert=True)
                    st.session_state[cache_key] = True
                    st.sidebar.success(f"Uploaded {len(documents)} document(s)")
                else:
                    st.sidebar.error(f"Could not read file: {uploaded_file.name}")

    if llm_os.knowledge.vector_db:
        if st.sidebar.button("Clear Knowledge Base"):
            llm_os.knowledge.vector_db.delete()
            st.sidebar.success("Knowledge base cleared")


def render_session_controls(llm_os: Agent, llm_id: str, debug: bool):
    st.sidebar.markdown("### Sessions")

    if st.sidebar.button("New Session"):
        restart_agent()


def render_sensitive_alerts():
    """侧边栏展示最近的违规词命中记录。"""
    alerts: List[SensitiveHit] = st.session_state.get("sensitive_alerts", [])
    if not alerts:
        return
    st.sidebar.markdown("### ⚠️ Sensitive Alerts")
    st.sidebar.caption(f"共 {len(alerts)} 条，展示最近 5 条")
    for hit in alerts[-5:][::-1]:
        with st.sidebar.expander(f"{hit.timestamp} · {' / '.join(hit.categories)}"):
            st.write(hit.to_alert_text())
            st.caption(f"input: {hit.user_input}")
    if st.sidebar.button("Clear Alerts"):
        st.session_state["sensitive_alerts"] = []
        st.rerun()


def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.title(PAGE_TITLE)
    st.markdown("##### :orange_heart: built using [agentica](https://github.com/shibing624/agentica)")

    init_session_state()

    llm_id, debug = render_sidebar()

    if st.session_state["llm_os"] is None:
        logger.info(f"Creating new LLM OS with model: {llm_id}")
        st.session_state["llm_os"] = create_llm_os(
            model_id=llm_id,
            debug=debug,
        )

    llm_os: Agent = st.session_state["llm_os"]

    chat_history = llm_os.working_memory.get_messages()
    if chat_history:
        st.session_state["messages"] = chat_history
    else:
        st.session_state["messages"] = [{"role": "assistant", "content": "Ask me questions..."}]

    if prompt := st.chat_input("Ask me anything..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})

    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message.get("content", ""))

    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in llm_os.run_stream_sync(question):
                response += delta.content
                resp_container.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

        # 回答返回后 SensitiveWordHook.on_end 已经把命中写进 session_state；
        # 这里若有最新命中，在对话区也提示一下，方便人工值班看到。
        alerts = st.session_state.get("sensitive_alerts", [])
        if alerts and alerts[-1].user_input == question:
            st.warning(f"⚠️ 检测到客户消息含违规风险词 → {alerts[-1].to_alert_text()}")

    render_knowledge_base_controls(llm_os)
    render_session_controls(llm_os, llm_id, debug)
    render_sensitive_alerts()


if __name__ == '__main__':
    main()
