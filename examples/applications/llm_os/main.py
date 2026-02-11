# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: LLM OS - AI助手系统，基于DeepAgent实现

使用DeepAgent作为核心，内置文件操作、代码执行、网络搜索等能力。
支持知识库管理和会话持久化。

pip install streamlit agentica text2vec sqlalchemy lancedb pyarrow

run:
cd examples/applications/llm_os
streamlit run main.py
"""
import os
import sys
from textwrap import dedent
from typing import List, Optional

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentica import DeepAgent, OpenAIChat
from agentica.db.sqlite import SqliteDb
from agentica.document import Document
from agentica.emb.text2vec_emb import Text2VecEmb
from agentica.knowledge.base import Knowledge
from agentica.utils.log import logger
from agentica.vectordb.lancedb_vectordb import LanceDb

# 页面配置
PAGE_TITLE = "LLM OS"
PAGE_ICON = ":orange_heart:"
DEFAULT_MODEL = "gpt-4o"
MODEL_OPTIONS = ["gpt-4o", "gpt-5.2", "gpt-4o-mini"]

# 系统描述
SYSTEM_DESCRIPTION = dedent("""\
    You are the most advanced AI system in the world called `LLM-OS`.
    You have access to a set of powerful tools to assist the user.
    Your goal is to assist the user in the best way possible.
""")

# 系统指令
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

# 欢迎消息
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


def create_llm_os(
        model_id: str = DEFAULT_MODEL,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        debug_mode: bool = False,
) -> DeepAgent:
    """创建LLM OS实例
    
    Args:
        model_id: 模型ID
        user_id: 用户ID
        session_id: 会话ID
        debug_mode: 是否开启调试模式
        
    Returns:
        DeepAgent实例
    """
    logger.info(f"Creating LLM OS with model: {model_id}")

    # 初始化向量数据库和知识库
    embedder = Text2VecEmb()
    vector_db = LanceDb(
        uri="outputs/llm_os_lancedb",
        table_name="llm_os_documents",
        embedder=embedder,
    )
    knowledge = Knowledge(vector_db=vector_db)

    # 创建DeepAgent
    llm_os = DeepAgent(
        name="llm_os",
        model=OpenAIChat(id=model_id),
        session_id=session_id,
        user_id=user_id,
        description=SYSTEM_DESCRIPTION,
        instructions=SYSTEM_INSTRUCTIONS,
        # 数据库用于会话持久化
        db=SqliteDb(db_file="outputs/llm_os.db"),
        # 知识库
        knowledge=knowledge,
        # DeepAgent内置工具配置
        include_file_tools=True,
        include_execute=True,
        include_web_search=True,
        include_fetch_url=True,
        include_todos=True,
        include_task=True,
        # 知识库搜索
        search_knowledge=True,
        # 聊天历史
        read_chat_history=True,
        add_history_to_messages=True,
        num_history_responses=6,
        # 显示配置
        markdown=True,
        add_datetime_to_instructions=True,
        introduction=INTRODUCTION,
        debug_mode=debug_mode,
    )

    return llm_os


def init_session_state():
    """初始化session state"""
    defaults = {
        "llm_id": DEFAULT_MODEL,
        "llm_os": None,
        "llm_os_session_id": None,
        "messages": [{"role": "assistant", "content": "Ask me questions..."}],
        "debug_mode": False,
        "url_scrape_key": 0,
        "file_uploader_key": 100,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def restart_agent():
    """重启agent"""
    logger.debug("Restarting agent...")
    st.session_state["llm_os"] = None
    st.session_state["llm_os_session_id"] = None
    st.session_state["url_scrape_key"] += 1
    st.session_state["file_uploader_key"] += 1
    st.rerun()


def render_sidebar():
    """渲染侧边栏配置"""
    st.sidebar.markdown("### Settings")

    # 模型选择
    llm_id = st.sidebar.selectbox("Select Model", options=MODEL_OPTIONS, index=0)
    if st.session_state["llm_id"] != llm_id:
        st.session_state["llm_id"] = llm_id
        restart_agent()

    # Debug模式
    debug_mode = st.sidebar.checkbox(
        "Debug Mode",
        value=st.session_state["debug_mode"],
        help="Enable debug mode to see detailed logs"
    )
    if st.session_state["debug_mode"] != debug_mode:
        st.session_state["debug_mode"] = debug_mode
        restart_agent()

    return llm_id, debug_mode


def render_knowledge_base_controls(llm_os: DeepAgent):
    """渲染知识库控制组件"""
    if not llm_os.knowledge:
        return

    st.sidebar.markdown("### Knowledge Base")

    # URL添加
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

    # 文件上传
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

    # 清空知识库
    if llm_os.knowledge.vector_db:
        if st.sidebar.button("Clear Knowledge Base"):
            llm_os.knowledge.vector_db.delete()
            st.sidebar.success("Knowledge base cleared")


def render_session_controls(llm_os: DeepAgent, llm_id: str, debug_mode: bool):
    """渲染会话控制组件"""
    if not llm_os.db:
        return

    st.sidebar.markdown("### Sessions")

    # 会话选择
    session_ids: List[str] = llm_os.db.get_all_session_ids()
    if session_ids:
        current_session = st.session_state.get("llm_os_session_id")
        selected_session = st.sidebar.selectbox("Session ID", options=session_ids)

        if current_session and current_session != selected_session:
            logger.info(f"Loading session: {selected_session}")
            st.session_state["llm_os"] = create_llm_os(
                model_id=llm_id,
                session_id=selected_session,
                debug_mode=debug_mode,
            )
            st.rerun()

    # 新建会话
    if st.sidebar.button("New Session"):
        restart_agent()


def main():
    """主函数"""
    # 页面配置
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.title(PAGE_TITLE)
    st.markdown("##### :orange_heart: built using [agentica](https://github.com/shibing624/agentica)")

    # 初始化
    init_session_state()

    # 侧边栏配置
    llm_id, debug_mode = render_sidebar()

    # 获取或创建agent
    if st.session_state["llm_os"] is None:
        logger.info(f"Creating new LLM OS with model: {llm_id}")
        st.session_state["llm_os"] = create_llm_os(
            model_id=llm_id,
            debug_mode=debug_mode,
        )

    llm_os: DeepAgent = st.session_state["llm_os"]

    # 加载会话
    try:
        st.session_state["llm_os_session_id"] = llm_os.load_session()
    except Exception as e:
        st.warning(f"Could not load session: {e}")
        return

    # 加载聊天历史
    chat_history = llm_os.memory.get_messages()
    if chat_history:
        st.session_state["messages"] = chat_history
    else:
        st.session_state["messages"] = [{"role": "assistant", "content": "Ask me questions..."}]

    # 用户输入
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})

    # 显示消息
    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message.get("content", ""))

    # 生成回复
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in llm_os.run_sync(question, stream=True):
                response += delta.content
                resp_container.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

    # 知识库控制
    render_knowledge_base_controls(llm_os)

    # 会话控制
    render_session_controls(llm_os, llm_id, debug_mode)


if __name__ == '__main__':
    main()
