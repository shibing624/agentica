# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Self evolving agent demo

具有反思和增强记忆能力的自我进化智能体(SAGE)，以解决大型语言模型（LLM）面临的挑战，如持续决策、缺乏长期记忆和动态环境中有限的上下文窗口。
SAGE框架它集成了迭代反馈、反思机制和记忆优化机制，以增强智能体在处理多任务和长时间信息方面的能力。
这些智能体可以自适应地调整策略、优化信息存储和传输，并通过自我进化有效地减轻认知负荷。

Key Points
- SAGE框架引入了反思机制，增强了代理的自我调整能力，使他们能够更有效地利用历史信息并做出高效决策。
- 该框架包括记忆优化机制，帮助代理有选择地保留关键信息，减少多代理系统中信息过载的问题。
- 实验结果表明，SAGE框架在多个具有挑战性的现实世界任务中都取得了显著的改进，对较小的模型效果尤其明显。
- SAGE框架可以扩展到其他大型语言模型，并在AgentBench等基准测试和长文本任务上取得了最先进的结果。

本项目的简化实现：
1. 使用PythonAgent作为SAGE智能体，使用AzureOpenAIChat作为LLM, 具备code-interpreter功能，可以执行Python代码，并自动纠错。
2. 使用CsvMemoryDb作为SAGE智能体的记忆，用于存储用户的问题和答案，下次遇到相似的问题时，可以直接返回答案。

install:
pip install streamlit agentica sqlalchemy lancedb pyarrow

run:
streamlit run self_evolving_agent_demo.py
"""
import os
import sys
from textwrap import dedent
from typing import List, Optional

import streamlit as st

sys.path.append('..')
from agentica import Agent, AzureOpenAIChat, PythonAgent
from agentica.utils.log import logger
from agentica.tools.search_serper_tool import SearchSerperTool
from agentica.knowledge import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb
from agentica import AgentMemory, CsvMemoryDb
from agentica import SqlAgentStorage


def get_sage(
        llm,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        debug_mode: bool = True,
) -> Agent:
    llm_model_name = llm.id
    logger.info(f"-*- Creating {llm_model_name} SAGE agent -*-")

    # Add tools available to the SAGE agent
    tools = [SearchSerperTool()]

    embedder = Text2VecEmb()
    lance_db = LanceDb(
        uri="outputs/sage_lancedb",
        table_name="sage_documents",
        embedder=embedder,
    )
    memory_file = "outputs/sage_memory.csv"
    knowledge_base = Knowledge(
        data_path=memory_file if os.path.exists(memory_file) else [],
        vector_db=lance_db
    )
    knowledge_base.load()

    # Create the SAGE agent Assistant
    sage = PythonAgent(
        name="sage",
        session_id=session_id,
        user_id=user_id,
        model=llm,
        description=dedent(
            """
        You are the most advanced AI system in the world called `具有反思和增强记忆能力的自我进化智能体(SAGE)`.
        Your goal is to assist the user.
        """
        ),
        # Add long-term memory to the SAGE agent backed by a Sqlite database
        storage=SqlAgentStorage(table_name="sage", db_file="outputs/sage.db"),
        # Add a knowledge base to the SAGE agent
        knowledge=knowledge_base,
        # Add selected tools to the SAGE agent
        tools=tools,
        # Show tool calls in the chat
        show_tool_calls=True,
        # This setting gives the LLM a tool to search the knowledge base for information
        search_knowledge=True,
        update_knowledge=True,
        # This setting gives the LLM a tool to get chat history
        read_chat_history=True,
        # This setting adds chat history to the messages
        add_history_to_messages=True,
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # This setting adds the current datetime to the instructions
        add_datetime_to_instructions=True,
        memory=AgentMemory(db=CsvMemoryDb(memory_file)),
        create_umemories=True,
        force_update_memory_after_run=True,
        debug_mode=debug_mode,
    )
    return sage


def main():
    st.set_page_config(
        page_title="SAGE Agent",
        page_icon=":orange_heart:",
    )
    st.title("SAGE Agent")
    st.markdown("##### :orange_heart: built using [agentica](https://github.com/shibing624/agentica)")

    # Get LLM Model
    llm_id = st.sidebar.selectbox("Select LLM", options=["gpt-4o", "gpt-4o-mini"]) or "gpt-4o"
    # Set llm_id in session state
    if "llm_id" not in st.session_state:
        st.session_state["llm_id"] = llm_id
    # Restart the assistant if llm_id changes
    elif st.session_state["llm_id"] != llm_id:
        st.session_state["llm_id"] = llm_id
        restart_assistant()

    # Get the assistant
    sage: Agent
    if "sage" not in st.session_state or st.session_state["sage"] is None:
        logger.info(f"---*--- Creating {llm_id} SAGE agent ---*---")
        llm = AzureOpenAIChat(model=llm_id)
        sage = get_sage(
            llm=llm,
        )
        st.session_state["sage"] = sage
    else:
        sage = st.session_state["sage"]

    # Create assistant run (i.e. log to database) and save run_id in session state
    try:
        st.session_state["sage_run_id"] = sage.create_run()
    except Exception:
        st.warning("Could not create SAGE agent run, is the database running?")
        return

    # Load existing messages
    assistant_chat_history = sage.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = assistant_chat_history
    else:
        logger.debug("No chat history found")
        st.session_state["messages"] = [{"role": "assistant", "content": "Ask me questions..."}]

    # Prompt for user input
    if prompt := st.chat_input():
        logger.debug(f"User: {prompt}")
        st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display existing chat messages
    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in sage.run(question):
                response += delta
                resp_container.markdown(response)
            logger.debug(f"Assistant response: {response}")
            st.session_state["messages"].append({"role": "assistant", "content": response})

    if sage.knowledge_base and sage.knowledge_base.vector_db:
        if st.sidebar.button("Clear memory"):
            sage.knowledge_base.vector_db.clear()
            st.sidebar.success("Memory cleared")

    if sage.storage:
        sage_run_ids: List[str] = sage.storage.get_all_session_ids()
        new_sage_run_id = st.sidebar.selectbox("Run ID", options=sage_run_ids)
        if st.session_state["sage_run_id"] != new_sage_run_id:
            logger.info(f"---*--- Loading {llm_id} run: {new_sage_run_id} ---*---")
            llm = AzureOpenAIChat(model=llm_id)
            st.session_state["sage"] = get_sage(
                llm=llm,
                session_id=new_sage_run_id,
            )
            st.rerun()

    if st.sidebar.button("New Run"):
        restart_assistant()


def restart_assistant():
    logger.debug("---*--- Restarting Assistant ---*---")
    st.session_state["sage"] = None
    st.session_state["sage_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    st.rerun()


if __name__ == '__main__':
    main()
