# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

install:
pip install streamlit agentica text2vec sqlalchemy lancedb pyarrow yfinance

run:
streamlit run llm_os_demo.py
"""
import os
import sys
from textwrap import dedent
from typing import List, Optional

import streamlit as st

sys.path.append('..')
from agentica import Agent, OpenAIChat, PythonAgent
from agentica.tools.file_tool import FileTool
from agentica.utils.log import logger
from agentica.tools.search_serper_tool import SearchSerperTool
from agentica.tools.shell_tool import ShellTool
from agentica.knowledge.base import Knowledge
from agentica.vectordb.lancedb_vectordb import LanceDb
from agentica.emb.text2vec_emb import Text2VecEmb
from agentica.tools.search_exa_tool import SearchExaTool
from agentica.tools.yfinance_tool import YFinanceTool
from agentica.storage.agent.sqlite import SqlAgentStorage
from agentica.document import Document


def get_llm_os(
        model,
        google_search: bool = False,
        file_tool: bool = False,
        shell_tool: bool = False,
        python_agent: bool = False,
        research_agent: bool = False,
        investment_agent: bool = False,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        debug_mode: bool = True,
) -> Agent:
    llm_model_name = model.id
    logger.info(f"-*- Creating {llm_model_name} LLM OS -*-")

    # Add tools available to the LLM OS
    tools = []
    extra_instructions: List[str] = []
    if google_search:
        tools.append(SearchSerperTool())
    if file_tool:
        tools.append(FileTool())
        extra_instructions.append(
            "You can use the `read_file` tool to read a file, `save_file` to save a file, and `list_files` to list files in the working directory."
        )
    if shell_tool:
        tools.append(ShellTool())
        extra_instructions.append(
            "You can use the `run_shell_command` tool to run shell commands. For example, `run_shell_command(args='ls')`."
        )

    # Add team members available to the LLM OS
    team: List[Agent] = []
    embedder = Text2VecEmb()
    lance_db = LanceDb(
        uri="outputs/llm_os_lancedb",
        table_name="llm_os_documents",
        embedder=embedder,
    )
    knowledge = None
    if python_agent:
        _python_agent = PythonAgent(
            name="Python agent",
            role="Write and run python code",
            pip_install=True,
            charting_libraries=["streamlit"],
        )
        team.append(_python_agent)
        extra_instructions.append("To write and run python code, delegate the task to the `Python agent`.")
    if research_agent:
        _research_agent = Agent(
            name="Research agent",
            role="Write a research report on a given topic",
            model=model,
            description="You are a Senior New York Times researcher tasked with writing a cover story research report.",
            instructions=[
                "For a given topic, use the `search_exa` to get the top 10 search results.",
                "Carefully read the results and generate a final - NYT cover story worthy report in the <report_format> provided below.",
                "Make your report engaging, informative, and well-structured.",
                "Remember: you are writing for the New York Times, so the quality of the report is important.",
            ],
            expected_output=dedent(
                """\
            An engaging, informative, and well-structured report in the following format:
            <report_format>
            ## Title

            - **Overview** Brief introduction of the topic.
            - **Importance** Why is this topic significant now?

            ### Section 1
            - **Detail 1**
            - **Detail 2**

            ### Section 2
            - **Detail 1**
            - **Detail 2**

            ## Conclusion
            - **Summary of report:** Recap of the key findings from the report.
            - **Implications:** What these findings mean for the future.

            ## References
            - [Reference 1](Link to Source)
            - [Reference 2](Link to Source)
            </report_format>
            """
            ),
            tools=[SearchExaTool(num_results=5, text_length_limit=1000)],
            # This setting tells the LLM to format messages in markdown
            markdown=True,
            add_datetime_to_instructions=True,
            debug_mode=debug_mode,
        )
        team.append(_research_agent)
        extra_instructions.append(
            "To write a research report, delegate the task to the `Research agent`. "
            "Return the report in the <report_format> to the user as is, without any additional text like 'here is the report'."
        )
    if investment_agent:
        _investment_agent = Agent(
            name="Investment agent",
            role="Write a investment report on a given company (stock) symbol",
            model=model,
            description="You are a Senior Investment Analyst for Goldman Sachs tasked with writing an investment report for a very important client.",
            instructions=[
                "For a given stock symbol, get the stock price, company information, analyst recommendations, and company news",
                "Carefully read the research and generate a final - Goldman Sachs worthy investment report in the <report_format> provided below.",
                "Provide thoughtful insights and recommendations based on the research.",
                "When you share numbers, make sure to include the units (e.g., millions/billions) and currency.",
                "REMEMBER: This report is for a very important client, so the quality of the report is important.",
            ],
            expected_output=dedent(
                """\
            <report_format>
            ## [Company Name]: Investment Report

            ### **Overview**
            {give a brief introduction of the company and why the user should read this report}
            {make this section engaging and create a hook for the reader}

            ### Core Metrics
            {provide a summary of core metrics and show the latest data}
            - Current price: {current price}
            - 52-week high: {52-week high}
            - 52-week low: {52-week low}
            - Market Cap: {Market Cap} in billions
            - P/E Ratio: {P/E Ratio}
            - Earnings per Share: {EPS}
            - 50-day average: {50-day average}
            - 200-day average: {200-day average}
            - Analyst Recommendations: {buy, hold, sell} (number of analysts)

            ### Financial Performance
            {analyze the company's financial performance}

            ### Growth Prospects
            {analyze the company's growth prospects and future potential}

            ### News and Updates
            {summarize relevant news that can impact the stock price}

            ### [Summary]
            {give a summary of the report and what are the key takeaways}

            ### [Recommendation]
            {provide a recommendation on the stock along with a thorough reasoning}

            </report_format>
            """
            ),
            tools=[YFinanceTool(stock_price=True, company_info=True, analyst_recommendations=True, company_news=True)],
            # This setting tells the LLM to format messages in markdown
            markdown=True,
            add_datetime_to_instructions=True,
            debug_mode=debug_mode,
        )
        team.append(_investment_agent)
        extra_instructions.extend(
            [
                "To get an investment report on a stock, delegate the task to the `Investment agent`. "
                "Return the report in the <report_format> to the user without any additional text like 'here is the report'.",
                "Answer any questions they may have using the information in the report.",
                "Never provide investment advise without the investment report.",
            ]
        )

    # Create the LLM OS agent
    llm_os = Agent(
        name="llm_os",
        session_id=session_id,
        user_id=user_id,
        model=model,
        description=dedent(
            """\
        You are the most advanced AI system in the world called `LLM-OS`.
        You have access to a set of tools and a team of AI agents at your disposal.
        Your goal is to assist the user in the best way possible.\
        """
        ),
        instructions=[
            "When the user sends a message, first **think** and determine if:\n"
            " - You can answer by using a tool available to you\n"
            " - You need to search the knowledge base\n"
            " - You need to search the internet\n"
            " - You need to delegate the task to a team member\n"
            " - You need to ask a clarifying question",
            "If the user asks about a topic, first ALWAYS search your knowledge base using the `search_knowledge` tool.",
            "If you dont find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If the user asks to summarize the conversation or if you need to reference your chat history with the user, use the `get_chat_history` tool.",
            "If the users message is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
            "You can delegate tasks to an AI agent in your team depending of their role and the tools available to them.",
        ],
        additional_context='\n'.join(extra_instructions),
        # Add long-term memory to the LLM OS backed by a PostgreSQL database
        storage=SqlAgentStorage(table_name="llm_os", db_file="outputs/llm_os.db"),
        # Add a knowledge base to the LLM OS
        knowledge=knowledge if knowledge else Knowledge(vector_db=lance_db),
        # Add selected tools to the LLM OS
        tools=tools,
        # Add selected team members to the LLM OS
        team=team,
        # Show tool calls in the chat
        show_tool_calls=True,
        # This setting gives the LLM a tool to search the knowledge base for information
        search_knowledge=True,
        # This setting gives the LLM a tool to get chat history
        read_chat_history=True,
        # This setting adds chat history to the messages
        add_history_to_messages=True,
        # This setting adds 6 previous messages from chat history to the messages sent to the LLM
        num_history_responses=6,
        # This setting tells the LLM to format messages in markdown
        markdown=True,
        # This setting adds the current datetime to the instructions
        add_datetime_to_instructions=True,
        # Add an introductory agent message
        introduction=dedent(
            """\
        Hi, I'm your LLM OS.
        I have access to a set of tools and AI agents to assist you.
        Let's solve some problems together!\
        """
        ),
        debug_mode=debug_mode,
    )
    return llm_os


def main():
    st.set_page_config(
        page_title="LLM OS",
        page_icon=":orange_heart:",
    )
    st.title("LLM OS")
    st.markdown("##### :orange_heart: built using [agentica](https://github.com/shibing624/agentica)")

    # Get LLM Model
    llm_id = st.sidebar.selectbox("Select LLM", options=["gpt-4o", "gpt-4-turbo"]) or "gpt-4o"
    # Set llm_id in session state
    if "llm_id" not in st.session_state:
        st.session_state["llm_id"] = llm_id
    # Restart the agent if llm_id changes
    elif st.session_state["llm_id"] != llm_id:
        st.session_state["llm_id"] = llm_id
        restart_agent()

    # Sidebar checkboxes for selecting tools
    st.sidebar.markdown("### Select Tools")

    # Enable file tools
    if "file_tool_enabled" not in st.session_state:
        st.session_state["file_tool_enabled"] = True
    # Get file_tools_enabled from session state if set
    file_tool_enabled = st.session_state["file_tool_enabled"]
    # Checkbox for enabling shell tools
    file_tool = st.sidebar.checkbox("File Tool", value=file_tool_enabled, help="Enable file tools.")
    if file_tool_enabled != file_tool:
        st.session_state["file_tool_enabled"] = file_tool
        file_tool_enabled = file_tool
        restart_agent()

    # Enable Web Search via Serper API
    if "google_search_enabled" not in st.session_state:
        st.session_state["google_search_enabled"] = True
    google_search_enabled = st.session_state["google_search_enabled"]
    # Checkbox for enabling web search
    google_search = st.sidebar.checkbox("Web Search", value=google_search_enabled,
                                        help="Enable web search using google.")
    if google_search_enabled != google_search:
        st.session_state["google_search_enabled"] = google_search
        google_search_enabled = google_search
        restart_agent()

    # Enable shell tools
    if "shell_tool_enabled" not in st.session_state:
        st.session_state["shell_tool_enabled"] = False
    shell_tool_enabled = st.session_state["shell_tool_enabled"]
    # Checkbox for enabling shell tools
    shell_tools = st.sidebar.checkbox("Shell Tool", value=shell_tool_enabled, help="Enable shell tool.")
    if shell_tool_enabled != shell_tools:
        st.session_state["shell_tool_enabled"] = shell_tools
        shell_tool_enabled = shell_tools
        restart_agent()

    # Sidebar checkboxes for selecting team members
    st.sidebar.markdown("### Select Team Members")

    # Enable Python agent
    if "python_agent_enabled" not in st.session_state:
        st.session_state["python_agent_enabled"] = False
    # Get python_agent_enabled from session state if set
    python_agent_enabled = st.session_state["python_agent_enabled"]
    # Checkbox for enabling web search
    python_agent = st.sidebar.checkbox(
        "Python agent",
        value=python_agent_enabled,
        help="Enable the Python agent for writing and running python code.",
    )
    if python_agent_enabled != python_agent:
        st.session_state["python_agent_enabled"] = python_agent
        python_agent_enabled = python_agent
        restart_agent()

    # Enable Research agent
    if "research_agent_enabled" not in st.session_state:
        st.session_state["research_agent_enabled"] = False
    # Get research_agent_enabled from session state if set
    research_agent_enabled = st.session_state["research_agent_enabled"]
    # Checkbox for enabling web search
    research_agent = st.sidebar.checkbox(
        "Research agent",
        value=research_agent_enabled,
        help="Enable the research agent (uses Exa).",
    )
    if research_agent_enabled != research_agent:
        st.session_state["research_agent_enabled"] = research_agent
        research_agent_enabled = research_agent
        restart_agent()

    # Enable Investment agent
    if "investment_agent_enabled" not in st.session_state:
        st.session_state["investment_agent_enabled"] = False
    # Get investment_agent_enabled from session state if set
    investment_agent_enabled = st.session_state["investment_agent_enabled"]
    # Checkbox for enabling web search
    investment_agent = st.sidebar.checkbox(
        "Investment agent",
        value=investment_agent_enabled,
        help="Enable the investment agent. NOTE: This is not financial advice.",
    )
    if investment_agent_enabled != investment_agent:
        st.session_state["investment_agent_enabled"] = investment_agent
        investment_agent_enabled = investment_agent
        restart_agent()

    # Get the agent
    llm_os: Agent
    if "llm_os" not in st.session_state or st.session_state["llm_os"] is None:
        logger.info(f"---*--- Creating {llm_id} LLM OS ---*---")
        llm_os = get_llm_os(
            model=OpenAIChat(id=llm_id),
            google_search=google_search_enabled,
            file_tool=file_tool_enabled,
            shell_tool=shell_tool_enabled,
            python_agent=python_agent_enabled,
            research_agent=research_agent_enabled,
            investment_agent=investment_agent_enabled,
        )
        st.session_state["llm_os"] = llm_os
    else:
        llm_os = st.session_state["llm_os"]

    # Create agent run (i.e. log to database) and save session_id in session state
    try:
        st.session_state["llm_os_session_id"] = llm_os.load_session()
    except Exception as e:
        st.warning(f"Could not create LLM OS run, is the database running? error: {e}")
        return

    # Load existing messages
    agent_chat_history = llm_os.memory.get_messages()
    if len(agent_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = agent_chat_history
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
            st.write(message.get("content", ""))

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("agent"):
            response = ""
            resp_container = st.empty()
            for delta in llm_os.run(question, stream=True):
                response += delta.content
                resp_container.markdown(response)
            logger.debug(f"agent response: {response}")
            st.session_state["messages"].append({"role": "assistant", "content": response})

    # Load LLM OS knowledge base
    if llm_os.knowledge:
        # -*- Add websites to knowledge base
        if "url_scrape_key" not in st.session_state:
            st.session_state["url_scrape_key"] = 0

        input_url = st.sidebar.text_input(
            "Add URL to Knowledge Base", type="default", key=st.session_state["url_scrape_key"]
        )
        add_url_button = st.sidebar.button("Add URL")
        input_url = input_url.strip()
        if add_url_button:
            if input_url is not None:
                alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
                if f"{input_url}_scraped" not in st.session_state:
                    web_documents = llm_os.knowledge.read_url(input_url)
                    if web_documents:
                        llm_os.knowledge.load_documents(web_documents, upsert=True)
                    else:
                        st.sidebar.error("Could not read website")
                    st.session_state[f"{input_url}_uploaded"] = True
                alert.empty()

        # Add PDFs to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 100

        file_types = ["pdf", "txt", "md", "docx", "xlsx", "json", "jsonl", "csv", "tsv", "html"]
        uploaded_file = st.sidebar.file_uploader(
            "Add a PDF :page_facing_up:", type=file_types, key=st.session_state["file_uploader_key"]
        )
        if uploaded_file is not None:
            save_pdf_dir = "outputs/pdfs/"
            if not os.path.exists(save_pdf_dir):
                os.makedirs(save_pdf_dir)
            save_file_path = os.path.join(save_pdf_dir, uploaded_file.name)
            with open(save_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"File saved to: {save_file_path}")
            alert = st.sidebar.info("Processing File(eg:pdf/txt/md/...)", icon="🧠")
            auto_rag_name = uploaded_file.name.split(".")[0]
            if f"{auto_rag_name}_uploaded" not in st.session_state:
                auto_rag_documents: List[Document] = llm_os.knowledge.read_file(save_file_path)
                logger.debug(f"auto_rag_documents size: {len(auto_rag_documents)}")
                if auto_rag_documents:
                    llm_os.knowledge.load_documents(auto_rag_documents, upsert=True)
                else:
                    st.sidebar.error(f"Could not read file: {uploaded_file.name}")
                st.session_state[f"{auto_rag_name}_uploaded"] = True
            alert.empty()

    if llm_os.knowledge and llm_os.knowledge.vector_db:
        if st.sidebar.button("Clear Knowledge Base"):
            llm_os.knowledge.vector_db.delete()
            st.sidebar.success("Knowledge base cleared")

    # Show team member memory
    if llm_os.team and len(llm_os.team) > 0:
        for team_member in llm_os.team:
            if len(team_member.memory.messages) > 0:
                with st.status(f"{team_member.name} Memory", expanded=False, state="complete"):
                    with st.container():
                        _team_member_memory_container = st.empty()
                        _team_member_memory_container.json(team_member.memory.get_messages())

    if llm_os.storage:
        llm_os_session_ids: List[str] = llm_os.storage.get_all_session_ids()
        new_llm_os_session_id = st.sidebar.selectbox("Session ID", options=llm_os_session_ids)
        if st.session_state["llm_os_session_id"] and st.session_state["llm_os_session_id"] != new_llm_os_session_id:
            logger.info(f"---*--- Loading {llm_id} run: {new_llm_os_session_id} ---*---")
            st.session_state["llm_os"] = get_llm_os(
                model=OpenAIChat(id=llm_id),
                google_search=google_search_enabled,
                file_tool=file_tool_enabled,
                shell_tool=shell_tool_enabled,
                python_agent=python_agent_enabled,
                research_agent=research_agent_enabled,
                investment_agent=investment_agent_enabled,
                session_id=new_llm_os_session_id,
            )
            st.rerun()

    if st.sidebar.button("New Run"):
        restart_agent()


def restart_agent():
    logger.debug("---*--- Restarting agent ---*---")
    st.session_state["llm_os"] = None
    st.session_state["llm_os_session_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()


if __name__ == '__main__':
    main()
