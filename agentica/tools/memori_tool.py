import json
from typing import Any, Dict, List, Optional

from agentica.tools.base import Tool
from agentica.utils.log import logger
from agentica.agent import Agent

try:
    from memori import Memori, create_memory_tool
except ImportError:
    raise ImportError("`memorisdk` package not found. Please install it with `pip install memorisdk`")


class MemoriTool(Tool):
    """
    Memori ToolKit for agentica Agents and Teams, providing persistent memory capabilities.

    This toolkit integrates Memori's memory system with agentica, allowing Agents and Teams to:
    - Store and retrieve conversation history
    - Search through past interactions
    - Maintain user preferences and context
    - Build long-term memory across sessions

    Requirements:
        - pip install memorisdk
        - Database connection string (SQLite, PostgreSQL, etc.)

    Example:
        ```python
        from agentica.tools.memori_tool import MemoriTool

        # Initialize with SQLite (default)
        memori_tool = MemoriTool(
            database_connect="sqlite:///agent_memory.db",
            namespace="my_agent",
            auto_ingest=True  # Automatically ingest conversations
        )

        # Add to agent
        agent = Agent(
            model=OpenAIChat(),
            tools=[memori_tool],
            description="An AI assistant with persistent memory"
        )
        ```
    """

    def __init__(
            self,
            database_connect: Optional[str] = None,
            namespace: Optional[str] = None,
            conscious_ingest: bool = True,
            auto_ingest: bool = True,
            verbose: bool = False,
            config: Optional[Dict[str, Any]] = None,
            auto_enable: bool = True,
            enable_search_memory: bool = True,
            enable_record_conversation: bool = True,
            enable_get_memory_stats: bool = True,
            all: bool = False,
            **kwargs,
    ):
        """
        Initialize Memori toolkit.

        Args:
            database_connect: Database connection string (e.g., "sqlite:///memory.db")
            namespace: Namespace for organizing memories (e.g., "agent_v1", "user_session")
            conscious_ingest: Whether to use conscious memory ingestion
            auto_ingest: Whether to automatically ingest conversations into memory
            verbose: Enable verbose logging from Memori
            config: Additional Memori configuration
            auto_enable: Automatically enable the memory system on initialization
            **kwargs: Additional arguments passed to Toolkit base class
        """
        super().__init__(name="memori_tool")
        tools: List[Any] = []
        if all or enable_search_memory:
            self.register(self.search_memory)
        if all or enable_record_conversation:
            self.register(self.record_conversation)
        if all or enable_get_memory_stats:
            self.register(self.get_memory_stats)

        # Set default database connection if not provided
        if not database_connect:
            sqlite_db = "sqlite:///memori_memory.db"
            logger.info(f"No database connection provided, using default SQLite database at {sqlite_db}")
            database_connect = sqlite_db

        self.database_connect = database_connect
        self.namespace = namespace or "agentica_default"
        self.conscious_ingest = conscious_ingest
        self.auto_ingest = auto_ingest
        self.verbose = verbose
        self.config = config or {}

        try:
            # Initialize Memori memory system
            logger.debug(f"Initializing Memori with database: {self.database_connect}")
            self.memory_system = Memori(
                database_connect=self.database_connect,
                conscious_ingest=self.conscious_ingest,
                auto_ingest=self.auto_ingest,
                verbose=self.verbose,
                namespace=self.namespace,
                **self.config,
            )

            # Enable the memory system if auto_enable is True
            if auto_enable:
                self.memory_system.enable()
                logger.debug("Memori memory system enabled")

            # Create the memory tool for internal use
            self._memory_tool = create_memory_tool(self.memory_system)

        except Exception as e:
            logger.error(f"Failed to initialize Memori: {e}")
            raise ConnectionError("Failed to initialize Memori memory system") from e

    def search_memory(
            self,
            agent: Agent,
            query: str,
            limit: Optional[int] = None,
    ) -> str:
        """
        Search the Agent's memory for past conversations and information.

        This performs semantic search across all stored memories to find
        relevant information based on the provided query.

        Args:
            query: What to search for in memory (e.g., "past conversations about AI", "user preferences")
            limit: Maximum number of results to return (optional)

        Returns:
            str: JSON-encoded search results or error message

        Example:
            search_memory("user's favorite programming languages")
            search_memory("previous discussions about machine learning")
        """
        try:
            if not query.strip():
                return json.dumps({"error": "Please provide a search query"})

            logger.debug(f"Searching memory for: {query}")

            # Execute search using Memori's memory tool
            result = self._memory_tool.execute(query=query.strip())

            if result:
                # If limit is specified, truncate results
                if limit and isinstance(result, list):
                    result = result[:limit]

                return json.dumps(
                    {
                        "success": True,
                        "query": query,
                        "results": result,
                        "count": len(result) if isinstance(result, list) else 1,
                    }, ensure_ascii=False
                )
            else:
                return json.dumps(
                    {
                        "success": True,
                        "query": query,
                        "results": [],
                        "count": 0,
                        "message": "No relevant memories found",
                    }, ensure_ascii=False
                )

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return json.dumps({"success": False, "error": f"Memory search error: {str(e)}"})

    def record_conversation(self, agent: Agent, content: str) -> str:
        """
        Add important information or facts to memory.

        Use this tool to store important information, user preferences, facts, or context that should be remembered
        for future conversations.

        Args:
            content: The information/facts to store in memory

        Returns:
            str: Success message or error details

        Example:
            record_conversation("User prefers Python over JavaScript")
            record_conversation("User is working on an e-commerce project using Django")
            record_conversation("User's name is John and they live in NYC")
        """
        try:
            if not content.strip():
                return json.dumps({"success": False, "error": "Content cannot be empty"})

            logger.debug(f"Adding conversation: {content}")

            # Extract the actual AI response from the agent's conversation history
            ai_output = "I've noted this information and will remember it."

            self.memory_system.record_conversation(user_input=content, ai_output=str(ai_output))
            return json.dumps(
                {
                    "success": True,
                    "message": "Memory added successfully via conversation recording",
                    "content_length": len(content),
                }
            )

        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            return json.dumps({"success": False, "error": f"Failed to add memory: {str(e)}"})

    def get_memory_stats(
            self,
            agent: Agent,
    ) -> str:
        """
        Get statistics about the memory system.

        Returns information about the current state of the memory system,
        including total memories, memory distribution by retention type
        (short-term vs long-term), and system configuration.

        Returns:
            str: JSON-encoded memory statistics

        Example:
            Returns statistics like:
            {
                "success": true,
                "total_memories": 42,
                "memories_by_retention": {
                    "short_term": 5,
                    "long_term": 37
                },
                "namespace": "my_agent",
                "conscious_ingest": true,
                "auto_ingest": true,
                "memory_system_enabled": true
            }
        """
        try:
            logger.debug("Retrieving memory statistics")

            # Base stats about the system configuration
            stats = {
                "success": True,
                "namespace": self.namespace,
                "database_connect": self.database_connect,
                "conscious_ingest": self.conscious_ingest,
                "auto_ingest": self.auto_ingest,
                "verbose": self.verbose,
                "memory_system_enabled": hasattr(self.memory_system, "_enabled") and self.memory_system._enabled,
            }

            # Get Memori's built-in memory statistics
            try:
                if hasattr(self.memory_system, "get_memory_stats"):
                    # Use the get_memory_stats method as shown in the example
                    memori_stats = self.memory_system.get_memory_stats()

                    # Add the Memori-specific stats to our response
                    if isinstance(memori_stats, dict):
                        # Include total memories
                        if "total_memories" in memori_stats:
                            stats["total_memories"] = memori_stats["total_memories"]

                        # Include memory distribution by retention type
                        if "memories_by_retention" in memori_stats:
                            stats["memories_by_retention"] = memori_stats["memories_by_retention"]

                            # Also add individual counts for convenience
                            retention_info = memori_stats["memories_by_retention"]
                            stats["short_term_memories"] = retention_info.get("short_term", 0)
                            stats["long_term_memories"] = retention_info.get("long_term", 0)

                        # Include any other available stats
                        for key, value in memori_stats.items():
                            if key not in stats:
                                stats[key] = value

                    logger.debug(
                        f"Retrieved memory stats: total={stats.get('total_memories', 0)}, "
                        f"short_term={stats.get('short_term_memories', 0)}, "
                        f"long_term={stats.get('long_term_memories', 0)}"
                    )

                else:
                    logger.debug("get_memory_stats method not available, providing basic stats only")
                    stats["total_memories"] = 0
                    stats["memories_by_retention"] = {"short_term": 0, "long_term": 0}
                    stats["short_term_memories"] = 0
                    stats["long_term_memories"] = 0

            except Exception as e:
                logger.debug(f"Could not retrieve detailed memory stats: {e}")
                # Provide basic stats if detailed stats fail
                stats["total_memories"] = 0
                stats["memories_by_retention"] = {"short_term": 0, "long_term": 0}
                stats["short_term_memories"] = 0
                stats["long_term_memories"] = 0
                stats["stats_warning"] = "Detailed memory statistics not available"

            return json.dumps(stats)

        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return json.dumps({"success": False, "error": f"Failed to get memory statistics: {str(e)}"})

    def enable_memory_system(self) -> bool:
        """Enable the Memori memory system."""
        try:
            self.memory_system.enable()
            logger.debug("Memori memory system enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable memory system: {e}")
            return False

    def disable_memory_system(self) -> bool:
        """Disable the Memori memory system."""
        try:
            if hasattr(self.memory_system, "disable"):
                self.memory_system.disable()
                logger.debug("Memori memory system disabled")
                return True
            else:
                logger.warning("Memory system disable method not available")
                return False
        except Exception as e:
            logger.error(f"Failed to disable memory system: {e}")
            return False


def create_memori_search_tool(memori_toolkit: MemoriTool):
    """
    Create a standalone memory search function for use with agentica agents.

    This is a convenience function that creates a memory search tool similar
    to the pattern shown in the Memori example code.

    Args:
        memori_toolkit: An initialized MemoriTool instance

    Returns:
        Callable: A memory search function that can be used as an agent tool
    """

    def search_memory(query: str) -> str:
        """
        Search the agent's memory for past conversations and information.

        Args:
            query: What to search for in memory

        Returns:
            str: Search results or error message
        """
        try:
            if not query.strip():
                return "Please provide a search query"

            result = memori_toolkit._memory_tool.execute(query=query.strip())
            return str(result) if result else "No relevant memories found"

        except Exception as e:
            return f"Memory search error: {str(e)}"

    return search_memory


if __name__ == '__main__':
    # add demo
    from agentica import OpenAIChat

    m = MemoriTool(enable_search_memory=True,
                   enable_record_conversation=True,
                   enable_get_memory_stats=True, )
    agent = Agent(
        model=OpenAIChat(),
        tools=[m],
        description="An AI assistant with persistent memory"
    )
    agent.print_response("I'm a Python developer and I love building web applications")

    # Thanks to the Memori ToolKit, your Agent can now remember the conversation:
    agent.print_response("What do you remember about my programming background?")

    # Using the Memori ToolKit, your Agent also gains access to memory statistics:
    agent.print_response("Show me your memory statistics")
