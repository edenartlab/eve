"""
Memory System v2 - RAG Tool

This module provides the tool definition for agents to search long-term memory.
The tool can be added to an agent's tool list to enable explicit memory retrieval.

This is Option A from the design doc - a tool call interface where the agent
decides when to query memory and what to search for.
"""

from typing import Any, Dict, List, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.memory2.constants import LOCAL_DEV, RAG_ENABLED, RAG_TOP_K
from eve.agent.memory2.rag import format_facts_for_tool_response, search_facts


# Tool definition for agent
MEMORY_SEARCH_TOOL = {
    "name": "search_memory",
    "description": """Search long-term memory for facts about users, projects, events, or any stored information.

Use this tool when you need to recall specific facts that may not be in your immediate context, such as:
- User preferences, birthdays, or personal information
- Project details, deadlines, or status
- Past conversations or decisions
- Any factual information that was stored

The search uses semantic similarity, so you can describe what you're looking for in natural language.""",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory. Be specific about what information you need.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of facts to return (default: 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


async def handle_memory_search(
    query: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    max_results: int = 5,
) -> str:
    """
    Handle a memory search tool call from the agent.

    Args:
        query: Search query
        agent_id: Agent ID
        user_id: User ID (for user-scoped facts)
        max_results: Maximum results to return

    Returns:
        Formatted string response for the agent
    """
    if not RAG_ENABLED:
        return "Memory search is currently disabled."

    try:
        facts = await search_facts(
            query=query,
            agent_id=agent_id,
            user_id=user_id,
            match_count=min(max_results, RAG_TOP_K),
        )

        return format_facts_for_tool_response(facts)

    except Exception as e:
        logger.error(f"Error in handle_memory_search: {e}")
        return f"Error searching memory: {str(e)}"


class MemorySearchTool:
    """
    Tool class for memory search that can be integrated into the agent's tool system.

    This provides a standard interface for the agent to search long-term memory.
    """

    def __init__(
        self,
        agent_id: ObjectId,
        user_id: Optional[ObjectId] = None,
    ):
        """
        Initialize the memory search tool.

        Args:
            agent_id: Agent ID
            user_id: User ID (optional, for user-scoped facts)
        """
        self.agent_id = agent_id
        self.user_id = user_id
        self.name = "search_memory"
        self.description = MEMORY_SEARCH_TOOL["description"]
        self.parameters = MEMORY_SEARCH_TOOL["parameters"]

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        **kwargs,
    ) -> str:
        """
        Execute a memory search.

        Args:
            query: Search query
            max_results: Maximum results
            **kwargs: Additional arguments (ignored)

        Returns:
            Formatted search results
        """
        return await handle_memory_search(
            query=query,
            agent_id=self.agent_id,
            user_id=self.user_id,
            max_results=max_results,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for tool registration.

        Returns:
            Tool definition dictionary
        """
        return MEMORY_SEARCH_TOOL.copy()


def get_memory_tool(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
) -> MemorySearchTool:
    """
    Get a memory search tool instance for an agent.

    Args:
        agent_id: Agent ID
        user_id: User ID (optional)

    Returns:
        MemorySearchTool instance
    """
    return MemorySearchTool(agent_id, user_id)


def get_memory_tool_definition() -> Dict[str, Any]:
    """
    Get the tool definition for memory search.

    Returns:
        Tool definition dictionary
    """
    return MEMORY_SEARCH_TOOL.copy()


async def proactive_memory_retrieval(
    user_message: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    threshold: float = 0.7,
    max_facts: int = 3,
) -> Optional[str]:
    """
    Proactively retrieve relevant facts based on user message.

    This is Option B from the design doc - a background subagent approach
    where memory is retrieved automatically without explicit tool calls.

    The returned facts can be injected into the agent's context alongside
    the always-in-context memory.

    Args:
        user_message: The user's message to analyze
        agent_id: Agent ID
        user_id: User ID
        threshold: Minimum relevance threshold (0-1)
        max_facts: Maximum facts to return

    Returns:
        Formatted facts string if relevant facts found, None otherwise
    """
    if not RAG_ENABLED:
        return None

    try:
        # Search for relevant facts
        facts = await search_facts(
            query=user_message,
            agent_id=agent_id,
            user_id=user_id,
            match_count=max_facts,
        )

        if not facts:
            return None

        # Filter by relevance threshold
        relevant_facts = [
            f for f in facts
            if (f.get("rrf_score") or f.get("score", 0)) >= threshold
        ]

        if not relevant_facts:
            return None

        # Format for context injection
        fact_lines = []
        for fact in relevant_facts:
            scope_str = ", ".join(fact.get("scope", []))
            fact_lines.append(f"- [{scope_str}] {fact['content']}")

        return "\n".join(fact_lines)

    except Exception as e:
        logger.error(f"Error in proactive_memory_retrieval: {e}")
        return None


def build_rag_context_section(facts_content: str) -> str:
    """
    Build an XML section for RAG-retrieved facts.

    Args:
        facts_content: Formatted facts string

    Returns:
        XML-formatted section for context injection
    """
    if not facts_content:
        return ""

    return f"""<RetrievedFacts>
{facts_content}
</RetrievedFacts>"""
