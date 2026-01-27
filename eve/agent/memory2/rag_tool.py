"""
Memory System v2 - RAG Tool (DEPRECATED)

NOTE: RAG retrieval is now implemented as a separate tool call in the agent stack.
This module is kept for backward compatibility but should not be used for new code.

The FIFO facts system provides recent facts in context automatically.
For explicit memory search, use the RAG tool implemented in the agent stack.
"""

from typing import Any, Dict, List, Optional

from bson import ObjectId
from loguru import logger


# Tool definition kept for backward compatibility
MEMORY_SEARCH_TOOL = {
    "name": "search_memory",
    "description": """Search long-term memory for facts about users, projects, events, or any stored information.

NOTE: This tool is deprecated. RAG retrieval is now handled by the agent stack.""",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory.",
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
    DEPRECATED: RAG retrieval is now implemented in the agent stack.

    This function returns a message indicating the tool is deprecated.
    """
    return "Memory search via this interface is deprecated. RAG retrieval is now handled by the agent stack."


class MemorySearchTool:
    """
    DEPRECATED: Tool class kept for backward compatibility.

    RAG retrieval is now implemented in the agent stack.
    """

    def __init__(
        self,
        agent_id: ObjectId,
        user_id: Optional[ObjectId] = None,
    ):
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
        return await handle_memory_search(
            query=query,
            agent_id=self.agent_id,
            user_id=self.user_id,
            max_results=max_results,
        )

    def to_dict(self) -> Dict[str, Any]:
        return MEMORY_SEARCH_TOOL.copy()


def get_memory_tool(
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
) -> MemorySearchTool:
    """DEPRECATED: Returns a deprecated MemorySearchTool instance."""
    return MemorySearchTool(agent_id, user_id)


def get_memory_tool_definition() -> Dict[str, Any]:
    """DEPRECATED: Returns the deprecated tool definition."""
    return MEMORY_SEARCH_TOOL.copy()


async def proactive_memory_retrieval(
    user_message: str,
    agent_id: ObjectId,
    user_id: Optional[ObjectId] = None,
    threshold: float = 0.7,
    max_facts: int = 3,
) -> Optional[str]:
    """
    DEPRECATED: Proactive memory retrieval is no longer used.

    RAG retrieval is now implemented in the agent stack.
    Returns None to indicate no facts retrieved.
    """
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
