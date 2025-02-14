import json
import requests
from pydantic import Field, BaseModel
from typing import Dict, Any, Optional, List
from bson import ObjectId
from eve.mongo import Document, Collection
from instructor.function_calls import openai_schema

import anthropic

MAX_MEMORY_WORDS = 200
EMPTY_MEMORY_STRING = "[Empty agent memory]"

# Define the Memory document model
@Collection("memory")
class AgentMemory(Document):
    userId: ObjectId
    agentId: ObjectId
    content: str = Field(default="")
    max_mem_words: int = Field(default=MAX_MEMORY_WORDS)

    @classmethod
    def find(cls, query: Dict[str, Any]) -> Optional["AgentMemory"]:
        """
        Find documents in the memory collection based on query parameters.
        
        Args:
            query (Dict[str, Any]): MongoDB query dictionary
            
        Returns:
            Optional[AgentMemory]: Matching AgentMemory document or None
        """
        try:
            return cls.load(**query)
        except Exception as e:
            print("No memory found!")
            print(str(e))
            return None


class MemoryResponse(BaseModel):
    """Response model for Agent memory update command"""
    updated_memory: str = Field(..., description="The updated agent memory")

async def get_or_create_memory(userId: ObjectId, agentId: ObjectId) -> AgentMemory:
    """
    Retrieve or create an agent's memory for a specific user
    
    Returns:
        AgentMemory: The existing or newly created memory document
    """
    # Try to find existing memory
    memory = AgentMemory.find({"userId": userId, "agentId": agentId})
    
    if memory:
        print("Found existing memory!")
        return memory
    
    # Create new memory if none exists
    print("Creating new memory for agent/user combo!")
    memory = AgentMemory(
        userId=userId,
        agentId=agentId,
        content=EMPTY_MEMORY_STRING,
        max_mem_words=MAX_MEMORY_WORDS
    )
    memory.save()
    return memory

async def update_memory_with_llm(
    current_memory: str,
    instruction: str,
    max_mem_words: int
) -> str:
    """
    Use LLM to update agent memory based on instruction
    """
    client = anthropic.AsyncAnthropic()

    prompt_parts = [
        "You are an expert at managing an AI agent's memory. Your task is to update the agent's memory based on new instructions while ensuring the memory stays concise and within the maximum word limit.",
        "The agent memory is a simple, written document that will be appended to the instruction context. Format the memory as instructions directed at the agents desired behavior.",
        "Important context:",
        "- Agent memory is for permanent, long-term behavioral changes and core preferences",
        "- Memory should only contain instructions that affect future interactions",
        "- Avoid storing temporary context, conversation-specific details, or factual information from a users personal background",
        "- Each memory entry should represent a meaningful modification to default behavior",
        "Technical rules:",
        "- If the instruction adds new information, integrate it naturally into existing memory",
        "- If the instruction removes information, update accordingly", 
        "- If information already exists, avoid duplicate entries",
        "- If new instructions conflict with existing memory contents, they take precedence and overwrite the memory",
        f"- Total memory must not exceed {max_mem_words} words, summarize or prioritize information if needed",
        "- Maintain clear and organized structure",
        "Memory format guidelines:",
        "- Format each entry as a clear behavioral directive",
        "- Focus on permanent changes that affect all future interactions",
        "- Exclude temporary role-play or context-specific instructions",
        "- Prioritize unique, significant behavior modifications",
        "\n",
        "Current memory:",
        current_memory or EMPTY_MEMORY_STRING,
        "Instruction:",
        instruction
    ]

    messages = [{"role": "user", "content": "\n\n".join(prompt_parts)}]

    print("---------------------------------------------------------------------")
    print("LLM Prompt:")
    print("\n\n".join(prompt_parts))
    print("---------------------------------------------------------------------")

    prompt = {
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 2048,
        "messages": messages,
        "system": "You are an expert at managing agent memories. Respond with a JSON object containing just the updated memory content.",
        "tools": [openai_schema(MemoryResponse).anthropic_schema],
        "tool_choice": {"type": "tool", "name": "MemoryResponse"}
    }
    
    response = await client.messages.create(**prompt)
    
    if not response.content:
        raise ValueError("Empty response from Anthropic API")
        
    return MemoryResponse(**response.content[0].input).updated_memory

async def handler(args: dict):
    """
    Handle memory update requests
    """
    if "instruction" not in args:
        raise ValueError("Instruction is required")
    
    if "userId" not in args or "agentId" not in args:
        raise ValueError("Both userId and agentId are required")

    instruction = args["instruction"]
    userId = ObjectId(args["userId"])
    agentId = ObjectId(args["agentId"])

    print("\nUpdating agent memory with instruction:")
    print(instruction)

    # Get or create memory document
    memory = await get_or_create_memory(userId, agentId)
    
    print("\nCurrent memory:")
    print(memory.content or EMPTY_MEMORY_STRING)

    # Update memory using LLM
    updated_content = await update_memory_with_llm(
        memory.content,
        instruction,
        memory.max_mem_words
    )

    # Save updated memory
    memory.update(content=updated_content)
    
    print("\nUpdated memory:")
    print(updated_content)

    return {
        "output": "Agent memory updated successfully!",
        "memory": updated_content
    }