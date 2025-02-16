import json
import requests
from pydantic import Field, BaseModel
from typing import Dict, Any, Optional, List
from bson import ObjectId
from eve.mongo import Document, Collection
from instructor.function_calls import openai_schema

import anthropic

MAX_MEMORY_WORDS = 200
LLM_CALL_MAX_TOKENS = MAX_MEMORY_WORDS*2
EMPTY_MEMORY_STRING = "[Empty agent memory]"

@openai_schema
class MemoryResponse(BaseModel):
    """Schema for memory update response from LLM."""
    updated_memory: str = Field(
        description="The updated memory content that incorporates the new instruction"
    )

# Define the Memory document model
@Collection("memory")
class AgentMemory(Document):
    userId: ObjectId
    agentId: ObjectId
    content: str = Field(default="")
    max_mem_words: int = Field(default=MAX_MEMORY_WORDS)

    @classmethod
    def find(cls, query: Dict[str, Any]) -> Optional["AgentMemory"]:
        """Find documents in the memory collection based on query parameters."""
        try:
            return cls.load(**query)
        except Exception as e:
            print(f"Error finding memory: {str(e)}")
            return None

async def get_or_create_memory(userId: ObjectId, agentId: ObjectId) -> AgentMemory:
    """Retrieve or create an agent's memory for a specific user."""
    try:
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
    except Exception as e:
        raise RuntimeError(f"Failed to get or create memory: {str(e)}")

async def update_memory_with_llm(
    current_memory: str,
    instruction: str,
    max_mem_words: int
) -> str:
    """Use LLM to update agent memory based on instruction."""
    if not instruction.strip():
        raise ValueError("Instruction cannot be empty")
        
    try:
        client = anthropic.AsyncAnthropic()
        
        # Sanitize inputs
        current_memory = current_memory.strip() if current_memory else EMPTY_MEMORY_STRING
        instruction = instruction.strip()

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
            "- Never hallucinate plausible memories that were not specified in the instructions"
            f"- Total memory must not exceed {max_mem_words} words, summarize or prioritize information if needed",
            "- Maintain clear and organized structure",
            "Memory format guidelines:",
            "- Format each entry as a clear behavioral directive",
            "- Focus on permanent changes that affect all future interactions",
            "- Exclude temporary role-play or context-specific instructions",
            "- Prioritize unique, significant behavior modifications",
            "\n",
            "Current memory contents:",
            current_memory,
            "Instruction:",
            instruction
        ]

        messages = [{"role": "user", "content": "\n".join(prompt_parts)}]

        print("---------------------------------------------------------------------")
        print("LLM Prompt:")
        print("\n".join(prompt_parts))
        print("---------------------------------------------------------------------")

        prompt = {
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": LLM_CALL_MAX_TOKENS,
            "messages": messages,
            "system": "You are an expert at managing agent memories. Respond with a JSON object containing just the updated memory content.",
            "tools": [openai_schema(MemoryResponse).anthropic_schema],
            "tool_choice": {"type": "tool", "name": "MemoryResponse"}
        }
        
        response = await client.messages.create(**prompt)
        
        if not response.content:
            raise ValueError("Empty response from Anthropic API")
            
        return MemoryResponse(**response.content[0].input).updated_memory

    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to update memory with LLM: {str(e)}")

async def handler(args: dict):
    """Handle memory update requests."""
    memory_content = EMPTY_MEMORY_STRING

    try:
        # Input validation
        if not isinstance(args, dict):
            raise ValueError("Args must be a dictionary")
            
        if "instruction" not in args or not args["instruction"].strip():
            raise ValueError("Valid instruction is required")
        
        if "userId" not in args or "agentId" not in args:
            raise ValueError("Both userId and agentId are required")

        # Validate ObjectId format
        try:
            userId = ObjectId(args["userId"])
            agentId = ObjectId(args["agentId"])
        except Exception:
            raise ValueError("Invalid userId or agentId format")

        instruction = args["instruction"].strip()

        print("\nUpdating agent memory with instruction:")
        print(instruction)

        # Get or create memory document
        memory = await get_or_create_memory(userId, agentId)
        memory_content = memory.content

        # Update memory using LLM
        updated_memory_content = await update_memory_with_llm(
            memory.content,
            instruction,
            memory.max_mem_words
        )

        # Save updated memory
        try:
            memory.update(content=updated_memory_content)
            memory_content = memory.content
            print("\nUpdated memory:")
            print(memory.content)
        except Exception as e:
            raise RuntimeError(f"Failed to save updated memory: {str(e)}")
        
        return {
            "output": memory_content
        }
        
    except ValueError as e:
        return {
            "output": memory_content,
            "error": f"Memory update failed. Invalid input: {str(e)}",
            "status": 400
        }
    except RuntimeError as e:
        return {
            "output": memory_content,
            "error": f"Memory update failed. Operation failed: {str(e)}",
            "status": 500
        }
    except Exception as e:
        return {
            "output": memory_content,
            "error": f"Memory update failed. Unexpected error: {str(e)}",
            "status": 500
        }