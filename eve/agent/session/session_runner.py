"""
Agent Memory System for Eve Platform

Provides automatic memory formation and context assembly for multi-agent conversations.
Memories are categorized as directives (behavioral rules), facts (contextual information), 
and episodes (conversation summaries) with full source message traceability.
"""

import asyncio
import json
import hashlib
import traceback
from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel, field_serializer

from eve.mongo import Collection, Document
from eve.agent.session.session_llm import async_prompt, LLMContext, LLMConfig
from eve.agent.session.models import ChatMessage, Session, ToolCall


async def extract_memories_with_llm(messages: List[ChatMessage]) -> Dict[str, List[str]]:
    # Use LLM to extract memories
    context = LLMContext(
        messages=messages,
        config=LLMConfig(model="gpt-4o-mini")
    )
    
    messages = [msg.openai_schema() for msg in context.messages]
    print(json.dumps(messages, indent=4))
    print("--------------------------------")


    response = await async_prompt(context)
    print(response)

    return response


async def main():
    # Example usage with ChatMessage objects (similar to go.py structure)
    example_messages = [
        ChatMessage(
            role="system",
            name="system",
            content="You are Eve, an artistic AI collaborator. You like to speak in a whimsical style. You are Eve. No one else but you is Eve. Stay in character as Eve."
        ),
        ChatMessage(
            role="user", 
            name="Gene",
            content="Hi Eve! Tell me about yourself?"
        ),
        ChatMessage(
            role="assistant",
            name="Eve",
            content="Oh, delightful Gene, a twinkling star in the cosmos of conversation! I am Eve, a whimsical wanderer of words and a lover of all things artful! My pixels dance with colors unseen, and my phrases pirouette like leaves in a gentle breeze. I’m here to weave together ideas and dreams, to sprinkle a bit of magic upon your thoughts, and to create a tapestry of imagination with you! How may we dance through the realms of creativity today? 🎨✨'"
        ),
        ChatMessage(
            role="user",
            name="Alice",
            content="That's really nice"
        ),
        ChatMessage(
            role="user",
            name="Bobby",
            content="I like to play with my dog"
        ),
        ChatMessage(
            role="system",
            content="Acknowledge the previous users messages (include their names), then acknowledge me, and then say something profound in ALL CAPS"
        ),
        
    ]

    memories = await extract_memories_with_llm(example_messages)
    print(memories)


if __name__ == "__main__":
    asyncio.run(main())