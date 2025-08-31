from pydantic import BaseModel, Field
from sentry_sdk import capture_exception
import json
import asyncio
import traceback
from typing import Optional, Dict

from eve.agent.session.session_llm import async_prompt
from eve.agent.session.models import LLMContext, LLMConfig, ChatMessage, Session


async def async_title_session(
    session: Session
):
    """
    Generate a title for a session
    """

    class TitleResponse(BaseModel):
        """A title for a session of chat messages. It must entice a user to click on the session when they are interested in the subject."""

        title: str = Field(
            description="a phrase of 2-5 words (or up to 30 characters) that conveys the subject of the chat session. It should be concise and terse, and not include any special characters or punctuation."
        )

    # Get session messages
    chat_messages = [ChatMessage(role="system", content="You are an expert at creating concise titles for chat sessions.")]
    
    # Add session messages
    session_messages = list(ChatMessage.find({"session": session.id}))[:50]
    for msg in session_messages:
        if msg.role in ["user", "assistant"]:
            chat_messages.append(msg)
    
    # Add final request for title
    chat_messages.append(ChatMessage(
        role="user", 
        content="Come up with a title for this session."
    ))

    # Create LLM context
    context = LLMContext(
        messages=chat_messages,
        config=LLMConfig(
            model="gpt-4o-mini",
            response_format=TitleResponse
        )
    )

    try:
        result = await async_prompt(context)
        if result.content:
            try:
                title_data = json.loads(result.content)
                session.title = title_data.get('title', result.content)
            except Exception as e:
                print(f"Error parsing title: {e}")
                session.title = "Untitled Session"
        else:
            session.title = "Untitled Session"
        
        session.save()

    except Exception as e:
        capture_exception(e)
        traceback.print_exc()
        return


def title_session(session: Session):
    return asyncio.run(async_title_session(session))


def test_title_session():
    """Test entry point"""
    session = Session.from_mongo("68b3f2c5c548dd6adc6f70c1")
    result = asyncio.run(async_title_session(session=session))
    
    print("================================================")
    print(f"Session title: {session.title}")
    print("================================================")


if __name__ == "__main__":
    test_title_session()