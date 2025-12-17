import os
import traceback
import uuid

from bson import ObjectId
from sentry_sdk import capture_exception

from eve.agent.llm.llm import async_prompt, get_provider
from eve.agent.session.models import (
    ChatMessage,
    LLMConfig,
    LLMContext,
    LLMContextMetadata,
    LLMTraceMetadata,
    Session,
)


async def async_title_session(session: Session, initial_message_content: str):
    """
    Generate a title for a session based on the initial message content
    """

    from pydantic import BaseModel, Field

    class TitleResponse(BaseModel):
        """A title for a session of chat messages. It must entice a user to click on the session when they are interested in the subject."""

        title: str = Field(
            description="a phrase of 2-5 words (or up to 30 characters) that conveys the subject of the chat session. It should be concise and terse, and not include any special characters or punctuation."
        )

    try:
        if not initial_message_content:
            # If no message content, return without setting a title
            return

        # Add a system message and the initial user message for title generation
        system_message = ChatMessage(
            session=[session.id],
            sender=ObjectId("000000000000000000000000"),  # System sender
            role="system",
            content="You are an expert at creating concise titles for chat sessions.",
        )

        # Add the initial user message
        user_message = ChatMessage(
            session=[session.id],
            sender=ObjectId("000000000000000000000000"),  # System sender (placeholder)
            role="user",
            content=initial_message_content,
        )

        # Add request message for title generation
        request_message = ChatMessage(
            session=[session.id],
            sender=ObjectId("000000000000000000000000"),  # System sender
            role="user",
            content="Come up with a title for this session based on the user's message.",
        )

        # Build message list
        messages = [system_message, user_message, request_message]

        # Create LLM context
        llm_context = LLMContext(
            messages=messages,
            tools=[],
            config=LLMConfig(
                model="gpt-5-nano",
                fallback_models=["gpt-4o-mini"],
                response_format=TitleResponse,
            ),
            metadata=LLMContextMetadata(
                session_id=f"{os.getenv('DB')}-{str(session.id)}",
                trace_name="FN_title_session",
                trace_id=str(uuid.uuid4()),
                generation_name="FN_title_session",
                trace_metadata=LLMTraceMetadata(
                    session_id=str(session.id),
                ),
            ),
            enable_tracing=False,
        )

        # Generate title using async_prompt
        provider = get_provider(llm_context)
        if provider is None:
            raise RuntimeError("No LLM provider available for session titling")

        result = await async_prompt(llm_context, provider)

        # Parse the response
        if hasattr(result, "content") and result.content:
            try:
                # Try to parse as JSON if response_format was used
                import json

                title_data = json.loads(result.content)
                if isinstance(title_data, dict) and "title" in title_data:
                    # session.title = title_data["title"]
                    session.update(title=title_data["title"])
                else:
                    # Fallback to using content directly
                    # session.title = result.content[:30]  # Limit to 30 chars
                    session.update(title=result.content[:30])
            except (json.JSONDecodeError, TypeError):
                # If JSON parsing fails, use content directly
                # session.title = result.content[:30]  # Limit to 30 chars
                session.update(title=result.content[:30])

            # session.save()

    except Exception as e:
        capture_exception(e)
        traceback.print_exc()
        return
