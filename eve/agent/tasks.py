from pydantic import BaseModel, Field
from sentry_sdk import capture_exception
import asyncio
import traceback
from typing import Optional, Dict

from .thread import Thread, UserMessage
from .llm import async_prompt


async def async_title_thread(
    thread: Thread, *extra_messages: UserMessage, metadata: Optional[Dict] = None
):
    """
    Generate a title for a thread
    """

    class TitleResponse(BaseModel):
        """A title for a thread of chat messages. It must entice a user to click on the thread when they are interested in the subject."""

        title: str = Field(
            description="a phrase of 2-5 words (or up to 30 characters) that conveys the subject of the chat thread. It should be concise and terse, and not include any special characters or punctuation."
        )

    system_message = "You are an expert at creating concise titles for chat threads."
    messages = thread.get_messages()
    messages.extend(extra_messages)
    messages.append(UserMessage(content="Come up with a title for this thread."))

    try:
        result = await async_prompt(
            messages,
            system_message=system_message,
            model="gpt-4o-mini",
            response_model=TitleResponse,
        )
        thread.title = result.title
        thread.save()

    except Exception as e:
        capture_exception(e)
        traceback.print_exc()
        return


def title_thread(thread: Thread, *extra_messages: UserMessage):
    return asyncio.run(async_title_thread(thread, *extra_messages))
