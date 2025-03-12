import os
import json
import magic
from bson import ObjectId
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict
from typing import List, Optional, Dict, Any, Literal, Union

from ..mongo import Document, Collection
from ..user import User
from .thread import Thread, UserMessage, AssistantMessage, ToolCall, ChatMessage
from .agent import Agent


"""
Todo:
- conversion: cast user->asst messages and asst->user
- run session with dispatcher
"""



class SessionMessage(ChatMessage):
    """
    A superset of user/assistant messages, capable of being
    cast into a UserMessage or AssistantMessage, depending on
    the target agent for whom we are deriving a single-agent Thread.
    """

    sender_id: ObjectId = Field(default_factory=ObjectId)
    role: Literal["user", "assistant"] = Field(default="user")  # placeholder

    # Common fields for both message types
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}
    hidden: bool = False
    
    # User-specific fields
    attachments: Optional[List[str]] = []
    
    # Assistant-specific fields
    tool_calls: Optional[List[ToolCall]] = []

    def to_thread_message(
        self,
        target_agent: Agent,
    ) -> Union[UserMessage, AssistantMessage]:
        """
        Cast this SessionMessage into either a UserMessage or an AssistantMessage,
        depending on whether `sender_id` matches `target_agent_id`.

        :param target_agent_id: The agent we are generating a Thread for.
        :param sender_name_lookup: Optional function that takes `sender_id` and returns
                                   a string name (for user messages). If not provided,
                                   we fallback to str(sender_id).
        """
        
        # If a message is sent by the target agent, it is an AssistantMessage
        if self.sender_id == target_agent.id:
            return AssistantMessage(
                id=self.id,
                createdAt=self.createdAt,
                role="assistant",
                reply_to=None,
                hidden=self.hidden,
                reactions={},
                agent_id=self.sender_id,
                content=self.content,
                tool_calls=self.tool_calls or [],
            )
        else:
            # All other senders => UserMessage
            # "name" can be a user name or agent label if you like
            name = target_agent.name  # ...
            

            # If message has tool calls, convert them to additional content and attachments
            # if self.tool_calls:
            #     extra_content = ""
            #     extra_attachments = []
            #     for tool_call in self.tool_calls:
            #         result = tool_call.get_result()
            #         if result.get("status") == "completed":
            #             result_items = result.get("result", [])
            #             for r in result_items:
            #                 for output in r.get("output", []):
            #                     if "url" in output:
            #                         extra_attachments.append(output["url"])

            return UserMessage(
                id=self.id,
                createdAt=self.createdAt,
                role="user",
                reply_to=None,
                hidden=self.hidden,
                reactions={},
                name=name,
                content=self.content,
                metadata=self.metadata,
                attachments=self.attachments or [],
            )




@Collection("sessions")
class Session(Document):
    """
    A Session is a higher-level container for a scenario or multi-agent conversation.
    - Contains multiple agents and their roles.
    - Holds a list of SessionMessage objects (the superset of user or assistant messages).
    - We can derive a single-agent Thread from it, by picking one agent as 'the assistant.'
    """
    # key: Optional[str] = None
    # title: Optional[str] = None

    # Scenario fields
    scenario: Optional[str] = None
    current: Optional[str] = None
    agents: List[ObjectId] = Field(default_factory=list)

    # The entire history of messages for this session
    messages: List[SessionMessage] = Field(default_factory=list)

    # We might keep a max number of messages if needed
    message_limit: Optional[int] = 25

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_messages(self, last_n: Optional[int] = None) -> List[SessionMessage]:
        """
        Return up to the last_n messages (default self.message_limit).
        This is the raw session-level message list (SessionMessages).
        """
        last_n = last_n or self.message_limit
        
        all_messages = self.messages[-last_n:]
        messages = [m for m in all_messages if not m.hidden]

        # hidden messages should be excluded except for the last one (e.g. a trigger)
        if all_messages and all_messages[-1].hidden:
            messages.append(all_messages[-1])

        return messages

    def to_thread(
        self,
        target_agent: Agent,
        user: User,
    ) -> Thread:
        """
        Produce a single-agent Thread from this Session for `target_agent_id`.
        All messages from that agent become AssistantMessage objects; 
        all other messages become UserMessage objects. The returned Thread 
        has a single `agent` and a single `user`.
        
        :param target_agent_id: Which agent in this session is 'the assistant'?
        :param user_id: The user or player ID to store in the resulting Thread.
        :param sender_name_lookup: Optional function that returns a display 
                                   name for a given sender_id (used by user messages).
        """
        raw_msgs = self.get_messages()
        casted = [
            msg.to_thread_message(target_agent=target_agent)
            for msg in raw_msgs
        ]

        return Thread(
            key=self.key,
            title=self.title,
            agent=target_agent.id,
            user=user.id,
            messages=casted,
            active=[],
            message_limit=self.message_limit,
        )

    @classmethod
    def load(
        cls,
        key: str,
        create_if_missing: bool = False,
        message_limit: int = 25,
    ):
        """
        Example helper to load a Session from DB by key. 
        (If your real logic includes agent/user filters, adapt as needed.)
        """
        filter_doc = {"key": key}
        doc = cls.get_collection().find_one(filter_doc)
        if doc:
            session = Session(**doc)
            session.messages = session.messages[-message_limit:]
            return session
        else:
            if create_if_missing:
                session = cls(key=key, message_limit=message_limit)
                session.save()
                return session
            else:
                db = os.getenv("DB")
                raise ValueError(
                    f"Session {key} not found in {cls.collection_name}:{db}"
                )

