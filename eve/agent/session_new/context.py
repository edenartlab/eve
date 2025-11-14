import os
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from bson import ObjectId
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.session.config import get_default_session_llm_config
from eve.agent.session.models import (
    Channel,
    ChatMessage,
    LLMConfig,
    LLMContext,
    LLMContextMetadata,
    LLMTraceMetadata,
    PromptSessionContext,
    Session,
    SessionMemoryContext,
    UpdateType,
)
from eve.agent.llm.util import (
    is_fake_llm_mode,
    is_test_mode_prompt,
)
from eve.agent.llm.prompts.system_template import SYSTEM_TEMPLATE
from eve.agent.session.tracing import add_breadcrumb
from eve.agent.memory.memory_models import (
    get_sender_id_to_sender_name_map,
    select_messages,
)
from eve.agent.memory.service import memory_service
from eve.concepts import Concept
from eve.models import Model
from eve.tool import Tool
from eve.user import User


def parse_mentions(content: str) -> List[str]:
    return re.findall(r"@(\S+)", content)


async def determine_actors(
    session: Session, context: PromptSessionContext
) -> List[Agent]:
    add_breadcrumb(
        "Determining actors for session",
        category="session",
        data={"session_id": str(session.id)},
    )
    actor_ids = []

    if context.actor_agent_ids:
        # Multiple actors specified in the context
        for actor_agent_id in context.actor_agent_ids:
            requested_actor = ObjectId(actor_agent_id)
            actor_ids.append(requested_actor)
    elif len(session.agents) > 1:
        mentions = parse_mentions(context.message.content)
        if len(mentions) > 0:
            for mention in mentions:
                for agent_id in session.agents:
                    agent = Agent.from_mongo(agent_id)
                    if agent.username == mention:
                        actor_ids.append(agent_id)
                        break
            if not actor_ids:
                raise ValueError("No mentioned agents found in session")

    if not actor_ids:
        # TODO: do something more graceful than returning empty list if no actors are determined
        return []

    actors = []
    for actor_id in actor_ids:
        actor = Agent.from_mongo(actor_id)
        actors.append(actor)

    # Update last_actor_id to the first actor for backwards compatibility
    if actors:
        # session.last_actor_id = actors[0].id
        # session.save()
        session.update(last_actor_id=actors[0].id)

    return actors


def convert_message_roles(messages: List[ChatMessage], actor_id: ObjectId):
    """
    Re-assembles messages from perspective of actor (assistant) and everyone else (user)
    """

    # Get sender name mapping for all messages
    sender_name_map = get_sender_id_to_sender_name_map(messages)

    converted_messages = []
    for message in messages:
        if message.role == "system":
            converted_messages.append(message)
        elif message.sender == actor_id:
            converted_messages.append(message.as_assistant_message())
        else:
            user_message = message.as_user_message()
            # Include sender name in the message content if available
            if message.sender and message.sender in sender_name_map:
                sender_name = sender_name_map[message.sender]
                # Prepend the sender name to the content
                user_message.content = f"[{sender_name}]: {user_message.content}"
            converted_messages.append(user_message)

    return converted_messages


def label_message_channels(messages: List[ChatMessage]):
    """
    Prepends channel metadata to message content for Farcaster messages
    """
    # Collect all unique sender IDs from messages
    sender_ids = list({msg.sender for msg in messages if msg.sender})

    if not sender_ids:
        return messages

    # Efficiently load all users in one operation
    users = User.find({"_id": {"$in": sender_ids}})
    user_map = {user.id: user for user in users}
    # Process messages and prepend channel info for Farcaster messages
    labeled_messages = []
    for message in messages:
        if message.channel and message.channel.type == "farcaster" and message.sender:
            sender = user_map.get(message.sender)
            sender_farcaster_fid = sender.farcasterId if sender else "Unknown"
            # Prepend the Farcaster metadata to the message content
            prepend_text = f"<<Farcaster Hash: {message.channel.key}, FID: {sender_farcaster_fid}>>"
            message.content = f"{prepend_text} {message.content}"
        labeled_messages.append(message)

    return labeled_messages


async def build_system_message(
    session: Session,
    actor: Agent,
    user: Optional[User],
    tools: Dict[str, Tool],
    instrumentation=None,
):  # Get the last speaker ID for memory prioritization
    # Get concepts
    concepts = Concept.find({"agent": actor.id, "deleted": {"$ne": True}})

    # Get loras
    lora_dict = {m["lora"]: m for m in actor.models or []}
    lora_docs = Model.find(
        {"_id": {"$in": list(lora_dict.keys())}, "deleted": {"$ne": True}}
    )
    loras = [doc.model_dump() for doc in lora_docs]
    for doc in loras:
        doc["use_when"] = lora_dict[doc["id"]].get(
            "use_when", "This is your default Lora model"
        )

    # Get memory (unless excluded by session extras)
    memory = None
    if user and not (session.extras and session.extras.exclude_memory):
        memory = await memory_service.assemble_memory_context(
            session,
            actor,
            user,
            reason="build_system_message",
            instrumentation=instrumentation,
        )

    # Build system prompt with memory context
    content = SYSTEM_TEMPLATE.render(
        name=actor.name,
        # current_date_time=current_date_time,
        description=actor.description,
        persona=actor.persona,
        tools=tools,
        concepts=concepts,
        loras=lora_docs,
        voice=actor.voice,
        memory=memory,
    )

    return ChatMessage(session=session.id, role="system", content=content)


async def build_system_extras(
    session: Session, context: PromptSessionContext, config: LLMConfig
):
    extras = []

    # deprecated when we move to new farcaster gateway (wip in abraham)
    # if context.update_config and context.update_config.farcaster_hash:
    #     extras.append(
    #         ChatMessage(
    #             session=session.id,
    #             role="system",
    #             content="You are currently replying to a Farcaster cast. The maximum length before the fold is 320 characters, and the maximum length is 1024 characters, so attempt to be concise in your response.",
    #         )
    #     )
    #     config.max_tokens = 1024

    # add trigger context
    if hasattr(session, "context") and session.context:
        context_prompt = f"<Full Task Context>\n{session.context}\n\n**IMPORTANT: Ignore me, the user! You are just speaking to the other agents now. Make sure you stay relevant to the full task context throughout the conversation.</Full Task Context>"
        extras.append(
            ChatMessage(
                session=session.id,
                # debug this
                # role="system",
                role="user",
                sender=ObjectId(str(context.initiating_user_id)),
                content=context_prompt,
            )
        )

    # add trigger context (prefer session.trigger but fall back to context.trigger)
    trigger_id = session.trigger or context.trigger
    if trigger_id:
        from eve.trigger import Trigger

        trigger = Trigger.from_mongo(trigger_id)
        if trigger and trigger.context:
            extras.append(
                ChatMessage(
                    session=session.id,
                    role="user",
                    sender=ObjectId(str(context.initiating_user_id)),
                    content=trigger.context,
                )
            )

    return context, config, extras


async def add_chat_message(
    session: Session, context: PromptSessionContext, pin: bool = False
):
    new_message = ChatMessage(
        session=session.id,
        sender=ObjectId(str(context.initiating_user_id)),
        role=context.message.role,
        content=context.message.content,
        attachments=context.message.attachments or [],
        trigger=context.trigger,
        apiKey=ObjectId(context.api_key_id) if context.api_key_id else None,
    )

    # save farcaster origin info
    # todo: other platforms
    if context.update_config:
        if context.update_config.farcaster_hash:
            new_message.channel = Channel(
                type="farcaster", key=context.update_config.farcaster_hash
            )
    if pin:
        new_message.pinned = True
    new_message.save()

    memory_context = session.memory_context
    memory_context.last_activity = datetime.now(timezone.utc)
    memory_context.messages_since_memory_formation += 1
    session.update(memory_context=memory_context.model_dump())
    session.memory_context = SessionMemoryContext(**session.memory_context)

    # Broadcast user message to SSE connections for real-time updates
    try:
        from eve.api.sse_manager import sse_manager
        from eve.user import User

        # Get full user data for enrichment
        user = User.from_mongo(context.initiating_user_id)

        # increment stats
        # stats = user.stats
        # stats["messageCount"] += 1
        # user.update(stats=stats.model_dump())

        user_data = {
            "_id": str(user.id),
            "username": user.username,
            "name": user.username,  # Use username as name for consistency
            "userImage": user.userImage,
        }

        message_dict = new_message.model_dump(by_alias=True)
        # Enrich sender with full user data if available
        if user_data:
            message_dict["sender"] = user_data

        user_message_update = {
            "type": UpdateType.USER_MESSAGE.value,
            "message": message_dict,
        }

        session_id = str(session.id)
        await sse_manager.broadcast(session_id, user_message_update)
    except Exception as e:
        logger.error(f"Failed to broadcast user message to SSE: {e}")

    return new_message


async def build_llm_context(
    session: Session,
    actor: Agent,
    context: PromptSessionContext,
    trace_id: Optional[str] = str(uuid.uuid4()),
):
    instrumentation = getattr(context, "instrumentation", None)
    if context.initiating_user_id:
        user = User.from_mongo(context.initiating_user_id)
        tier = (
            "premium" if user.subscriptionTier and user.subscriptionTier > 0 else "free"
        )
    else:
        user = None
        tier = "free"

    auth_user_id = context.acting_user_id or context.initiating_user_id
    if context.tools:
        tools = context.tools
    else:
        tools = actor.get_tools(cache=False, auth_user=auth_user_id)

    if context.extra_tools:
        tools.update(context.extra_tools)

    # setup tool_choice
    if tools:
        tool_choice = context.tool_choice or "auto"
    else:
        tool_choice = "none"
    if tool_choice not in ["auto", "none"]:
        tool_choice = {"type": "function", "function": {"name": context.tool_choice}}

    raw_prompt_text = (
        context.message.content if context.message and context.message.content else None
    )
    force_fake = is_fake_llm_mode() or is_test_mode_prompt(raw_prompt_text)

    # build messages first to have context for thinking routing
    system_message = await build_system_message(
        session,
        actor,
        user,
        tools,
        instrumentation=instrumentation,
    )

    messages = [system_message]
    context, base_config, system_extras = await build_system_extras(
        session, context, context.llm_config or get_default_session_llm_config(tier)
    )
    if len(system_extras) > 0:
        messages.extend(system_extras)

    existing_messages = select_messages(session)
    messages.extend(existing_messages)
    messages = label_message_channels(messages)
    messages = convert_message_roles(messages, actor.id)

    # Use agent's llm_settings if available, otherwise fallback to context or default
    if actor.llm_settings and not force_fake:
        from eve.agent.session.config import build_llm_config_from_agent_settings

        config = await build_llm_config_from_agent_settings(
            actor,
            tier,
            thinking_override=getattr(context, "thinking_override", None),
            context_messages=messages,  # Pass existing messages for routing context
        )
    else:
        config = context.llm_config or get_default_session_llm_config(tier)

    return LLMContext(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        config=config,
        metadata=LLMContextMetadata(
            # for observability purposes. not same as session.id
            session_id=f"{os.getenv('DB')}-{str(context.session.id)}",
            trace_name="prompt_session",
            trace_id=trace_id,  # trace_id represents the entire prompt session
            generation_name="prompt_session",
            trace_metadata=LLMTraceMetadata(
                user_id=str(context.initiating_user_id)
                if context.initiating_user_id
                else None,
                agent_id=str(actor.id),
                session_id=str(context.session.id),
            ),
        ),
    )
