import copy
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from bson import ObjectId
from jinja2 import Template
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.llm.prompts.system_template import social_media_template, system_template
from eve.agent.llm.util import is_fake_llm_mode, is_test_mode_prompt
from eve.agent.memory.memory_models import (
    get_sender_id_to_sender_name_map,
    select_messages,
)
from eve.agent.memory.service import memory_service
from eve.agent.session.config import (
    build_llm_config_from_agent_settings,
    get_default_session_llm_config,
)
from eve.agent.session.models import (
    Channel,
    ChatMessage,
    Deployment,
    LLMConfig,
    LLMContext,
    LLMContextMetadata,
    LLMTraceMetadata,
    PromptSessionContext,
    Session,
    SessionMemoryContext,
    UpdateType,
)
from eve.agent.session.tracing import add_breadcrumb
from eve.concepts import Concept
from eve.models import Model
from eve.tool import Tool
from eve.user import User, increment_message_count

# Rich notification templates for social media channels
twitter_notification_template = Template("""
│ 📨 TWITTER NOTIFICATION
│ From: @{{ username }}
│ Tweet ID: {{ tweet_id }}
├─────────────────────────────────────
{{ content }}
""")

farcaster_notification_template = Template("""
│ 📨 FARCASTER NOTIFICATION
│ From: {{ farcaster_username }} (FID {{ fid }})
│ Farcaster Hash: {{ farcaster_hash }}
├─────────────────────────────────────
{{ content }}
""")


def parse_mentions(content: str) -> List[str]:
    return re.findall(r"@(\S+)", content)


async def determine_actors(
    session: Session, context: PromptSessionContext
) -> List[Agent]:
    """Determine which agent(s) should respond in this session.

    Actor selection logic:
    1. If actor_agent_ids explicitly specified in context, use those
    2. If session_type is "automatic", use conductor to select next speaker
    3. If session_type is "normal" or "natural":
       - Single agent: return that agent
       - Multiple agents: parse @mentions from message content
    """
    logger.info(f"[ACTORS] determine_actors called for session {session.id}")
    logger.info(
        f"[ACTORS] session_type={session.session_type}, agents={session.agents}"
    )
    logger.info(f"[ACTORS] context.actor_agent_ids={context.actor_agent_ids}")

    add_breadcrumb(
        "Determining actors for session",
        category="session",
        data={"session_id": str(session.id), "session_type": session.session_type},
    )

    # 1. Explicit actor override from context
    if context.actor_agent_ids:
        logger.info(
            f"[ACTORS] Using explicit actor_agent_ids: {context.actor_agent_ids}"
        )
        actors = [Agent.from_mongo(ObjectId(aid)) for aid in context.actor_agent_ids]
        if actors:
            session.update(last_actor_id=actors[0].id)
        logger.info(f"[ACTORS] Returning {len(actors)} explicit actors")
        return actors

    # 2. Automatic session: use conductor to select next speaker
    if session.session_type == "automatic":
        logger.info("[ACTORS] Automatic session - calling conductor_select_actor")
        from eve.agent.session.conductor import conductor_select_actor

        actor = await conductor_select_actor(session)
        logger.info(f"[ACTORS] Conductor selected actor: {actor.username} ({actor.id})")
        session.update(last_actor_id=actor.id)
        return [actor]

    # 3. Normal/natural session
    if not session.agents:
        logger.info("[ACTORS] No agents in session, returning empty list")
        return []

    # Single agent: return it directly
    if len(session.agents) == 1:
        logger.info("[ACTORS] Single agent session, returning that agent")
        actor = Agent.from_mongo(session.agents[0])
        session.update(last_actor_id=actor.id)
        return [actor]

    # Multiple agents: parse @mentions from message
    logger.info("[ACTORS] Multiple agents, parsing @mentions from message")
    mentions = parse_mentions(context.message.content) if context.message else []
    logger.info(f"[ACTORS] Found mentions: {mentions}")
    if mentions:
        actors = []
        for mention in mentions:
            for agent_id in session.agents:
                agent = Agent.from_mongo(agent_id)
                if agent.username == mention:
                    actors.append(agent)
                    break
        if actors:
            session.update(last_actor_id=actors[0].id)
            logger.info(f"[ACTORS] Returning {len(actors)} mentioned actors")
            return actors
        raise ValueError(f"No mentioned agents found in session. Mentions: {mentions}")

    # No actors determined
    logger.info("[ACTORS] No actors determined, returning empty list")
    return []


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
    Prepends channel metadata to message content for Farcaster and Twitter messages
    """
    # Collect all unique sender IDs from messages
    sender_ids = list({msg.sender for msg in messages if msg.sender})

    if not sender_ids:
        return messages

    # Efficiently load all users in one operation
    users = User.find({"_id": {"$in": sender_ids}})
    user_map = {user.id: user for user in users}

    # Process messages and prepend channel info for social media messages
    labeled_messages = []
    for message in messages:
        if message.channel and message.channel.type == "farcaster" and message.sender:
            sender = user_map.get(message.sender)

            # Wrap message content in Farcaster metadata using rich template
            message.content = farcaster_notification_template.render(
                farcaster_hash=message.channel.key,
                farcaster_username=sender.farcasterUsername if sender else "Unknown",
                fid=sender.farcasterId if sender else "Unknown",
                content=message.content,
            )

        elif message.channel and message.channel.type == "twitter" and message.sender:
            sender = user_map.get(message.sender)

            # Wrap message content in Twitter metadata using rich template
            message.content = twitter_notification_template.render(
                username=sender.twitterUsername if sender else "Unknown",
                tweet_id=message.channel.key,
                content=message.content,
            )

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

    # Build social media instructions if this is a social media platform session
    social_instructions = None
    if session.platform == "farcaster":
        deployment = Deployment.find_one({"agent": actor.id, "platform": "farcaster"})
        if deployment and deployment.config and deployment.config.farcaster:
            farcaster_instructions = deployment.config.farcaster.instructions or ""
        else:
            farcaster_instructions = ""
        social_instructions = social_media_template.render(
            has_farcaster=True,
            farcaster_instructions=farcaster_instructions,
        )
    elif session.platform == "twitter":
        deployment = Deployment.find_one({"agent": actor.id, "platform": "twitter"})
        # Twitter deployment may have instructions in the future
        twitter_instructions = ""
        if deployment and deployment.config and deployment.config.twitter:
            twitter_instructions = (
                getattr(deployment.config.twitter, "instructions", "") or ""
            )
        social_instructions = social_media_template.render(
            has_twitter=True,
            twitter_instructions=twitter_instructions,
        )

    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Build system prompt with memory context
    content = system_template.render(
        name=actor.name,
        current_date=current_date,
        description=actor.description,
        persona=actor.persona,
        tools=tools,
        concepts=concepts,
        loras=lora_docs,
        voice=actor.voice,
        memory=memory,
        social_instructions=social_instructions,
    )

    return ChatMessage(session=session.id, role="system", content=content)


async def build_system_extras(
    session: Session, context: PromptSessionContext, config: LLMConfig
):
    extras = []

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

    return context, extras


def add_user_to_session(session: Session, user_id: ObjectId):
    """Add a user to Session.users if not already present."""
    current_users = set(session.users or [])
    if user_id in current_users:
        return

    session.get_collection().update_one(
        {"_id": session.id},
        {"$addToSet": {"users": user_id}},
    )
    session.users = list(current_users | {user_id})


async def add_chat_message(
    session: Session, context: PromptSessionContext, pin: bool = False
):
    # Resolve the triggering user (who initiated this action)
    triggering_user_id = (
        ObjectId(str(context.acting_user_id or context.initiating_user_id))
        if (context.acting_user_id or context.initiating_user_id)
        else None
    )

    new_message = ChatMessage(
        session=session.id,
        sender=ObjectId(str(context.initiating_user_id)),
        role=context.message.role,
        content=context.message.content,
        attachments=context.message.attachments or [],
        trigger=context.trigger,
        apiKey=ObjectId(context.api_key_id) if context.api_key_id else None,
        triggering_user=triggering_user_id,
    )

    # Save channel origin info for social media platforms
    if context.update_config:
        if context.update_config.farcaster_hash:
            new_message.channel = Channel(
                type="farcaster", key=context.update_config.farcaster_hash
            )
        elif context.update_config.twitter_tweet_id:
            new_message.channel = Channel(
                type="twitter", key=context.update_config.twitter_tweet_id
            )
    if pin:
        new_message.pinned = True
    new_message.save()

    # Increment message count for sender
    if context.initiating_user_id:
        increment_message_count(ObjectId(str(context.initiating_user_id)))

    # Add user to Session.users for user role messages
    if context.message.role == "user" and context.initiating_user_id:
        add_user_to_session(session, ObjectId(str(context.initiating_user_id)))

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
    instrumentation=None,
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
        # Deduplicate tools based on tool.name attribute, not just dict key
        # This prevents duplicate tool names when tools are converted to a list for LLM
        existing_tool_names = {
            tool.name for tool in tools.values() if hasattr(tool, "name")
        }
        for tool_key, tool in context.extra_tools.items():
            tool_name = tool.name if hasattr(tool, "name") else tool_key
            if tool_name not in existing_tool_names:
                tools[tool_key] = tool
                existing_tool_names.add(tool_name)

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
    context, system_extras = await build_system_extras(
        session, context, context.llm_config
    )
    if len(system_extras) > 0:
        messages.extend(system_extras)

    existing_messages = select_messages(session)
    messages.extend(existing_messages)
    messages = label_message_channels(messages)
    messages = convert_message_roles(messages, actor.id)

    config = copy.deepcopy(context.llm_config) if context.llm_config else None
    if not config:
        if actor.llm_settings and not force_fake:
            config = await build_llm_config_from_agent_settings(
                actor,
                tier,
                thinking_override=getattr(context, "thinking_override", None),
                context_messages=messages,
            )
        else:
            config = get_default_session_llm_config(
                "premium" if tier != "free" else "free"
            )

    llm_context = LLMContext(
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        config=config,
        metadata=LLMContextMetadata(
            session_id=f"{os.getenv('DB')}-{str(context.session.id)}",
            trace_name="prompt_session",
            trace_id=trace_id,
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
    llm_context.instrumentation = instrumentation
    return llm_context
