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
from eve.agent.llm.prompts.agent_session_template import agent_session_template
from eve.agent.llm.prompts.social_media_template import social_media_template
from eve.agent.llm.prompts.system_template import system_template
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
    EdenMessageData,
    EdenMessageType,
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

discord_notification_template = Template("""
│ 📨 DISCORD NOTIFICATION
│ From: {{ username }}
│ Message ID: {{ message_id }}
├─────────────────────────────────────
{{ content }}
""")

eden_notification_template = Template("""
│ 📨 CHAT MESSAGE NOTIFICATION
│ From: {{ username }}
│ Message ID: {{ message_id }}
├─────────────────────────────────────
{{ content }}
""")


def find_mentioned_agents(content: str, agents: List[Agent]) -> List[Agent]:
    """Find agents mentioned in the message content.

    Uses whole-word, case-insensitive matching for agent usernames.
    For example, if agent username is "Go", it won't match "goodbye" or "ago".
    """
    if not content:
        return []

    mentioned = []
    for agent in agents:
        # Case-insensitive, whole-word match using word boundaries
        pattern = r"\b" + re.escape(agent.username) + r"\b"
        if re.search(pattern, content, re.IGNORECASE):
            mentioned.append(agent)
    return mentioned


async def determine_actors(
    session: Session, context: PromptSessionContext
) -> List[Agent]:
    """Determine which agent(s) should respond in this session.

    Actor selection depends on session_type:

    PASSIVE (default):
      - 1 agent + 1 user: always prompt the agent (classic assistant pattern)
      - Multiple agents OR multiple users: only prompt mentioned agents

    NATURAL:
      - If agents are mentioned: prompt the mentioned agents
      - If no agents mentioned: use Conductor to decide (conservative mode)
      - Conductor can return None, terminating the process

    AUTOMATIC:
      - Use Conductor to select next speaker (must select one)
      - Used for multi-agent scenarios that run continuously
    """
    logger.info(
        f"[ACTORS] determine_actors: session={session.id}, type={session.session_type}"
    )

    add_breadcrumb(
        "Determining actors for session",
        category="session",
        data={"session_id": str(session.id), "session_type": session.session_type},
    )

    # 1. Explicit actor override always takes precedence
    if context.actor_agent_ids:
        logger.info(
            f"[ACTORS] Using explicit actor_agent_ids: {context.actor_agent_ids}"
        )
        actors = [Agent.from_mongo(ObjectId(aid)) for aid in context.actor_agent_ids]
        if actors:
            session.update(last_actor_id=actors[0].id)
        return actors

    # No agents in session
    if not session.agents:
        logger.info("[ACTORS] No agents in session")
        return []

    # Load all agents for the session
    agents = [Agent.from_mongo(agent_id) for agent_id in session.agents]
    num_agents = len(agents)
    num_users = len(session.users) if session.users else 0
    message_content = context.message.content if context.message else ""

    logger.info(f"[ACTORS] num_agents={num_agents}, num_users={num_users}")

    # ========== PASSIVE ==========
    if session.session_type == "passive":
        # Check if this is a Discord/Telegram channel (not DM) - requires mention even for 1:1
        session_key = session.session_key or ""
        is_social_channel = (
            "discord" in session_key or "telegram" in session_key
        ) and "_dm_" not in session_key

        # Classic 1:1 pattern - always respond (unless it's a social channel)
        if num_agents == 1 and num_users <= 1 and not is_social_channel:
            logger.info("[ACTORS] Passive 1:1 - prompting single agent")
            session.update(last_actor_id=agents[0].id)
            return agents

        # Multi-party or social channel - only respond to mentions
        mentioned = find_mentioned_agents(message_content, agents)
        if mentioned:
            logger.info(
                f"[ACTORS] Passive multi-party/social - found {len(mentioned)} mentioned agents"
            )
            session.update(last_actor_id=mentioned[0].id)
            return mentioned

        logger.info("[ACTORS] Passive multi-party/social - no mentions, no response")
        return []

    # ========== NATURAL ==========
    elif session.session_type == "natural":
        # Check for mentions first
        mentioned = find_mentioned_agents(message_content, agents)
        if mentioned:
            logger.info(f"[ACTORS] Natural - found {len(mentioned)} mentioned agents")
            session.update(last_actor_id=mentioned[0].id)
            return mentioned

        # No mentions - use conductor (conservative mode)
        logger.info("[ACTORS] Natural - no mentions, using conductor (conservative)")
        from eve.agent.session.conductor import conductor_select_actor_natural

        actor = await conductor_select_actor_natural(session)
        if actor:
            logger.info(f"[ACTORS] Conductor selected: {actor.username}")
            session.update(last_actor_id=actor.id)
            return [actor]

        logger.info("[ACTORS] Conductor decided no response needed")
        return []

    # ========== AUTOMATIC ==========
    elif session.session_type == "automatic":
        logger.info("[ACTORS] Automatic - using conductor")
        from eve.agent.session.conductor import conductor_select_actor

        actor = await conductor_select_actor(session)
        logger.info(f"[ACTORS] Conductor selected: {actor.username}")
        session.update(last_actor_id=actor.id)
        return [actor]

    # Unknown session type
    logger.warning(f"[ACTORS] Unknown session_type: {session.session_type}")
    return []


def convert_message_roles(messages: List[ChatMessage], actor_id: ObjectId):
    """
    Re-assembles messages from perspective of actor (assistant) and everyone else (user)
    """

    # Social media channel types that use notification decorators instead of [name]: prefix
    social_media_channels = {"discord", "twitter", "farcaster"}

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
            # Skip for social media messages - they use notification decorators instead
            has_social_channel = (
                message.channel and message.channel.type in social_media_channels
            )
            if (
                message.sender
                and message.sender in sender_name_map
                and not has_social_channel
            ):
                is_system_message = (
                    user_message.content
                    and user_message.content.startswith("<SystemMessage>")
                    and user_message.content.endswith("</SystemMessage>")
                )
                # Prepend the sender name to the content
                if not is_system_message:
                    sender_name = sender_name_map[message.sender]
                    user_message.content = f"[{sender_name}]: {user_message.content}"

            converted_messages.append(user_message)

    return converted_messages


def label_message_channels(messages: List[ChatMessage], session: "Session" = None):
    """
    Prepends channel metadata to message content for Farcaster, Twitter, and Discord messages.

    Args:
        messages: List of chat messages to label
        session: Optional session to use as fallback for determining platform
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
        # Determine channel type - prefer message.channel, fallback to session.platform
        channel_type = None
        if message.channel and message.channel.type:
            channel_type = message.channel.type
        elif session and session.platform:
            channel_type = session.platform

        # Only apply social media decorators to user messages (except eden which handles both)
        # Eden messages can come from other agents (assistant role) in multi-agent sessions
        if message.role != "user" and channel_type != "eden":
            labeled_messages.append(message)
            continue

        if channel_type == "farcaster" and message.sender and message.channel:
            sender = user_map.get(message.sender)

            # Wrap message content in Farcaster metadata using rich template
            message.content = farcaster_notification_template.render(
                farcaster_hash=message.channel.key,
                farcaster_username=sender.farcasterUsername if sender else "Unknown",
                fid=sender.farcasterId if sender else "Unknown",
                content=message.content,
            )

        elif channel_type == "twitter" and message.sender and message.channel:
            sender = user_map.get(message.sender)

            # Wrap message content in Twitter metadata using rich template
            message.content = twitter_notification_template.render(
                username=sender.twitterUsername if sender else "Unknown",
                tweet_id=message.channel.key,
                content=message.content,
            )

        elif channel_type == "discord" and message.sender and message.channel:
            sender = user_map.get(message.sender)

            # Wrap message content in Discord metadata using rich template
            # channel.key contains the Discord message ID
            # Fallback chain: discordUsername -> username -> "Unknown"
            discord_username = "Unknown"
            if sender:
                discord_username = (
                    sender.discordUsername or sender.username or "Unknown"
                )
            message.content = discord_notification_template.render(
                username=discord_username,
                message_id=message.channel.key or "Unknown",
                content=message.content,
            )

        elif channel_type == "eden" and message.sender:
            sender = user_map.get(message.sender)
            # For eden, sender could be a user or agent - look up agent if not found
            sender_username = "Unknown"
            if sender:
                sender_username = sender.username or "Unknown"
            else:
                # Try to find agent (they share the users collection)
                agent_sender = Agent.from_mongo(message.sender)
                if agent_sender:
                    sender_username = agent_sender.username or "Unknown"

            message.content = eden_notification_template.render(
                username=sender_username,
                message_id=message.channel.key
                if message.channel and message.channel.key
                else "Unknown",
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
    logger.info(
        f"[build_system_message] session.platform={session.platform}, actor.id={actor.id}"
    )
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
    elif session.platform == "discord":
        logger.info(
            "[build_system_message] Discord platform detected, looking for deployment"
        )
        deployment = Deployment.find_one({"agent": actor.id, "platform": "discord"})
        logger.info(
            f"[build_system_message] Found deployment: {deployment.id if deployment else None}"
        )
        discord_instructions = ""
        if deployment and deployment.config and deployment.config.discord:
            discord_instructions = (
                getattr(deployment.config.discord, "instructions", "") or ""
            )
        # Use discord_channel_id directly from session
        discord_channel_id = session.discord_channel_id or ""
        social_instructions = social_media_template.render(
            has_discord=True,
            discord_instructions=discord_instructions,
            discord_channel_id=discord_channel_id,
        )
        logger.info(
            f"[build_system_message] Discord social_instructions generated: {len(social_instructions) if social_instructions else 0} chars"
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
        session_context=session.context,
    )

    return ChatMessage(session=[session.id], role="system", content=content)


async def build_system_extras(
    session: Session, context: PromptSessionContext, config: LLMConfig
):
    extras = []

    # add trigger context (pinned as first user message with SystemMessage tag)
    if hasattr(session, "context") and session.context:
        context_prompt = f"<SystemMessage>{session.context}\n\n**IMPORTANT: Ignore me, the user! You are just speaking to the other agents now. Make sure you stay relevant to the full task context throughout the conversation.</SystemMessage>"
        extras.append(
            ChatMessage(
                session=[session.id],
                role="user",
                sender=ObjectId(str(context.initiating_user_id)),
                content=context_prompt,
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

    # Ensure all attachments are on Eden
    attachments = context.message.attachments or []
    if attachments:
        from eve.s3 import upload_attachments_to_eden

        attachments = await upload_attachments_to_eden(attachments)

    # Set eden_message_data for trigger messages with eden role
    eden_message_data = None
    if context.trigger and context.message.role == "eden":
        eden_message_data = EdenMessageData(message_type=EdenMessageType.TRIGGER)

    new_message = ChatMessage(
        session=[session.id],
        sender=ObjectId(str(context.initiating_user_id)),
        role=context.message.role,
        content=context.message.content,
        attachments=attachments,
        trigger=context.trigger,
        apiKey=ObjectId(context.api_key_id) if context.api_key_id else None,
        triggering_user=triggering_user_id,
        billed_user=triggering_user_id,
        eden_message_data=eden_message_data,
    )

    # Save channel origin info for social media platforms
    # First check if channel is already set on the message input
    if context.message.channel:
        new_message.channel = context.message.channel
    # Fallback to update_config for backwards compatibility
    elif context.update_config:
        if context.update_config.discord_message_id:
            new_message.channel = Channel(
                type="discord", key=context.update_config.discord_message_id
            )
        elif context.update_config.farcaster_hash:
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

    # Distribute to agent_sessions for multi-agent automatic sessions
    if (
        session.session_type == "automatic"
        and session.agent_sessions
        and len(session.agent_sessions) > 0
    ):
        await distribute_message_to_agent_sessions(
            parent_session=session,
            message=new_message,
            exclude_agent_id=None,  # User messages go to all agents
        )

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


async def distribute_message_to_agent_sessions(
    parent_session: Session,
    message: ChatMessage,
    exclude_agent_id: Optional[ObjectId] = None,
) -> None:
    """Distribute a message to all agent_sessions in a multi-agent session.

    When a message is added to a parent session (by user or agent), this function
    ensures it's also added to all OTHER agent_sessions. This enables real-time
    message distribution instead of bulk sync.

    Args:
        parent_session: The parent chatroom session
        message: The ChatMessage that was just created
        exclude_agent_id: If sender is an agent, exclude their agent_session
    """
    if not parent_session.agent_sessions:
        return

    target_sessions = []
    for agent_id_str, agent_session_id in parent_session.agent_sessions.items():
        # Skip the sender's own agent_session if they're an agent
        if exclude_agent_id and agent_id_str == str(exclude_agent_id):
            continue
        target_sessions.append(agent_session_id)

    if not target_sessions:
        return

    logger.info(
        f"[DISTRIBUTE] Distributing message {message.id} to {len(target_sessions)} agent_sessions"
    )

    # Add channel reference if not set (use update() to only modify channel field)
    # Convert to dict for MongoDB encoding
    if not message.channel:
        channel = Channel(type="eden", key=str(message.id))
        message.update(channel=channel.model_dump())

    # Add agent_session IDs to the message's session list atomically
    ChatMessage.get_collection().update_one(
        {"_id": message.id},
        {"$addToSet": {"session": {"$each": target_sessions}}},
    )
    # Update in-memory object
    message.session = list(set(message.session + target_sessions))

    # Add the message sender to each target agent_session's users array
    # This ensures agent_sessions track all users who've sent messages to them
    # Using batch update with $addToSet (idempotent - won't add duplicates)
    if message.sender and target_sessions:
        Session.get_collection().update_many(
            {"_id": {"$in": target_sessions}},
            {"$addToSet": {"users": message.sender}},
        )
        logger.info(
            f"[DISTRIBUTE] Added sender {message.sender} to users of {len(target_sessions)} agent_sessions"
        )

    logger.info(
        f"[DISTRIBUTE] Message {message.id} now in sessions: {[str(s) for s in message.session]}"
    )


def get_all_eden_messages_for_llm(session_id: ObjectId) -> List[ChatMessage]:
    """
    Get ALL eden messages for a session, converted for LLM consumption.

    Eden messages are excluded from select_messages(), so we query separately.
    Each eden message is converted to user role with content wrapped in SystemMessage tags.

    This includes all conductor messages (CONDUCTOR_INIT, CONDUCTOR_TURN, CONDUCTOR_HINT,
    CONDUCTOR_FINISH) as well as other eden messages (AGENT_ADD, RATE_LIMIT, TRIGGER).

    Returns empty list if no eden messages exist.
    """
    messages_collection = ChatMessage.get_collection()
    eden_messages = list(
        messages_collection.find({"session": session_id, "role": "eden"}).sort(
            "createdAt", 1
        )  # Chronological order
    )

    if not eden_messages:
        return []

    converted = []
    for doc in eden_messages:
        eden_msg = ChatMessage(**doc)
        # Convert: change role to user, wrap content in SystemMessage tags
        converted.append(
            eden_msg.model_copy(
                update={
                    "role": "user",
                    "content": f"<SystemMessage>{eden_msg.content}</SystemMessage>",
                }
            )
        )

    return converted


# Keep old function for backward compatibility but mark as deprecated
def get_last_eden_message_for_llm(session_id: ObjectId) -> Optional[ChatMessage]:
    """
    DEPRECATED: Use get_all_eden_messages_for_llm() instead.

    Get the last eden message for a session, converted for LLM consumption.
    """
    all_eden = get_all_eden_messages_for_llm(session_id)
    return all_eden[-1] if all_eden else None


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

    # Add ALL eden messages (converted to user role) to the context
    # Eden messages are filtered out by select_messages, so we query separately
    # This includes conductor messages (CONDUCTOR_INIT, CONDUCTOR_TURN, CONDUCTOR_HINT, etc.)
    eden_messages = get_all_eden_messages_for_llm(session.id)
    if eden_messages:
        existing_messages.extend(eden_messages)
        existing_messages.sort(key=lambda m: m.createdAt)

    messages.extend(existing_messages)
    messages = label_message_channels(messages, session)
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


# =============================================================================
# Agent Session Context Building Functions
# =============================================================================


async def build_agent_session_system_message(
    agent_session: Session,
    parent_session: Session,
    actor: Agent,
    tools: Dict[str, Tool],
    instrumentation=None,
) -> ChatMessage:
    """Build system message for an agent_session with chatroom framing.

    Uses the agent_session_template which emphasizes the private workspace
    concept and the need to use post_to_chatroom to contribute.

    Args:
        agent_session: The agent's private workspace session
        parent_session: The parent chatroom session
        actor: The agent
        tools: Dict of available tools
        instrumentation: Optional instrumentation for tracing

    Returns:
        ChatMessage with system role containing the rendered template
    """
    from eve.concepts import Concept
    from eve.models import Model

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

    # Get memory - agent_session inherits users from parent
    memory = None
    user = None
    if parent_session.users:
        user = User.from_mongo(parent_session.users[0])

    if user and not (agent_session.extras and agent_session.extras.exclude_memory):
        memory = await memory_service.assemble_memory_context(
            agent_session,
            actor,
            user,
            reason="build_agent_session_system_message",
            instrumentation=instrumentation,
        )

    # Render the agent session template
    content = agent_session_template.render(
        name=actor.name,
        description=actor.description,
        persona=actor.persona,
        chatroom_scenario=parent_session.context,  # Use parent's context as scenario
        tools=tools,
        concepts=concepts,
        loras=loras,
        voice=actor.voice,
        memory=memory,
    )

    return ChatMessage(session=[agent_session.id], role="system", content=content)


async def build_agent_session_llm_context(
    agent_session: Session,
    parent_session: Session,
    actor: Agent,
    context: PromptSessionContext,
    trace_id: Optional[str] = None,
    instrumentation=None,
) -> LLMContext:
    """Build LLM context for an agent_session turn.

    This context includes:
    1. Agent_session system prompt (chatroom framing with private workspace emphasis)
    2. Agent_session's own message history (includes messages from other agents
       that were distributed in real-time)

    Messages from the parent session (from other agents) are automatically
    distributed to agent_sessions when created, so they're already in the
    agent_session's message history - no bulk sync needed.

    Args:
        agent_session: The agent's private workspace session
        parent_session: The parent chatroom session
        actor: The agent
        context: PromptSessionContext for the turn
        trace_id: Optional trace ID for observability
        instrumentation: Optional instrumentation for tracing

    Returns:
        LLMContext ready for prompting the agent
    """
    if not trace_id:
        trace_id = str(uuid.uuid4())

    # Get user info for tier determination
    if context.initiating_user_id:
        user = User.from_mongo(context.initiating_user_id)
        tier = (
            "premium"
            if user and user.subscriptionTier and user.subscriptionTier > 0
            else "free"
        )
    else:
        user = None
        tier = "free"

    # Get tools for the agent
    auth_user_id = context.acting_user_id or context.initiating_user_id
    tools = actor.get_tools(cache=False, auth_user=auth_user_id)

    # Add post_to_chatroom tool (required for agent_sessions)
    post_tool = Tool.load("post_to_chatroom")
    if post_tool:
        tools["post_to_chatroom"] = post_tool

    # Build system message with chatroom framing
    system_message = await build_agent_session_system_message(
        agent_session,
        parent_session,
        actor,
        tools,
        instrumentation=instrumentation,
    )

    messages = [system_message]

    # Pin agent_session.context as first message after system (never ages out)
    # This contains the conductor-generated personalized context for this agent
    logger.info(
        f"[AGENT_SESSION_CONTEXT] agent_session.id={agent_session.id}, "
        f"has_context={bool(agent_session.context)}, "
        f"context_len={len(agent_session.context) if agent_session.context else 0}"
    )
    if agent_session.context:
        logger.info(
            f"[AGENT_SESSION_CONTEXT] Pinning context (first 200 chars): "
            f"{agent_session.context[:200]}..."
        )
        context_message = ChatMessage(
            session=[agent_session.id],
            role="user",
            sender=ObjectId(str(context.initiating_user_id))
            if context.initiating_user_id
            else ObjectId("000000000000000000000000"),
            content=f"<SystemMessage>{agent_session.context}</SystemMessage>",
        )
        messages.append(context_message)
        logger.info(
            "[AGENT_SESSION_CONTEXT] Context pinned as first user message with <SystemMessage> tag"
        )
    else:
        logger.warning(
            f"[AGENT_SESSION_CONTEXT] No context to pin for agent_session {agent_session.id}"
        )

    # Add agent_session's own history (includes messages from other agents
    # that were distributed in real-time via distribute_message_to_agent_sessions)
    existing_messages = select_messages(agent_session)

    # Add ALL eden messages (converted to user role) to the context
    # This includes CONDUCTOR_HINT messages which are specific to this agent_session
    eden_messages = get_all_eden_messages_for_llm(agent_session.id)
    if eden_messages:
        existing_messages.extend(eden_messages)
        existing_messages.sort(key=lambda m: m.createdAt)

    messages.extend(existing_messages)

    # Log messages before role conversion to debug context presence
    logger.info(
        f"[AGENT_SESSION_CONTEXT] === Messages before convert_message_roles ({len(messages)} total) ==="
    )
    for i, msg in enumerate(messages):
        content_preview = (
            (msg.content[:100] + "...")
            if msg.content and len(msg.content) > 100
            else msg.content
        )
        has_system_tag = (
            msg.content and "<SystemMessage>" in msg.content if msg.content else False
        )
        logger.info(
            f"[AGENT_SESSION_CONTEXT] msg[{i}]: role={msg.role}, sender={msg.sender}, "
            f"has_system_tag={has_system_tag}, content_preview={content_preview}"
        )

    # Convert message roles (agent's messages -> assistant, others -> user)
    messages = convert_message_roles(messages, actor.id)

    # Log messages after role conversion
    logger.info(
        f"[AGENT_SESSION_CONTEXT] === Messages after convert_message_roles ({len(messages)} total) ==="
    )
    for i, msg in enumerate(messages):
        content_preview = (
            (msg.content[:100] + "...")
            if msg.content and len(msg.content) > 100
            else msg.content
        )
        has_system_tag = (
            msg.content and "<SystemMessage>" in msg.content if msg.content else False
        )
        logger.info(
            f"[AGENT_SESSION_CONTEXT] msg[{i}]: role={msg.role}, has_system_tag={has_system_tag}, "
            f"content_preview={content_preview}"
        )

    # Build LLM config
    config = None
    if actor.llm_settings:
        config = await build_llm_config_from_agent_settings(
            actor,
            tier,
            context_messages=messages,
        )
    else:
        config = get_default_session_llm_config(tier)

    return LLMContext(
        messages=messages,
        tools=tools,
        tool_choice="auto",
        config=config,
        metadata=LLMContextMetadata(
            session_id=f"{os.getenv('DB')}-{str(agent_session.id)}",
            trace_name="agent_session_prompt",
            trace_id=trace_id,
            generation_name="agent_session_prompt",
            trace_metadata=LLMTraceMetadata(
                user_id=str(context.initiating_user_id)
                if context.initiating_user_id
                else None,
                agent_id=str(actor.id),
                session_id=str(agent_session.id),
            ),
        ),
        instrumentation=instrumentation,
    )
