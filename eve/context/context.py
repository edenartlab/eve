import os
import pytz
from bson import ObjectId
from typing import Dict, List, Optional
from datetime import datetime

from eve.user import User
from eve.tool import Tool
from eve.agent import Agent
from eve.models import Model
from eve.llm.llm import LLMContextMetadata, LLMTraceMetadata, LLMContext
from eve.llm.config import get_default_session_llm_config, LLMConfig
from eve.session.message import ChatMessage
from eve.session.session import Session

from eve.agent.session.models import PromptSessionContext, ChatMessageRequestInput

from eve.agent.session.session_prompts import (
    system_template,
    model_template,
)



DEFAULT_SESSION_SELECTION_LIMIT = 25


def select_messages(
    session: Session, 
    selection_limit: Optional[int] = DEFAULT_SESSION_SELECTION_LIMIT
):
    messages = ChatMessage.get_collection()
    selected_messages = messages.find({
        "session": session.id, 
        "role": {"$ne": "eden"}
    }).sort(
        "createdAt", -1
    )
    if selection_limit is not None:
        selected_messages = selected_messages.limit(selection_limit)
    selected_messages = list(selected_messages)

    pinned_messages = messages.find({
        "session": session.id, 
        "pinned": True
    })
    pinned_messages = list(pinned_messages)
    pinned_messages = [m for m in pinned_messages if m["_id"] not in [msg["_id"] for msg in selected_messages]]    
    selected_messages.extend(pinned_messages)

    selected_messages.reverse()
    selected_messages = [ChatMessage(**msg) for msg in selected_messages]    
    
    # Filter out cancelled tool calls from the messages
    # Todo: is this needed?
    # #### selected_messages = [msg.filter_cancelled_tool_calls() for msg in selected_messages]

    return selected_messages
    


def convert_message_roles(messages: List[ChatMessage], actor_id: ObjectId):
    """
    Re-assembles messages from perspective of actor (assistant) and everyone else (user)
    """
    
    # Get sender name mapping for all messages
    # sender_name_map = get_sender_id_to_sender_name_map(messages)
    
    converted_messages = []
    for message in messages:
        if message.sender == actor_id:
            converted_messages.append(message.as_assistant_message())
        else:
            user_message = message.as_user_message()
            # Include sender name in the message content if available
            # if message.sender and message.sender in sender_name_map:
            #     sender_name = sender_name_map[message.sender]
            #     # Prepend the sender name to the content
            #     user_message.content = f"[{sender_name}]: {user_message.content}"
            converted_messages.append(user_message)
    
    return converted_messages


async def build_system_message(
    session: Session,
    actor: Agent,
    context: PromptSessionContext,
    tools: Dict[str, Tool],
):  # Get the last speaker ID for memory prioritization
    last_speaker_id = None
    if context.initiating_user_id:
        last_speaker_id = ObjectId(context.initiating_user_id)

    # Get agent memory context
    memory_context = ""
    try:
        memory_context = ""
        # memory_context = await assemble_memory_context(
        #     session,
        #     actor.id,
        #     last_speaker_id=last_speaker_id,
        #     reason="build_system_message",
        #     agent=actor
        # )
        # if memory_context:
        #     memory_context = f"\n\n{memory_context}"
    except Exception as e:
        print(
            f"Warning: Failed to load memory context for agent {actor.id} in session {session.id}: {e}"
        )

    # Get text describing models
    lora_name = None
    if actor.models:
        models_collection = get_collection(Model.collection_name)
        loras_dict = {m["lora"]: m for m in actor.models}
        lora_docs = models_collection.find(
            {"_id": {"$in": list(loras_dict.keys())}, "deleted": {"$ne": True}}
        )
        lora_docs = list(lora_docs or [])
        if lora_docs:
            lora_name = lora_docs[0]["name"]
        for doc in lora_docs:
            doc["use_when"] = loras_dict[ObjectId(doc["_id"])].get(
                "use_when", "This is your default Lora model"
            )
        loras = "\n".join(model_template.render(doc) for doc in lora_docs or [])
    else:
        loras = ""

    # Build system prompt with memory context
    base_content = system_template.render(
        name=actor.name,
        current_date_time=datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S"),
        description=actor.description,
        persona=actor.persona,
        scenario=session.scenario,
        loras=loras,
        lora_name=lora_name,
        voice=actor.voice,
        tools=tools,
    )

    content = f"{base_content}{memory_context}"
    return ChatMessage(
        session=session.id, sender=ObjectId(actor.id), role="system", content=content
    )

    
async def build_system_extras(
    session: Session, context: PromptSessionContext, config: LLMConfig
):
    extras = []
    if context.update_config and context.update_config.farcaster_hash:
        extras.append(
            ChatMessage(
                session=session.id,
                sender=ObjectId("000000000000000000000000"),
                role="system",
                content="You are currently replying to a Farcaster cast. The maximum length before the fold is 320 characters, and the maximum length is 1024 characters, so attempt to be concise in your response.",
            )
        )
        config.max_tokens = 1024
    return context, config, extras


from eve.llm.config import build_llm_config_from_agent_settings
async def build_llm_context(
    session: Session,
    actor: Agent,
    context: PromptSessionContext,
    trace_id: Optional[str] = None,
):
    user = User.from_mongo(context.initiating_user_id)
    tier = "premium" if user.subscriptionTier and user.subscriptionTier > 0 else "free"
    
    tools = actor.get_tools(cache=False, auth_user=context.initiating_user_id)
    if context.custom_tools:
        tools.update(context.custom_tools)
    # build messages first to have context for thinking routing
    system_message = await build_system_message(session, actor, context, tools)
    messages = [system_message]
    # context, base_config, system_extras = await build_system_extras(session, context, context.llm_config or get_default_session_llm_config(tier))
    # if len(system_extras) > 0:
    #     messages.extend(system_extras)
    existing_messages = select_messages(session)
    messages.extend(existing_messages)
    messages = convert_message_roles(messages, actor.id)

    # Use agent's llm_settings if available, otherwise fallback to context or default
    if actor.llm_settings:
        config = await build_llm_config_from_agent_settings(
            actor.llm_settings, 
            tier, 
            thinking_override=getattr(context, 'thinking_override', None),
            context_messages=existing_messages  # Pass existing messages for routing context
        )
    else:
        config = context.llm_config or get_default_session_llm_config(tier)


    # print(type(tools))
    # print(tools.keys())
    # # tools = tools[:2]
    # print(type(tools))
    # tools = {t: Tool.load(t) for t in tools}
    # tools = [Tool.load(t) for t in tools]


    


    config = LLMConfig(model="gpt-4o")

    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print(tools)
    print(type(config))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("go!")

    print(type(tools))
    

    return LLMContext(
        messages=messages,
        tools=tools,
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

