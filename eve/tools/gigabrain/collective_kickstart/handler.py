import os
import isodate
import asyncio
import aiohttp
import pytz
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from collections import defaultdict
from bson import ObjectId
from loguru import logger

from eve.tool import ToolContext
from eve.agent.agent import Agent, AgentPermission
from eve.agent.session.models import Session, ChatMessage, LLMContext, LLMConfig
from eve.agent.session.memory_models import messages_to_text
from eve.agent.session.session_llm import async_prompt
from eve.agent.session.session_prompts import system_template
from eve.agent.session.memory_assemble_context import assemble_memory_context
from eve.concepts import Concept
from eve.user import User
from eve.utils import serialize_json

def parse_timedelta_string(timedelta_str: str) -> timedelta:
    """
    Parse an ISO 8601 timedelta string into a timedelta object.

    Args:
        timedelta_str: ISO 8601 duration string like "PT2H", "P1D", "P3D"

    Returns:
        timedelta object

    Raises:
        ValueError: If format is invalid
    """
    try:
        duration = isodate.parse_duration(timedelta_str)
        # isodate returns a Duration or timedelta object
        if isinstance(duration, timedelta):
            return duration
        else:
            # If it's a Duration object, convert to timedelta
            # Duration objects represent months/years which need a reference date
            # For our purposes, we'll estimate months as 30 days
            return duration.totimedelta(start=datetime.now(timezone.utc))
    except Exception as e:
        raise ValueError(
            f"Invalid ISO 8601 timedelta format: '{timedelta_str}'. "
            f"Expected format like 'PT2H' (2 hours), 'P1D' (1 day), or 'P3D' (3 days). Error: {str(e)}"
        )


def check_agent_owner_permission(agent: Agent, user_id: ObjectId) -> bool:
    """
    Check if a user is an owner of the agent.

    Args:
        agent: Agent object
        user_id: User ObjectId to check

    Returns:
        True if user is an owner, False otherwise
    """
    # Check if user is the primary owner
    if agent.owner == user_id:
        return True

    # Check agent_permissions collection for owner-level permission
    permission = AgentPermission.find_one({
        "agent": agent.id,
        "user": user_id,
        "level": "owner"
    })

    return permission is not None


def fetch_recent_sessions(agent_id: ObjectId, cutoff_time: datetime) -> List[Session]:
    """
    Fetch sessions created within the time window for this agent.
    Excludes sub-sessions (sessions with a parent_session field).

    Args:
        agent_id: Agent ObjectId
        cutoff_time: Datetime to filter sessions after

    Returns:
        List of Session objects
    """
    sessions = Session.find({
        "agents": agent_id,
        "createdAt": {"$gte": cutoff_time},
        "status": "active",
        "$or": [
            {"parent_session": {"$exists": False}},
            {"parent_session": None}
        ]
    })

    return list(sessions)


def get_session_user_count(session: Session, messages: List[ChatMessage]) -> int:
    """
    Count the number of unique non-agent users in a session.

    Args:
        session: Session object
        messages: List of messages in the session

    Returns:
        Number of unique non-agent users
    """
    user_ids: Set[ObjectId] = set()

    for msg in messages:
        if msg.sender and msg.role == "user":
            user_ids.add(msg.sender)

    return len(user_ids)


def collect_user_messages(
    sessions: List[Session],
    include_agent_messages: bool = True,
    ignore_multi_user_sessions: bool = True,
    session_cutoff_time: Optional[datetime] = None
) -> tuple[Dict[ObjectId, List[List[ChatMessage]]], Dict[str, ObjectId]]:
    """
    Collect messages grouped by user, with sessions kept separate per user.

    Args:
        sessions: List of Session objects
        include_agent_messages: Whether to include agent messages alongside user messages
        ignore_multi_user_sessions: Whether to filter out sessions with multiple users
        session_cutoff_time: Optional datetime cutoff to filter out messages older than this (same as session creation cutoff)

    Returns:
        Tuple of:
        - Dict of {user_id: [[messages_from_session1], [messages_from_session2], ...]}
        - Dict of {username: user_id} mapping for efficient user lookup
    """
    # Structure: {user_id: [[session1 messages], [session2 messages], ...]}
    messages_by_user = defaultdict(list)
    # Structure: {username: user_id}
    username_to_user_id = {}

    for session in sessions:
        # Fetch all messages for this session, sorted by creation time
        messages_list = ChatMessage.find(
            {"session": session.id},
            sort="createdAt",
            desc=False
        )  # Sort chronologically

        # If we're filtering multi-user sessions, check user count
        if ignore_multi_user_sessions:
            user_count = get_session_user_count(session, messages_list)
            if user_count > 1:
                logger.info(f"Skipping session {session.id}: {user_count} users (multi-user)")
                continue

        # Collect messages per user per session
        session_messages_by_user = defaultdict(list)
        current_user = None

        for msg in messages_list:
            # Filter messages by session cutoff time if specified
            if session_cutoff_time and msg.createdAt:
                # Ensure both datetimes are timezone-aware for comparison
                msg_created = msg.createdAt
                if msg_created.tzinfo is None:
                    msg_created = msg_created.replace(tzinfo=timezone.utc)
                if msg_created < session_cutoff_time:
                    continue

            if msg.role == "user" and msg.sender:
                current_user = msg.sender
                session_messages_by_user[msg.sender].append(msg)
            elif msg.role == "assistant" and include_agent_messages and current_user:
                # Include agent messages in the context for the current user
                session_messages_by_user[current_user].append(msg)

        # Add session messages to the user's list of sessions
        for user_id, session_msgs in session_messages_by_user.items():
            if session_msgs:  # Only add non-empty sessions
                messages_by_user[user_id].append(session_msgs)
                # Build username to user_id mapping using session.owner
                if user_id not in username_to_user_id.values() and session.owner:
                    # Get the username for this user_id
                    user = User.from_mongo(user_id)
                    username_to_user_id[user.username] = session.owner

    return messages_by_user, username_to_user_id


def format_user_profiles(
    messages_by_user: Dict[ObjectId, List[List[ChatMessage]]],
    username_to_user_id: Dict[str, ObjectId],
    min_messages: int = 3
) -> Dict[str, str]:
    """
    Format user messages into profile strings using messages_to_text.

    Args:
        messages_by_user: Dict of {user_id: [[session1_messages], [session2_messages], ...]}
        username_to_user_id: Dict of {username: user_id} for efficient lookup
        min_messages: Minimum number of user messages required to include user

    Returns:
        Dict of {username: profile_text}
    """
    user_profiles = {}

    for user_id, session_messages_list in messages_by_user.items():
        try:
            # Count only user messages for the threshold check across all sessions
            user_message_count = sum(
                1 for session_msgs in session_messages_list
                for msg in session_msgs if msg.role == "user"
            )

            if user_message_count < min_messages:
                logger.info(
                    f"Skipping user {user_id}: only {user_message_count} user messages "
                    f"(minimum: {min_messages})"
                )
                continue

            # Get user object (still needed for username)
            user = User.from_mongo(user_id)

            # Build profile text from all sessions
            if session_messages_list:
                session_texts = []
                total_messages = 0

                # Convert each session's messages to text
                for session_msgs in session_messages_list:
                    if session_msgs:
                        profile_text, _ = messages_to_text(session_msgs, skip_trigger_messages=True)
                        if profile_text.strip():
                            session_texts.append(profile_text.strip())
                            total_messages += len(session_msgs)

                # Join session texts with separator
                if session_texts:
                    if len(session_texts) > 1:
                        # Multiple sessions - add separator
                        combined_profile = "\n\n--- new session ---\n\n".join(session_texts)
                    else:
                        # Single session - no separator needed
                        combined_profile = session_texts[0]

                    # Prepend username context
                    final_profile = f"Profile context of username: {user.username}:\n\n{combined_profile}"
                    user_profiles[user.username] = final_profile
                    logger.info(
                        f"Added profile for user {user.username} "
                        f"({user_message_count} user messages, {total_messages} total messages, "
                        f"{len(session_texts)} sessions)"
                    )

        except Exception as e:
            logger.error(f"Error processing user {user_id}: {e}")
            continue

    return user_profiles


async def call_profile_matching_handler(user_profiles: Dict[str, str], agent_id: ObjectId, user_id: ObjectId) -> Dict:
    """
    Call the profile_matching tool's handler directly.

    Args:
        user_profiles: Dict of {username: profile_text}
        agent_id: Agent ID
        user_id: User ID (agent owner)

    Returns:
        Profile matching results dict
    """
    # Import the profile_matching handler
    from eve.tools.gigabrain.profile_matching.handler import handler as profile_matching_handler
    from eve.tool import ToolContext

    # Create a mock context for the profile_matching tool
    mock_context = ToolContext(
        args={
            "user_profiles": user_profiles,
            "config_path": "config/config.yaml",
            "force": True
        },
        agent=str(agent_id),
        user=str(user_id)
    )

    # Call the handler
    result = await profile_matching_handler(mock_context)

    # Get the output and convert numpy types to native Python types for JSON serialization
    output = result.get("output", {})
    if isinstance(output, dict):
        output = serialize_json(output)

    return output


async def generate_personalized_message(
    agent: Agent,
    matching_data: str,
    user: User,
    session: Session
) -> str:
    """
    Use the agent to generate a personalized message based on matching results.
    Uses the same system message assembly as build_system_message() in session.py.

    Args:
        agent: Agent object to use for generation
        matching_data: Raw matching results to be reformatted
        user: User object to personalize for
        session: Session object for context

    Returns:
        Generated personalized message string
    """
    instruction_prompt = f"""(This is an automatic system instruction that will not be visible to the user)
You are {agent.name}, and you've identified some interesting connections for {user.username} based on their profile and conversation history.

Based on the following matching results (coming from your analysis in another session), create a warm, personalized message introducing these potential collaborators or connections to the user. Make it feel natural and conversational, highlighting why these connections might be valuable in the context of your purpose and goals.

Matching Results:
{matching_data}

Generate a friendly, personalized introduction message that presents these connections in an engaging way and inspires the user to reach out and start collaborating."""

    # Build system message using same components as build_system_message() in session.py
    # Only including essential components (concepts, date/time, persona, memory)

    # Get concepts
    concepts = Concept.find({"agent": agent.id})

    # Get current date/time
    current_date_time = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    memory = await assemble_memory_context(
        session,
        agent,
        user,
        reason="generating_personalized_message",
    )

    # Build system prompt with memory context using the same template
    system_content = system_template.render(
        name=agent.name,
        # current_date_time=current_date_time,
        description=agent.description,
        scenario=None,
        persona=agent.persona,
        tools=None,
        concepts=concepts,
        loras=None,
        voice=None,
        memory=memory,
    )
    
    system_message = ChatMessage(
        session=session.id,
        role="system",
        content=system_content
    )

    # Build LLM context with full system message
    context = LLMContext(
        messages=[
            system_message,
            ChatMessage(role="user", content=instruction_prompt, session=session.id)
        ],
        config=LLMConfig(model="claude-sonnet-4-5"),
    )

    # Generate the response
    response = await async_prompt(context)

    return response.content


async def create_user_session_with_message(
    agent_id: ObjectId,
    user: User,
    intro_text: str,
    agent_username: str,
    agent: Agent
) -> Dict:
    """
    Create a new session with a user and insert an agent-generated message.
    The agent generates the message organically based on the intro_text input.

    Args:
        agent_id: Agent ObjectId
        user: User object
        intro_text: Raw matching data to be reformatted by the agent
        agent_username: Agent username for logging
        agent: Agent object for generating personalized message

    Returns:
        Dict with session creation result
    """
    # Create unique session key
    session_key = f"collective-kickstart-{agent_id}-{user.id}-{datetime.now(timezone.utc).isoformat()}"

    try:
        # Create the session first
        session = Session(
            owner=user.id,
            agents=[agent_id],
            session_key=session_key,
            users=[user.id],
            platform=None,
            status="active",
            title=f"Collective Intelligence Kickstart",
        )

        # Save the session
        session.save()

        # Generate personalized message using the agent (with session context)
        logger.info(f"Generating personalized message for user {user.username}")
        generated_agent_message = await generate_personalized_message(agent, intro_text, user, session)

        # Create and save the agent message with generated content
        agent_message = ChatMessage(
            session=session.id,
            role="assistant",
            content=generated_agent_message,
            sender=agent_id,
            createdAt=datetime.now(timezone.utc)
        )
        agent_message.save()

        logger.info(f"Created session {session.id} with agent-generated message for user {user.username}")

        return {
            "user": user.username,
            "user_id": str(user.id),
            "session_id": str(session.id),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error creating session for user {user.username}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "user": user.username,
            "user_id": str(user.id),
            "status": "failed",
            "error": str(e)
        }

async def handler(context: ToolContext):
    """
    Main handler for collective_kickstart tool.

    Workflow:
    1. Check agent owner permissions
    2. Fetch recent sessions within time window
    3. Group messages by user
    4. Format user profiles using messages_to_text
    5. Call profile_matching tool
    6. Create personalized intro sessions for each user
    """
    # Validate that agent context exists
    if not context.agent:
        return {
            "output": "Error: This tool requires an agent context."
        }

    # Validate that user context exists
    if not context.user:
        return {
            "output": "Error: This tool requires a user context."
        }

    # Load agent
    agent = Agent.from_mongo(context.agent)

    # Check permissions - must be agent owner
    is_owner = check_agent_owner_permission(agent, ObjectId(context.user))

    if not is_owner:
        return {
            "output": "This tool can only be run by an agent owner."
        }

    # Defensive check: ensure context.args is a dict
    if not isinstance(context.args, dict):
        logger.error(f"context.args is not a dict, it's a {type(context.args)}: {context.args}")
        return {
            "output": f"Error: Tool received invalid arguments format. Expected dict, got {type(context.args).__name__}."
        }

    # Extract and validate parameters
    time_window_str = context.args.get("time_window", "P2H")
    min_messages = context.args.get("min_messages", 2)
    include_agent_messages = context.args.get("include_agent_messages", True)
    ignore_multi_user_sessions = context.args.get("ignore_multi_user_sessions", True)

    logger.info(f"Starting collective_kickstart for agent {agent.username}")
    logger.info(f"Time window: {time_window_str}, Min messages: {min_messages}")
    logger.info(f"Include agent messages: {include_agent_messages}, Ignore multi-user sessions: {ignore_multi_user_sessions}")

    try:
        # Parse time window
        time_delta = parse_timedelta_string(time_window_str)
        cutoff_time = datetime.now(timezone.utc) - time_delta

        logger.info(f"Fetching sessions created after {cutoff_time}")
        logger.info(f"Filtering messages created after {cutoff_time} (same as session cutoff)")

        # Step 1: Fetch recent sessions
        sessions = fetch_recent_sessions(agent.id, cutoff_time)

        if not sessions:
            return {
                "output": f"No sessions found in the last {time_window_str}."
            }

        logger.info(f"Found {len(sessions)} sessions in time window")

        # Step 2: Collect and group user messages
        messages_by_user, username_to_user_id = collect_user_messages(
            sessions,
            include_agent_messages=include_agent_messages,
            ignore_multi_user_sessions=ignore_multi_user_sessions,
            session_cutoff_time=cutoff_time
        )

        if not messages_by_user:
            return {
                "output": f"No user messages found in the {len(sessions)} sessions from the last {time_window_str}."
            }

        logger.info(f"Found messages from {len(messages_by_user)} unique users")

        # Step 3: Format user profiles
        user_profiles = format_user_profiles(messages_by_user, username_to_user_id, min_messages)

        if not user_profiles:
            return {
                "output": f"No users met the minimum message threshold of {min_messages} messages."
            }

        logger.info(f"Created profiles for {len(user_profiles)} users")

        # Step 4: Call profile matching
        logger.info("Calling profile_matching tool...")
        matching_results = await call_profile_matching_handler(user_profiles, agent.id, ObjectId(context.user))

        # Check for errors - handle both string and dict responses
        if isinstance(matching_results, str):
            return {
                "output": [f"Profile matching failed: {matching_results}"]
            }

        if not isinstance(matching_results, dict):
            return {
                "output": [f"Profile matching returned unexpected type: {type(matching_results)}"]
            }

        if matching_results.get("error"):
            error_msg = matching_results.get("error")
            return {
                "output": [f"Profile matching failed: {error_msg}"]
            }

        # Parse the cohort_summary structure
        users_data = matching_results.get("users", {})

        if not users_data:
            return {
                "output": ["Profile matching completed but no matches were generated."]
            }

        logger.info(f"Profile matching generated results for {len(users_data)} users")

        # Step 5: Extract intro documents per user
        # Build a map of username -> intro text
        user_intros = {}
        for username, user_data in users_data.items():
            matches = user_data.get("matches", [])
            for match in matches:
                partner = match.get("partner", "")
                intro = match.get("intro", "")

                if intro:
                    if username not in user_intros:
                        user_intros[username] = []
                    # Format the intro with partner information
                    formatted_intro = f"You might want to connect with {partner}:\n\n{intro}"
                    user_intros[username].append(formatted_intro)

        if not user_intros:
            return {
                "output": "Profile matching completed but no introductions were generated."
            }

        logger.info(f"Generated introductions for {len(user_intros)} users")

        # Step 6: Create sessions for each user with their intro
        session_results = []

        for username, intros in user_intros.items():
            # Combine multiple intros if user appears in multiple matches
            combined_intro = "\n\n".join(intros)

            # Get user_id from the mapping we built earlier
            try:
                user_id = username_to_user_id.get(username)
                if not user_id:
                    logger.warning(f"Could not find user_id for username: {username}")
                    continue

                # Get User object from user_id
                user = User.from_mongo(user_id)

                # Create session with agent-generated message
                result = await create_user_session_with_message(
                    agent.id,
                    user,
                    combined_intro,
                    agent.username,
                    agent
                )

                session_results.append(result)

            except Exception as e:
                logger.error(f"Error creating session for user {username}: {e}")
                session_results.append({
                    "user": username,
                    "status": "failed",
                    "error": str(e)
                })

        # Compile results summary
        successful = [r for r in session_results if r["status"] == "success"]
        failed = [r for r in session_results if r["status"] == "failed"]

        summary = f"""Collective Kickstart Complete!

Time Window: {time_window_str}
Sessions Analyzed: {len(sessions)}
Active Users Found: {len(user_profiles)}
Introductions Generated: {len(user_intros)}

Session Creation Results:
✓ Successful: {len(successful)}
✗ Failed: {len(failed)}

"""

        if successful:
            summary += "Successfully created sessions for:\n"
            for result in successful:
                summary += f"  - {result['user']} (session: {result['session_id']})\n"

        if failed:
            summary += "\nFailed to create sessions for:\n"
            for result in failed:
                error = result.get('error', 'Unknown error')
                summary += f"  - {result['user']}: {error}\n"

        return {"output": summary}

    except ValueError as e:
        return {
            "output": f"Invalid parameter: {str(e)}"
        }

    except Exception as e:
        logger.error(f"Error in collective_kickstart handler: {e}")
        import traceback
        traceback.print_exc()
        return {
            "output": f"Error running collective_kickstart: {str(e)}"
        }
