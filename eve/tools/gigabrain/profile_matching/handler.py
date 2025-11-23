import json

import modal
from bson import ObjectId
from loguru import logger

from eve.agent.agent import Agent, AgentPermission
from eve.tool import ToolContext


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
    permission = AgentPermission.find_one(
        {"agent": agent.id, "user": user_id, "level": "owner"}
    )

    return permission is not None


async def handler(context: ToolContext):
    """
    Handler for Gigabrain profile matching tool.

    Calls the Modal app to run the matching pipeline remotely.
    """
    # Validate that agent context exists
    if not context.agent:
        return {"output": "Error: This tool requires an agent context."}

    # Validate that user context exists
    if not context.user:
        return {"output": "Error: This tool requires a user context."}

    # Load agent
    agent = Agent.from_mongo(context.agent)

    # Check permissions - must be agent owner
    is_owner = check_agent_owner_permission(agent, ObjectId(context.user))

    if not is_owner:
        return {"output": "This tool can only be run by an agent owner."}

    args = context.args

    # Required parameters
    user_profiles = args["user_profiles"]  # Dict of user_id -> profile_text

    # Optional parameters with defaults
    config_path = args.get("config_path", "config/config.yaml")
    group_name = args.get("group_name")
    force = args.get("force", False)

    logger.info("Calling Gigabrain Modal app for profile matching...")
    logger.info(f"Number of user profiles: {len(user_profiles)}")
    logger.info(f"Config path: {config_path}")
    if group_name:
        logger.info(f"Group name: {group_name}")
    logger.info(f"Force re-run: {force}")

    try:
        # Convert user_profiles dict to JSON string
        user_profiles_json = json.dumps(user_profiles)

        # Get the Modal function
        run_matching_pipeline = modal.Function.from_name(
            "profile-matching", "run_matching_pipeline"
        )

        # Call the function remotely
        result = run_matching_pipeline.remote(
            user_profiles_json=user_profiles_json,
            config_path=config_path,
            group_name=group_name,
            force=force,
        )

        logger.info("Matching pipeline completed successfully")

        # Log results - check for error or success
        if result.get("error"):
            logger.warning(f"Pipeline returned with errors: {result.get('error')}")
        elif result.get("users"):
            num_users = len(result.get("users", {}))
            logger.info(f"Generated matches for {num_users} users")

        return {
            "output": result,
        }

    except Exception as e:
        logger.error(f"Error running matching pipeline: {e}")
        raise
