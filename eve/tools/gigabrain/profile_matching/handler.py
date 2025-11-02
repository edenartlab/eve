import json
import modal
from eve.tool import ToolContext
from loguru import logger


async def handler(context: ToolContext):
    """
    Handler for Gigabrain profile matching tool.

    Calls the Modal app to run the matching pipeline remotely.
    """
    args = context.args

    # Required parameters
    user_profiles = args["user_profiles"]  # Dict of user_id -> profile_text

    # Optional parameters with defaults
    config_path = args.get("config_path", "config/config.yaml")
    group_name = args.get("group_name")
    force = args.get("force", False)

    logger.info(f"Calling Gigabrain Modal app for profile matching...")
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
            "profile-matching",
            "run_matching_pipeline"
        )

        # Call the function remotely
        result = run_matching_pipeline.remote(
            user_profiles_json=user_profiles_json,
            config_path=config_path,
            group_name=group_name,
            force=force,
        )

        logger.info(f"Matching pipeline completed successfully")

        if result.get("success"):
            num_matches = len(result.get("matches", []))
            logger.info(f"Generated {num_matches} matches")
        else:
            logger.warning(f"Pipeline returned with errors: {result.get('error')}")

        return {
            "output": result,
        }

    except Exception as e:
        logger.error(f"Error running matching pipeline: {e}")
        raise
