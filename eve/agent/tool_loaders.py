"""
Tool loaders - dynamic parameter injection and filtering for platform-specific tools
"""

from typing import Dict, Any, Callable, Optional, List
from loguru import logger
import sentry_sdk
import traceback


# Registry of tool parameter loaders
# Maps tool name -> loader function that takes deployment and returns parameter updates
TOOL_PARAMETER_LOADERS: Dict[str, Callable] = {}


# Platform-to-tools mapping
PLATFORM_TOOL_SETS = {
    "discord": None,  # Lazy-loaded to avoid circular imports
    "telegram": None,
    "twitter": None,
    "farcaster": None,
    "shopify": None,
    "printify": None,
    "captions": None,
    "tiktok": None,
}


def _get_platform_tool_sets():
    """Lazy-load platform tool sets to avoid circular imports"""
    global PLATFORM_TOOL_SETS
    if PLATFORM_TOOL_SETS["discord"] is None:
        from ..tool_constants import (
            DISCORD_TOOLS,
            TELEGRAM_TOOLS,
            TWITTER_TOOLS,
            FARCASTER_TOOLS,
            SHOPIFY_TOOLS,
            PRINTIFY_TOOLS,
            CAPTIONS_TOOLS,
            TIKTOK_TOOLS,
            EMAIL_TOOLS,
            GMAIL_TOOLS,
        )

        PLATFORM_TOOL_SETS.update(
            {
                "discord": DISCORD_TOOLS,
                "telegram": TELEGRAM_TOOLS,
                "twitter": TWITTER_TOOLS,
                "farcaster": FARCASTER_TOOLS,
                "shopify": SHOPIFY_TOOLS,
                "printify": PRINTIFY_TOOLS,
                "captions": CAPTIONS_TOOLS,
                "tiktok": TIKTOK_TOOLS,
                "email": EMAIL_TOOLS,
                "gmail": GMAIL_TOOLS,
            }
        )
    return PLATFORM_TOOL_SETS


def get_agent_specific_tools(username: str) -> List[str]:
    """
    Get agent-specific tools based on username

    Args:
        username: Agent username

    Returns:
        List of tool names specific to this agent
    """
    # TODO: systemize this for other agents
    if username == "abraham":
        return [
            "abraham_publish",
            "abraham_daily",
            "abraham_covenant",
            "abraham_rest",
            "abraham_seed",
        ]
    elif username == "verdelis":
        return ["verdelis_story"]
    
    if "gigabrain" in username.lower():
        from ..tool_constants import GIGABRAIN_TOOLS
        return GIGABRAIN_TOOLS

    return []


def load_lora_docs(models: Optional[List[Dict]], models_collection) -> List[Dict]:
    """
    Load LoRA documents from MongoDB

    Args:
        models: List of model configs from agent
        models_collection: MongoDB collection for models

    Returns:
        List of LoRA documents
    """
    if not models:
        return []

    loras_dict = {m["lora"]: m for m in models}
    lora_ids = list(loras_dict.keys())

    # Single batch query for all loras
    lora_docs = list(
        models_collection.find({"_id": {"$in": lora_ids}, "deleted": {"$ne": True}})
    )

    return lora_docs


def load_deployments(agent_id, deployment_class) -> Dict[str, Any]:
    """
    Load deployments from MongoDB

    Args:
        agent_id: Agent ObjectId
        deployment_class: Deployment model class

    Returns:
        Dict mapping platform names to deployment objects
    """
    from bson import ObjectId

    return {
        deployment.platform.value: deployment
        for deployment in deployment_class.find({"agent": ObjectId(str(agent_id))})
    }


def register_tool_loader(tool_name: str):
    """Decorator to register a tool parameter loader"""

    def decorator(func: Callable):
        TOOL_PARAMETER_LOADERS[tool_name] = func
        return func

    return decorator


@register_tool_loader("discord_post")
def load_discord_post_parameters(deployment) -> Optional[Dict[str, Any]]:
    """
    Load dynamic parameters for discord_post tool based on deployment config

    Returns parameter updates dict with channel_id choices and tips
    """
    try:
        allowed_channels = deployment.get_allowed_channels()
        if not allowed_channels:
            return None

        channels_description = " | ".join(
            [f"ID {c.id} ({c.note})" for c in allowed_channels]
        )

        return {
            "channel_id": {
                "choices": [c.id for c in allowed_channels],
                "tip": f"Some hints about the available channels: {channels_description}",
            },
        }
    except Exception as e:
        logger.error(f"Error loading discord_post parameters: {e}")
        return None


@register_tool_loader("telegram_post")
def load_telegram_post_parameters(deployment) -> Optional[Dict[str, Any]]:
    """
    Load dynamic parameters for telegram_post tool based on deployment config

    Returns parameter updates dict with channel_id choices and tips
    """
    try:
        allowed_channels = deployment.get_allowed_channels()
        if not allowed_channels:
            return None

        channels_description = " | ".join(
            [f"ID {c.id} ({c.note})" for c in allowed_channels]
        )

        return {
            "channel_id": {
                "choices": [c.id for c in allowed_channels],
                "tip": f"Some hints about the available topics: {channels_description}",
            },
        }
    except Exception as e:
        logger.error(f"Error loading telegram_post parameters: {e}")
        return None


def inject_deployment_parameters(
    tools: Dict, deployments: Dict, agent_username: str = None
) -> Dict:
    """
    Inject deployment-specific parameters into tools

    Args:
        tools: Dict of tool_name -> tool object
        deployments: Dict of platform_name -> deployment object
        agent_username: Optional agent username for logging

    Returns:
        Updated tools dict
    """
    for tool_name, tool in tools.items():
        if tool_name not in TOOL_PARAMETER_LOADERS:
            continue

        # Infer platform from tool name (e.g., "discord_post" -> "discord")
        platform = tool_name.split("_")[0]

        if platform not in deployments:
            continue

        deployment = deployments[platform]
        loader = TOOL_PARAMETER_LOADERS[tool_name]

        try:
            param_updates = loader(deployment)
            if param_updates:
                tool.update_parameters(param_updates)
        except Exception as e:
            logger.error(f"Error loading parameters for {tool_name}: {e}")
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("component", "tool_loader")
                scope.set_tag("tool_name", tool_name)
                scope.set_tag("platform", platform)
                if agent_username:
                    scope.set_tag("agent_username", agent_username)
                scope.set_context(
                    "tool_loader_context",
                    {
                        "tool_name": tool_name,
                        "platform": platform,
                        "agent_username": agent_username,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                sentry_sdk.capture_exception(e)

    return tools


def remove_non_deployed_platform_tools(tools: Dict, deployments: Dict) -> Dict:
    """
    Remove tools for platforms that are not deployed

    Args:
        tools: Dict of tool_name -> tool object
        deployments: Dict of platform_name -> deployment object

    Returns:
        Updated tools dict
    """
    platform_tool_sets = _get_platform_tool_sets()

    for platform, tool_list in platform_tool_sets.items():
        if platform not in deployments:
            for tool in tool_list:
                tools.pop(tool, None)

    return tools


def filter_tools_by_feature_flags(
    tools: Dict, feature_flags: Optional[List[str]], premium_tools: Dict[str, List[str]]
) -> Dict:
    """
    Remove premium tools based on feature flags

    Args:
        tools: Dict of tool_name -> tool object
        feature_flags: List of enabled feature flags
        premium_tools: Dict mapping feature flag names to lists of tools requiring that flag

    Returns:
        Updated tools dict
    """
    if not feature_flags:
        return tools

    # For now, hardcoded logic for premium social media tools
    # Can be extended for other feature-gated tools
    if "tool_access_premium_social_media" not in feature_flags:
        # Import here to avoid circular dependency
        from ..tool_constants import TWITTER_TOOLS

        for tool in TWITTER_TOOLS:
            if tool != "tweet":  # Basic tweet is allowed
                tools.pop(tool, None)

    return tools


def inject_lora_parameters(
    tools: Dict, lora_docs: List[Dict], models: List[Dict], agent_username: str = None
) -> Dict:
    """
    Inject LoRA defaults and tips into tools that use loras

    Args:
        tools: Dict of tool_name -> tool object
        lora_docs: List of LoRA documents from mongo
        models: List of model configs from agent
        agent_username: Optional agent username for logging

    Returns:
        Updated tools dict
    """
    if not lora_docs:
        return tools

    for tool_name, tool in tools.items():
        try:
            if "lora" not in tool.parameters:
                continue

            # Build LoRA information for the tip
            lora_info = []
            for lora_doc in lora_docs:
                lora_id = lora_doc["_id"]
                agent_lora_config = {m["lora"]: m for m in models}[lora_id]
                lora_info.append(
                    f"{{ ID: {lora_id}, Name: {lora_doc['name']}, Description: {lora_doc['lora_trigger_text']}, Use When: {agent_lora_config['use_when']} }}"
                )

            params = {
                "lora": {
                    "default": str(lora_docs[0]["_id"]),
                    "tip": "Users may request one of your known LoRAs, or a different unknown one, or no LoRA at all. When referring to a LoRA, strictly use its name, not its description. Notes on when to use the known LoRAs: "
                    + " | ".join(lora_info),
                },
            }

            if "use_lora" in tool.parameters:
                params["use_lora"] = {"default": True}

            tool.update_parameters(params)

        except Exception as e:
            logger.error(f"Error injecting lora parameters for {tool_name}: {e}")
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("component", "tool_loader")
                scope.set_tag("operation", "lora_parameter_injection")
                scope.set_tag("tool_name", tool_name)
                if agent_username:
                    scope.set_tag("agent_username", agent_username)
                scope.set_context(
                    "lora_injection_context",
                    {
                        "tool_name": tool_name,
                        "agent_username": agent_username,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )
                sentry_sdk.capture_exception(e)

    return tools


def inject_voice_parameters(
    tools: Dict, voice: Optional[str], agent_username: str = None
) -> Dict:
    """
    Inject voice parameter for elevenlabs tool

    Args:
        tools: Dict of tool_name -> tool object
        voice: Voice ID to use
        agent_username: Optional agent username for logging

    Returns:
        Updated tools dict
    """
    if not voice:
        return tools

    try:
        if "elevenlabs" in tools:
            tools["elevenlabs"].update_parameters({"voice": {"default": voice}})
    except Exception as e:
        logger.error(f"Error injecting voice parameters: {e}")
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("component", "tool_loader")
            scope.set_tag("operation", "elevenlabs_voice_update")
            if agent_username:
                scope.set_tag("agent_username", agent_username)
            scope.set_context(
                "voice_injection_context",
                {
                    "voice": voice,
                    "agent_username": agent_username,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            sentry_sdk.capture_exception(e)

    return tools
