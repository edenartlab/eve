"""
Constants used by both tool.py and agent.py, extracted to break circular imports.
"""

from pathlib import Path
from typing import Literal


def _discover_tools(subfolder: str):
    tools_path = Path(__file__).parent / "tools" / subfolder
    tools = [
        item.name
        for item in tools_path.iterdir()
        if item.is_dir()
        and not item.name.startswith(".")
        and not item.name.startswith("__")
    ]
    return tools


OUTPUT_TYPES = Literal[
    "boolean",
    "string",
    "integer",
    "float",
    "array",
    "image",
    "video",
    "audio",
    "lora",
    "object",
]

HANDLERS = Literal["local", "modal", "comfyui", "replicate", "gcp", "fal", "mcp"]

BASE_MODELS = Literal[
    "sd15",
    "sdxl",
    "flux-dev",
    "flux-schnell",
    "stable-audio-open",
    "inspyrenet-rembg",
    "runway",
    "mmaudio",
    "librosa",
    "musicgen",
    "kling",
    "svd-xt",
    "ltxv",
    "nano-banana",
]

# These tools are default agent tools except Eve
ALL_TOOLS = [
    "flux_dev_lora",
    "flux_dev",
    "flux_kontext",
    "txt2img",
    # more image generation
    "outpaint",
    "seedream3",
    "seedream45",
    # video
    "runway",
    "runway2",
    "runway3",
    "kling_pro",
    "kling",
    "veo2",
    "hedra",
    "seedance1",
    "vid2vid_sdxl",
    "video_FX",
    "texture_flow",
    # audio
    # "ace_step_musicgen",
    "elevenlabs",
    "elevenlabs_dialogue",
    "vibevoice",
    # "mmaudio",
    # "thinksound",
    # "stable_audio",
    "transcription",
    "elevenlabs_music",
    "elevenlabs_fx",
    # editing
    "media_editor",
    # search
    # "search_agents",
    # "search_models",
    # "search_collections",
    # "add_to_collection",
    # misc
    # "news",
    "weather",
    # inactive
    # "ominicontrol",
    # "flux_redux",
    # "reel",
    # "animate_3d",
    "openai_image_edit",
    "openai_image_generate",
]

GIGABRAIN_TOOLS = _discover_tools("gigabrain")
ABRAHAM_TOOLS = _discover_tools("abraham")
VERDELIS_TOOLS = _discover_tools("verdelis")
CHIBA_TOOLS = _discover_tools("chiba")
WZRD_TOOLS = _discover_tools("wzrd")

DISCORD_TOOLS = ["discord_post", "discord_search", "discord_broadcast_dm"]
TELEGRAM_TOOLS = ["telegram_post"]
TWITTER_TOOLS = ["tweet", "twitter_mentions", "twitter_search"]
FARCASTER_TOOLS = ["farcaster_cast", "farcaster_mentions", "farcaster_search"]
INSTAGRAM_TOOLS = ["instagram_post"]
SHOPIFY_TOOLS = ["shopify"]
PRINTIFY_TOOLS = ["printify"]
CAPTIONS_TOOLS = ["captions"]
TIKTOK_TOOLS = ["tiktok_post"]
EMAIL_TOOLS = ["email_send"]
GMAIL_TOOLS = ["gmail_send"]
GOOGLE_CALENDAR_TOOLS = [
    "google_calendar_query",
    "google_calendar_edit",
    "google_calendar_delete",
]

SOCIAL_MEDIA_TOOLS = [
    *TWITTER_TOOLS,
    *DISCORD_TOOLS,
    *FARCASTER_TOOLS,
    *TELEGRAM_TOOLS,
    *INSTAGRAM_TOOLS,
    *SHOPIFY_TOOLS,
    *PRINTIFY_TOOLS,
    *CAPTIONS_TOOLS,
    *TIKTOK_TOOLS,
    *EMAIL_TOOLS,
    *GMAIL_TOOLS,
    *GOOGLE_CALENDAR_TOOLS,
    *INSTAGRAM_TOOLS,
]

EDEN_DB_TOOLS = [
    # "search_collections",
    # "add_to_collection",
    # "search_agents",
    # "search_models",
    "eden_search",
    "get_messages",
    "get_messages_digest",
]

CONTEXT7_MCP_TOOLS = ["context7_resolve_library_id"]
CALCULATOR_MCP_TOOLS = ["calculator_calculate"]
EDEN_MCP_TOOLS = ["eden_ping", "eden_search_creations"]

TOOL_SETS = {
    "create_image": ["create", "media_editor", "reel"],  #  "magic_8_ball"
    "create_video": [],  # deprecated
    "create_audio": [
        "elevenlabs",
        "elevenlabs_dialogue",
        "elevenlabs_music",
        "elevenlabs_fx",
    ],
    "vj_tools": ["texture_flow", "video_FX", "reel"],
    "news": [],  # deprecated
    "manage_collections": ["eden_search", "add_to_collection"],
    "social_media_tools": SOCIAL_MEDIA_TOOLS,
    "context7_mcp_tools": CONTEXT7_MCP_TOOLS,
    "calculator_mcp_tools": CALCULATOR_MCP_TOOLS,
    "eden_mcp_tools": EDEN_MCP_TOOLS,
    "legacy_tools": [
        "legacy_create",
        "legacy_interpolate",
        "legacy_controlnet",
        "legacy_real2real",
        "legacy_txt2vid",
    ],
    "all_tools": ALL_TOOLS,
}

FEATURE_FLAG_TOOL_SETS = {
    "eden_mcp": EDEN_MCP_TOOLS,
}

BASE_TOOLS = [
    "create",
    "media_editor",
]

FLUX_LORA_TXT2IMG_TOOLS = ["flux_dev_lora", "flux_dev"]

SDXL_LORA_TXT2IMG_TOOLS = ["txt2img"]

AGENTIC_TOOLS = [*SOCIAL_MEDIA_TOOLS, "reel"]

# Tools that should skip automatic upload processing of URLs in their results
SKIP_UPLOAD_PROCESSING_TOOLS = [*SOCIAL_MEDIA_TOOLS]
