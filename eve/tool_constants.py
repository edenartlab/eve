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
    "boolean", "string", "integer", "float", "array", "image", "video", "audio", "lora"
]

HANDLERS = Literal["local", "modal", "comfyui", "replicate", "gcp", "fal", "mcp"]

BASE_MODELS = Literal[
    "sd15",
    "sdxl",
    "flux-dev",
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
    "seedream4",
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
    "ace_step_musicgen",
    "elevenlabs",
    "vibevoice",
    "mmaudio",
    "thinksound",
    "stable_audio",
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

DISCORD_TOOLS = ["discord_post", "discord_search", "discord_broadcast_dm"]
TELEGRAM_TOOLS = ["telegram_post"]
TWITTER_TOOLS = ["tweet", "twitter_mentions", "twitter_search"]
FARCASTER_TOOLS = ["farcaster_cast", "farcaster_mentions", "farcaster_search"]
SHOPIFY_TOOLS = ["shopify"]
PRINTIFY_TOOLS = ["printify"]
CAPTIONS_TOOLS = ["captions"]
TIKTOK_TOOLS = ["tiktok_post"]
EMAIL_TOOLS = ["email_send"]
GMAIL_TOOLS = ["gmail_send"]

SOCIAL_MEDIA_TOOLS = [
    *TWITTER_TOOLS,
    *DISCORD_TOOLS,
    *FARCASTER_TOOLS,
    *TELEGRAM_TOOLS,
    *SHOPIFY_TOOLS,
    *PRINTIFY_TOOLS,
    *CAPTIONS_TOOLS,
    *TIKTOK_TOOLS,
    *EMAIL_TOOLS,
    *GMAIL_TOOLS,
]

EDEN_DB_TOOLS = [
    "search_collections",
    "add_to_collection",
    "search_agents",
    "search_models",
]

CONTEXT7_MCP_TOOLS = ["context7_resolve_library_id"]
CALCULATOR_MCP_TOOLS = ["calculator_calculate"]

TOOL_SETS = {
    "create_image": ["create", "media_editor"],  #  "magic_8_ball"
    "create_video": [],  # deprecated
    "create_audio": ["elevenlabs", "elevenlabs_music", "elevenlabs_fx"],
    "vj_tools": ["texture_flow", "video_FX", "reel"],
    "news": [],  # deprecated
    "manage_collections": ["search_collections", "add_to_collection"],
    "social_media_tools": SOCIAL_MEDIA_TOOLS,
    "context7_mcp_tools": CONTEXT7_MCP_TOOLS,
    "calculator_mcp_tools": CALCULATOR_MCP_TOOLS,
    "legacy_tools": [
        "legacy_create",
        "legacy_interpolate",
        "legacy_controlnet",
        "legacy_real2real",
        "legacy_txt2vid",
    ],
    "all_tools": ALL_TOOLS,
}

BASE_TOOLS = [
    "create",
    "media_editor",
]

FLUX_LORA_TXT2IMG_TOOLS = ["flux_dev_lora", "flux_dev"]

SDXL_LORA_TXT2IMG_TOOLS = ["txt2img"]

AGENTIC_TOOLS = [*SOCIAL_MEDIA_TOOLS, "reel"]
