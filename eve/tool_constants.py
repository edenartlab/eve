"""
Constants used by both tool.py and agent.py, extracted to break circular imports.
"""

from typing import Literal

OUTPUT_TYPES = Literal[
    "boolean", "string", "integer", "float", "array", "image", "video", "audio", "lora"
]

HANDLERS = Literal["local", "modal", "comfyui", "replicate", "gcp", "fal", "mcp"]

BASE_MODELS = Literal[
    "sd15",
    "sdxl",
    "sd3",
    "sd35",
    "flux-dev",
    "flux-schnell",
    "hellomeme",
    "stable-audio-open",
    "inspyrenet-rembg",
    "mochi-preview",
    "runway",
    "mmaudio",
    "librosa",
    "musicgen",
    "kling",
    "svd-xt",
    "wan21",
    "ltxv",
]

# These tools are default agent tools except Eve
ALL_TOOLS = [
    # text-to-image
    "flux_schnell",
    "flux_dev_lora",
    "flux_dev",
    "flux_kontext",
    "txt2img",
    # more image generation
    "flux_inpainting",
    "outpaint",
    "remix_flux_schnell",
    "flux_double_character",
    "seedream3",
    # video
    "runway",
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
    "mmaudio",
    "thinksound",
    "stable_audio",
    "zonos",
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
    "reel",
    # "animate_3d",
    "openai_image_edit",
    "openai_image_generate",
]

DISCORD_TOOLS = ["discord_post", "discord_search"]
TELEGRAM_TOOLS = ["telegram_post"]
TWITTER_TOOLS = ["tweet", "twitter_mentions", "twitter_search"]
FARCASTER_TOOLS = ["farcaster_cast", "farcaster_mentions", "farcaster_search"]
SHOPIFY_TOOLS = ["shopify"]
PRINTIFY_TOOLS = ["printify"]
CAPTIONS_TOOLS = ["captions"]
TIKTOK_TOOLS = ["tiktok_post"]

SOCIAL_MEDIA_TOOLS = [
    *TWITTER_TOOLS,
    *DISCORD_TOOLS,
    *FARCASTER_TOOLS,
    *TELEGRAM_TOOLS,
    *SHOPIFY_TOOLS,
    *PRINTIFY_TOOLS,
    *CAPTIONS_TOOLS,
    *TIKTOK_TOOLS,
]

CONTEXT7_MCP_TOOLS = ["context7_resolve_library_id"]
CALCULATOR_MCP_TOOLS = ["calculator_calculate"]
# Future MCP server tools can be added here
# GITHUB_MCP_TOOLS = ["github_search", "github_issues", "github_prs"]
# SLACK_MCP_TOOLS = ["slack_post", "slack_search", "slack_channels"]

TOOL_SETS = {
    "create_image": ["create", "reel", "media_editor", "magic_8_ball"],
    "create_video": [],  # deprecated
    "create_audio": ["elevenlabs", "elevenlabs_music", "elevenlabs_fx"],
    "vj_tools": ["texture_flow", "video_FX"],
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
    "create_video",
    "elevenlabs",
    "musicgen",
    "media_editor",
    "news",
]

FLUX_LORA_TXT2IMG_TOOLS = ["flux_dev_lora", "flux_dev", "flux_schnell"]

SDXL_LORA_TXT2IMG_TOOLS = ["txt2img"]

AGENTIC_TOOLS = [*SOCIAL_MEDIA_TOOLS, "reel"]
