"""
Constants used by both tool.py and agent.py, extracted to break circular imports.
"""

from typing import Literal

OUTPUT_TYPES = Literal[
    "boolean", "string", "integer", "float", "array", "image", "video", "audio", "lora"
]

HANDLERS = Literal[
    "local", "modal", "comfyui", "replicate", "gcp", "fal"
]

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
    # video
    "runway",
    "kling_pro",
    "veo2",
    "hedra",
    "vid2vid_sdxl",
    "video_FX",
    "texture_flow",
    # audio
    "ace_step_musicgen",
    "elevenlabs",
    "mmaudio",
    "stable_audio",
    "zonos",
    "transcription",
    # editing
    "media_editor",    
    # search
    # "search_agents",
    # "search_models",
    # "search_collections",
    # "add_to_collection",
    # misc
    "news",
    "weather",
    # inactive
    # "ominicontrol",
    # "flux_redux",
    "reel",
    # "animate_3d",
    "openai_image_edit",
    "openai_image_generate",
]


TOOL_SETS = {
    "create_image": ["create"],
    "create_video": ["create_video", "media_editor", "reel"],
    "create_audio": ["elevenlabs", "musicgen"],
    "vj_tools": ["texture_flow", "video_FX"],
    "news": ["news"],
    "social_media_tools": ["tweet", "twitter_mentions", "twitter_search", "discord_search", "discord_post", "telegram_post"],
    "legacy_tools": ["legacy_create", "legacy_interpolate", "legacy_controlnet", "legacy_real2real", "legacy_txt2vid"],
    "all_tools": ALL_TOOLS
}


BASE_TOOLS = [
    "create",
    "create_video",
    "elevenlabs",
    "musicgen",
    "media_editor",
    "news"
]

BASE_TOOLS2 = [
    "create_image",
    "create_video",
    "create_audio",
    "media_editor",

    "texture_flow",
    "video_FX",
]



FLUX_LORA_TXT2IMG_TOOLS = ["flux_dev_lora", "flux_dev", "flux_schnell"]

SDXL_LORA_TXT2IMG_TOOLS = ["txt2img"]

OWNER_ONLY_TOOLS = ["tweet", "twitter_mentions", "twitter_search", "discord_search", "discord_post"]

AGENTIC_TOOLS = OWNER_ONLY_TOOLS + ["reel"]