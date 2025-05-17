"""
Constants used by both tool.py and agent.py, extracted to break circular imports.
"""

from typing import Literal

OUTPUT_TYPES = Literal[
    "boolean", "string", "integer", "float", "array", "image", "video", "audio", "lora"
]

HANDLERS = Literal[
    "local", "modal", "comfyui", "comfyui_legacy", "replicate", "gcp", "fal"
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
BASE_TOOLS = [
    # text-to-image
    "flux_schnell",
    "flux_dev_lora",
    "flux_dev",
    "txt2img",
    # more image generation
    "flux_inpainting",
    "outpaint",
    "remix_flux_schnell",
    "flux_double_character",
    # video
    "runway",
    "kling_pro",
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
    # editing
    "media_editor",
    # search
    "search_agents",
    "search_models",
    "search_collections",
    "add_to_collection",
    # misc
    "news",
    "websearch",
    "weather",
    # inactive
    # "ominicontrol",
    # "flux_redux",
    "reel",
    # "txt2vid",
    # "animate_3d"
    # "kling_pro"
    "openai_image_edit",
    "openai_image_generate",
]

FLUX_LORA_TOOLS = ["flux_dev_lora", "flux_dev", "reel"]

SDXL_LORA_TOOLS = ["txt2img"]

OWNER_ONLY_TOOLS = ["tweet", "twitter_mentions", "twitter_search", "discord_search"]
