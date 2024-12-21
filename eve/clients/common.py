import os
import time

from eve.models import ClientType


HOUR_IMAGE_LIMIT = 50
HOUR_VIDEO_LIMIT = 10
DAY_IMAGE_LIMIT = 200
DAY_VIDEO_LIMIT = 40

LONG_RUNNING_TOOLS = [
    "txt2vid",
    "style_mixing",
    "img2vid",
    "vid2vid",
    "video_upscale",
    "vid2vid_sdxl",
    "lora_trainer",
    "animate_3D",
    "reel",
    "story",
]

VIDEO_TOOLS = [
    "animate_3D",
    "txt2vid",
    "img2vid",
    "vid2vid_sdxl",
    "style_mixing",
    "video_upscaler",
    "reel",
    "story",
    "lora_trainer",
    "runway",
]


DISCORD_DM_WHITELIST = [
    494760194203451393,
    623923865864765452,
    404322488215142410,
    363287706798653441,
    142466375024115712,
    598627733576089681,
    551619012140990465,
    668831945941188648,
]


hour_timestamps = {}
day_timestamps = {}


def user_over_rate_limits(user):
    user_id = str(user.id)

    if user_id not in hour_timestamps:
        hour_timestamps[user_id] = []
    if user_id not in day_timestamps:
        day_timestamps[user_id] = []

    hour_timestamps[user_id] = [
        t for t in hour_timestamps[user_id] if time.time() - t["time"] < 3600
    ]
    day_timestamps[user_id] = [
        t for t in day_timestamps[user_id] if time.time() - t["time"] < 86400
    ]

    hour_video_tool_calls = len(
        [t for t in hour_timestamps[user_id] if t["tool"] in VIDEO_TOOLS]
    )
    hour_image_tool_calls = len(
        [t for t in hour_timestamps[user_id] if t["tool"] not in VIDEO_TOOLS]
    )

    day_video_tool_calls = len(
        [t for t in day_timestamps[user_id] if t["tool"] in VIDEO_TOOLS]
    )
    day_image_tool_calls = len(
        [t for t in day_timestamps[user_id] if t["tool"] not in VIDEO_TOOLS]
    )

    if (
        hour_video_tool_calls >= HOUR_VIDEO_LIMIT
        or hour_image_tool_calls >= HOUR_IMAGE_LIMIT
    ):
        return True
    if (
        day_video_tool_calls >= DAY_VIDEO_LIMIT
        or day_image_tool_calls >= DAY_IMAGE_LIMIT
    ):
        return True
    return False


def register_tool_call(user, tool_name):
    user_id = str(user.id)
    hour_timestamps[user_id].append({"time": time.time(), "tool": tool_name})
    day_timestamps[user_id].append({"time": time.time(), "tool": tool_name})


def get_ably_channel_name(agent_username: str, client_platform: ClientType):
    env = os.getenv("UPDATE_CHANNEL_ENV", "DEV")
    return f"{agent_username.lower()}_{client_platform.value}_{env}"
