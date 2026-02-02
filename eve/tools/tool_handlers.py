import importlib

from eve.tool_constants import (
    ABRAHAM_TOOLS,
    CHIBA_TOOLS,
    GIGABRAIN_TOOLS,
    HOME_ASSISTANT_TOOLS,
    RETRIEVAL_TOOLS,
    VERDELIS_TOOLS,
    WZRD_TOOLS,
)

handlers = {}


# Map tool names to their import paths
HANDLER_PATHS = {
    "example_tool": "example_tool.handler",
    "magic_8_ball": "magic_8_ball.handler",
    "audio_video_combine": "media_utils.audio_video_combine.handler",
    "audio_mix": "media_utils.audio_mix.handler",
    "audio_concat": "media_utils.audio_concat.handler",
    "audio_pad": "media_utils.audio_pad.handler",
    "image_concat": "media_utils.image_concat.handler",
    "image_crop": "media_utils.image_crop.handler",
    "video_concat": "media_utils.video_concat.handler",
    "ffmpeg_multitool": "media_utils.ffmpeg_multitool.handler",
    "media_editor": "media_utils.media_editor.handler",
    "create": "media_utils.create.handler",
    "create_image": "media_utils.create_image.handler",
    "create_video": "media_utils.create_video.handler",
    "search_agents": "mongo_utils.search_agents.handler",
    "search_models": "mongo_utils.search_models.handler",
    "search_collections": "mongo_utils.search_collections.handler",
    "add_to_collection": "mongo_utils.add_to_collection.handler",
    "create_collection": "mongo_utils.create_collection.handler",
    "get_messages": "eden_utils.get_messages.handler",
    "eden_search": "eden_utils.eden_search.handler",
    "get_messages_digest": "gigabrain.get_messages_digest.handler",
    "twitter_mentions": "twitter.twitter_mentions.handler",
    "twitter_search": "twitter.twitter_search.handler",
    "tweet": "twitter.tweet.handler",
    "discord_post": "discord.discord_post.handler",
    "discord_search": "discord.discord_search.handler",
    "discord_broadcast_dm": "discord.discord_broadcast_dm.handler",
    "telegram_post": "telegram.telegram_post.handler",
    "farcaster_cast": "farcaster.farcaster_cast.handler",
    "farcaster_mentions": "farcaster.farcaster_mentions.handler",
    "farcaster_search": "farcaster.farcaster_search.handler",
    "instagram_post": "instagram.instagram_post.handler",
    "shopify": "shopify.handler",
    "news": "news.handler",
    "reel": "media_utils.reel.handler",
    "reel2": "media_utils.reel2.handler",
    "reel_audio": "media_utils.reel_audio.handler",
    "reel_storyboard": "media_utils.reel_storyboard.handler",
    "reel_video": "media_utils.reel_video.handler",
    "runway": "runway.handler",
    "runway2": "runway2.handler",
    "runway3": "runway3.handler",
    "hedra": "hedra.handler",
    "elevenlabs": "elevenlabs.handler",
    "elevenlabs_dialogue": "elevenlabs_dialogue.handler",
    "elevenlabs_music": "elevenlabs_music.handler",
    "elevenlabs_fx": "elevenlabs_fx.handler",
    "elevenlabs_search_voices": "elevenlabs_search_voices.handler",
    "elevenlabs_speech": "elevenlabs_speech.handler",
    "transcription": "transcription.handler",
    "send_eth": "wallet.send_eth.handler",
    "vibevoice": "voice_cloning.vibevoice.handler",
    "weather": "weather.handler",
    "openai_image_generate": "openai_image_generate.handler",
    "openai_image_edit": "openai_image_edit.handler",
    "veo2": "google.veo2.handler",
    "veo3": "google.veo3.handler",
    "nano_banana": "google.nano_banana.handler",
    "nano_banana_pro": "google.nano_banana_pro.handler",
    "nano_banana_fal": "fal.nano_banana_fal.handler",
    "nano_banana_pro_fal": "fal.nano_banana_pro_fal.handler",
    "captions": "captions.handler",
    "printify": "printify.handler",
    "tiktok_post": "tiktok.tiktok_post.handler",
    "session_post": "session_post.handler",
    "chat": "chat.handler",
    # Moderator tools for multi-agent session orchestration
    "start_session": "moderator.start_session.handler",
    "finish_session": "moderator.finish_session.handler",
    "prompt_agent": "moderator.prompt_agent.handler",
    "conduct_vote": "moderator.conduct_vote.handler",
    "email_send": "email.email_send.handler",
    "gmail_send": "gmail.gmail_send.handler",
    # Google Calendar tools
    "google_calendar_query": "google_calendar.google_calendar_query.handler",
    "google_calendar_edit": "google_calendar.google_calendar_edit.handler",
    "google_calendar_delete": "google_calendar.google_calendar_delete.handler",
    # Agent tools are added dynamically below
    **{tool: f"abraham.{tool}.handler" for tool in ABRAHAM_TOOLS},
    **{tool: f"verdelis.{tool}.handler" for tool in VERDELIS_TOOLS},
    **{tool: f"gigabrain.{tool}.handler" for tool in GIGABRAIN_TOOLS},
    **{tool: f"chiba.{tool}.handler" for tool in CHIBA_TOOLS},
    **{tool: f"wzrd.{tool}.handler" for tool in WZRD_TOOLS},
    **{tool: f"home_assistant.{tool}.handler" for tool in HOME_ASSISTANT_TOOLS},
    **{tool: f"retrieval.{tool}.handler" for tool in RETRIEVAL_TOOLS},
}


def load_handler(name):
    if name not in handlers:
        if name not in HANDLER_PATHS:
            raise ValueError(f"Unknown handler: {name}")

        module_path = f".{HANDLER_PATHS[name]}"
        module = importlib.import_module(module_path, package="eve.tools")
        handlers[name] = module.handler

    return handlers[name]
