import importlib

handlers = {}

# Map tool names to their import paths
HANDLER_PATHS = {
    "example_tool": "example_tool.handler",
    "magic_8_ball": "magic_8_ball.handler",
    "audio_video_combine": "media_utils.audio_video_combine.handler",
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
    "shopify": "shopify.handler",
    "news": "news.handler",
    "reel": "reel.handler",
    "runway": "runway.handler",
    "runway2": "runway2.handler",
    "runway3": "runway3.handler",
    "hedra": "hedra.handler",
    "elevenlabs": "elevenlabs.handler",
    "elevenlabs_music": "elevenlabs_music.handler",
    "elevenlabs_fx": "elevenlabs_fx.handler",
    "transcription": "transcription.handler",
    "send_eth": "wallet.send_eth.handler",
    "vibevoice": "voice_cloning.vibevoice.handler",
    "weather": "weather.handler",
    "openai_image_generate": "openai_image_generate.handler",
    "openai_image_edit": "openai_image_edit.handler",
    "veo2": "google.veo2.handler",
    "veo3": "google.veo3.handler",
    "captions": "captions.handler",
    "printify": "printify.handler",
    "tiktok_post": "tiktok.tiktok_post.handler",
    "session_post": "session_post.handler",
    "email_send": "email.email_send.handler",
    "abraham_publish": "abraham.abraham_publish.handler",
    "abraham_daily": "abraham.abraham_daily.handler",
    "abraham_covenant": "abraham.abraham_covenant.handler",
    "abraham_rest": "abraham.abraham_rest.handler",
    "abraham_seed": "abraham.abraham_seed.handler",
    "abraham_learn": "abraham.abraham_learn.handler",
    "verdelis_story": "verdelis.verdelis_story.handler"
}


def load_handler(name):
    if name not in handlers:
        if name not in HANDLER_PATHS:
            raise ValueError(f"Unknown handler: {name}")

        module_path = f".{HANDLER_PATHS[name]}"
        module = importlib.import_module(module_path, package="eve.tools")
        handlers[name] = module.handler

    return handlers[name]
