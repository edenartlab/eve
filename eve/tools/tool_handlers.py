handlers = {}


def load_handler(name):
    if name not in handlers.keys():
        if name == "example_tool":
            from .example_tool.handler import handler

            handlers[name] = handler

        elif name == "memory":
            from .memory.handler import handler

            handlers[name] = handler

        elif name == "magic_8_ball":
            from .magic_8_ball.handler import handler

            handlers[name] = handler

        elif name == "audio_video_combine":
            from .media_utils.audio_video_combine.handler import handler

            handlers[name] = handler

        elif name == "image_concat":
            from .media_utils.image_concat.handler import handler

            handlers[name] = handler

        elif name == "image_crop":
            from .media_utils.image_crop.handler import handler

            handlers[name] = handler

        elif name == "video_concat":
            from .media_utils.video_concat.handler import handler

            handlers[name] = handler

        elif name == "ffmpeg_multitool":
            from .media_utils.ffmpeg_multitool.handler import handler

            handlers[name] = handler

        elif name == "media_editor":
            from .media_utils.media_editor.handler import handler

            handlers[name] = handler

        elif name == "create":
            from .media_utils.create.handler import handler

            handlers[name] = handler

        elif name == "create_image":
            from .media_utils.create_image.handler import handler

            handlers[name] = handler

        elif name == "create_video":
            from .media_utils.create_video.handler import handler

            handlers[name] = handler

        elif name == "time_remapping":
            from .media_utils.time_remapping.handler import handler

            handlers[name] = handler

        elif name == "search_agents":
            from .mongo_utils.search_agents.handler import handler

            handlers[name] = handler

        elif name == "create_session":
            from .mongo_utils.create_session.handler import handler

            handlers[name] = handler

        elif name == "search_models":
            from .mongo_utils.search_models.handler import handler

            handlers[name] = handler

        elif name == "search_collections":
            from .mongo_utils.search_collections.handler import handler

            handlers[name] = handler

        elif name == "add_to_collection":
            from .mongo_utils.add_to_collection.handler import handler

            handlers[name] = handler

        elif name == "twitter_mentions":
            from .twitter.twitter_mentions.handler import handler

            handlers[name] = handler

        elif name == "twitter_search":
            from .twitter.twitter_search.handler import handler

            handlers[name] = handler

        elif name == "tweet":
            from .twitter.tweet.handler import handler

            handlers[name] = handler

        elif name == "twitter_trends":
            from .twitter.twitter_trends.handler import handler

            handlers[name] = handler

        elif name == "discord_post":
            from .discord.discord_post.handler import handler

            handlers[name] = handler

        elif name == "discord_search":
            from .discord.discord_search.handler import handler

            handlers[name] = handler

        elif name == "discord_broadcast_dm":
            from .discord.discord_broadcast_dm.handler import handler

            handlers[name] = handler

        elif name == "telegram_post":
            from .telegram.telegram_post.handler import handler

            handlers[name] = handler

        elif name == "farcaster_cast":
            from .farcaster.farcaster_cast.handler import handler

            handlers[name] = handler

        elif name == "farcaster_mentions":
            from .farcaster.farcaster_mentions.handler import handler

            handlers[name] = handler

        elif name == "farcaster_search":
            from .farcaster.farcaster_search.handler import handler

            handlers[name] = handler

        elif name == "shopify":
            from .shopify.handler import handler

            handlers[name] = handler

        elif name == "news":
            from .news.handler import handler

            handlers[name] = handler

        elif name == "reel":
            from .reel.handler import handler

            handlers[name] = handler

        elif name == "runway":
            from .runway.handler import handler

            handlers[name] = handler

        elif name == "runway2":
            from .runway2.handler import handler

            handlers[name] = handler

        elif name == "runway3":
            from .runway3.handler import handler

            handlers[name] = handler

        elif name == "hedra":
            from .hedra.handler import handler

            handlers[name] = handler

        elif name == "elevenlabs":
            from .elevenlabs.handler import handler

            handlers[name] = handler

        elif name == "elevenlabs_music":
            from .elevenlabs_music.handler import handler

            handlers[name] = handler

        elif name == "elevenlabs_fx":
            from .elevenlabs_fx.handler import handler

            handlers[name] = handler

        elif name == "transcription":
            from .transcription.handler import handler

            handlers[name] = handler

        elif name == "send_eth":
            from .wallet.send_eth.handler import handler

            handlers[name] = handler

        elif name == "weather":
            from .weather.handler import handler

            handlers[name] = handler

        elif name == "openai_image_generate":
            from .openai_image_generate.handler import handler

            handlers[name] = handler

        elif name == "openai_image_edit":
            from .openai_image_edit.handler import handler

            handlers[name] = handler

        elif name == "veo2":
            from .google.veo2.handler import handler

            handlers[name] = handler

        elif name == "veo3":
            from .google.veo3.handler import handler

            handlers[name] = handler

        elif name == "captions":
            from .captions.handler import handler

            handlers[name] = handler

        elif name == "printify":
            from .printify.handler import handler

            handlers[name] = handler

        elif name == "tiktok_post":
            from .tiktok.tiktok_post.handler import handler

            handlers[name] = handler

        elif name == "session_post":
            from .session_post.handler import handler

            handlers[name] = handler

        elif name == "abraham_publish":
            from .abraham.abraham_publish.handler import handler

            handlers[name] = handler
        
        elif name == "abraham_daily":
            from .abraham.abraham_daily.handler import handler

            handlers[name] = handler
        
        elif name == "abraham_covenant":
            from .abraham.abraham_covenant.handler import handler

            handlers[name] = handler

        elif name == "abraham_rest":
            from .abraham.abraham_rest.handler import handler

            handlers[name] = handler

        elif name == "abraham_seed":
            from .abraham.abraham_seed.handler import handler

            handlers[name] = handler

        elif name == "verdelis_story":
            from .verdelis.verdelis_story.handler import handler

            handlers[name] = handler

    return handlers[name]
