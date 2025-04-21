handlers = {}


def load_handler(name):
    if name not in handlers:
        if name == "example_tool":
            from .example_tool.handler import handler
            handlers[name] = handler

        elif name == "memory":
            from .memory.handler import handler
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

        elif name == "get_tweets":
            from .twitter.get_tweets.handler import handler
            handlers[name] = handler

        elif name == "tweet":
            from .twitter.tweet.handler import handler
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

        elif name == "hedra":
            from .hedra.handler import handler
            handlers[name] = handler

        elif name == "memegen":
            from .memegen.handler import handler
            handlers[name] = handler

        elif name == "elevenlabs":
            from .elevenlabs.handler import handler
            handlers[name] = handler

        elif name == "websearch":
            from .websearch.handler import handler
            handlers[name] = handler

        elif name == "send_eth":
            from .wallet.send_eth.handler import handler
            handlers[name] = handler

        elif name == "weather":
            from .weather.handler import handler
            handlers[name] = handler

        elif name == "fal":
            from .fal_tool import FalTool
            handlers[name] = FalTool

    return handlers[name]
