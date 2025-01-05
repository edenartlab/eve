from .example_tool.handler import handler as example_tool

from .media_utils.audio_video_combine.handler import handler as audio_video_combine
from .media_utils.image_concat.handler import handler as image_concat
from .media_utils.image_crop.handler import handler as image_crop
from .media_utils.video_concat.handler import handler as video_concat

from .wallet.send_eth.handler import handler as send_eth
from .websearch.handler import handler as websearch

from .media_utils.time_remapping.handler import handler as time_remapping
from .twitter.get_tweets.handler import handler as get_tweets
from .twitter.tweet.handler import handler as tweet

from .news.handler import handler as news
from .reel.handler import handler as reel
from .runway.handler import handler as runway
from .hedra.handler import handler as hedra
from .memegen.handler import handler as memegen

from .elevenlabs.handler import handler as elevenlabs


handlers = {
    "example_tool": example_tool,

    "audio_video_combine": audio_video_combine,
    "image_concat": image_concat,
    "image_crop": image_crop,
    "video_concat": video_concat,
    "time_remapping": time_remapping,
    "get_tweets": get_tweets,
    "tweet": tweet,

    "news": news,
    "reel": reel,
    "runway": runway,
    
    "hedra": hedra,
    "elevenlabs": elevenlabs,
    "memegen": memegen,
    "websearch": websearch,
    "send_eth": send_eth,
}
