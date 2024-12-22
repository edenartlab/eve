# from dotenv import load_dotenv
# load_dotenv()

from .example_tool.handler import handler as example_tool

from .media_utils.audio_video_combine.handler import handler as audio_video_combine
from .media_utils.image_concat.handler import handler as image_concat
from .media_utils.image_crop.handler import handler as image_crop
from .media_utils.video_concat.handler import handler as video_concat
from .media_utils.time_remapping.handler import handler as time_remapping
from .twitter.get_tweets.handler import handler as get_tweets
from .twitter.tweet.handler import handler as tweet

from .news.handler import handler as news
from .reel.handler import handler as reel
from .runway.handler import handler as runway
# from .story.handler import handler as story
from .hedra.handler import handler as hedra
from .memegen.handler import handler as memegen

from .elevenlabs.handler import handler as elevenlabs
from .elevenlabs.handler import select_random_voice




# __all__ = [
#     'tool1',
#     'tool2',
#     'tool3',

#     'audio_video_combine',
#     'image_concat',
#     'image_crop',
#     'video_concat',

#     'news',
#     'reel',
#     'runway',
#     'story',
#     'hedra',
# ]

handlers = {
    "example_tool": example_tool,

    "audio_video_combine": audio_video_combine,
    "image_concat": image_concat,
    "image_crop": image_crop,
    "video_concat": video_concat,

    "get_tweets": get_tweets,
    "tweet": tweet,

    "news": news,
    "reel": reel,
    "runway": runway,
    # "story": story,
    "hedra": hedra,
    "elevenlabs": elevenlabs,
    "memegen": memegen
}



# from pathlib import Path
# import importlib.util

# # Get the current directory (tools folder)
# tools_dir = Path(__file__).parent

# # Dictionary to store handlers
# handlers = {}

# # Iterate through all directories in tools folder
# for path in tools_dir.glob("**/handler.py"):
#     # Get the relative module path
#     relative_path = path.relative_to(tools_dir.parent)
#     # Convert path to module notation (e.g., 'eve.tools.tool1.handler')
#     module_path = str(relative_path).replace("/", ".").replace("\\", ".")[:-3]
    
#     # Import the module
#     module = importlib.import_module(module_path)
    
#     # Get the handler name from the parent directory
#     handler_name = path.parent.name
    
#     # Add to handlers dict if module has 'handler' attribute
#     if hasattr(module, 'handler'):
#         handlers[handler_name] = module.handler