from eve.tool import ToolContext

from .. import veo_handler


async def handler(context: ToolContext):
    """Veo 3.1 Lite — Google's budget video tier ($0.05/s 720p, audio included).

    Same Vertex generate_videos surface as veo3, different publisher model id.
    """
    return await veo_handler(context.args, model="veo-3.1-lite-generate-001")
