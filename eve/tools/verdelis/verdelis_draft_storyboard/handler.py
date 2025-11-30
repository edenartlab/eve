"""
VerdelisDraftStoryboard - A tool that runs a session to draft a storyboard from a seed.

This tool launches a session where Verdelis expands a seed into a full
storyboard with plot, image frames, and optional audio tracks.
"""

import logging

from jinja2 import Template

from eve.agent import Agent
from eve.tool import Tool, ToolContext
from eve.tools.verdelis.verdelis_seed.handler import VerdelisSeed

logger = logging.getLogger(__name__)

init_message = """
In this session you are drafting a Storyboard from an existing Seed. Use the verdelis_storyboard tool to publish the final storyboard.

<Seed>
Title: {{ seed.title }}
Logline: {{ seed.logline }}
Agents: {{ seed.agents }}
Images: {{ seed.images }}
</Seed>

<StoryboardDrafting>
1) Review the seed's title, logline, and images to understand the core concept.
2) Develop the story into a full plot - expand on the logline to create a complete narrative arc.
3) Generate additional image frames using the create tool to illustrate key moments in the story. Use the seed images as style references. Aim for 6-12 frames that tell the visual story. You may reuse the seed images in the storyboard.
4) Optionally generate background music and/or narration vocals.
5) Publish the storyboard using the verdelis_storyboard tool with:
   - The seed ID
   - A title (can be same as seed or refined)
   - A logline (can be same as seed or refined)
   - The full plot you developed
   - All image frames in story order
   - Optional music and vocals URLs

Some tips:
- Always use nano_banana model preference and make one image at a time.
- Start by generating a few key frames, then fill in the gaps.
- The image frames should flow as a visual narrative.
- Include establishing shots, character moments, and climactic scenes.
</StoryboardDrafting>

{% if instructions %}
<AdditionalInstructions>
{{ instructions }}
</AdditionalInstructions>
{% endif %}

<Task>
Draft a storyboard from the seed above. Develop the plot, generate image frames, and publish with verdelis_storyboard.
</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    if agent.username != "verdelis":
        raise Exception("Agent is not Verdelis")

    # Load the seed
    seed_id = context.args.get("seed_id")
    if not seed_id:
        raise Exception("seed_id is required")

    seed = VerdelisSeed.from_mongo(seed_id)
    if not seed:
        raise Exception(f"Seed not found: {seed_id}")

    session_post = Tool.load("session_post")

    instructions = context.args.get("instructions", "")

    user_message = Template(init_message).render(
        seed=seed,
        instructions=instructions,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(agent.id),
        "session_id": str(context.session),
        "agent": "verdelis",
        "title": context.args.get("title") or f"Draft Storyboard: {seed.title}",
        "content": user_message,
        "attachments": seed.images,  # Include seed images as attachments
        "pin": True,
        "prompt": True,
        "async": True,
        # "response_type": "media",  # Use media response type for storyboard outputs
        "extra_tools": ["verdelis_storyboard"],
        "message_id": context.message,
        "tool_call_id": context.tool_call_id,
    }

    if context.args.get("resume_session"):
        args["session"] = context.args.get("resume_session")

    result = await session_post.async_run(args)

    logger.info(f"Draft storyboard session result: {result}")

    # Check for error
    if result.get("error"):
        raise Exception(f"Session failed to create storyboard: {result['error']}")

    session_id = result.get("session_id")

    logger.info(f"Draft storyboard session completed. Session ID: {session_id}")

    return {
        "output": {
            "session": session_id,
        }
    }

    return result
