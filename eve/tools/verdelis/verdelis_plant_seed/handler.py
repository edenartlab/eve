"""
VerdelisPlantSeed - A tool that runs a session to generate new seed ideas.

This tool launches a session where Verdelis comes up with new story ideas
and publishes them as Seeds using the verdelis_seed tool.
"""

import logging

from jinja2 import Template

from eve.agent import Agent
from eve.tool import Tool, ToolContext

logger = logging.getLogger(__name__)

init_message = """
In this session you are coming up with Seeds for new story ideas. Use the verdelis_seed tool exclusively to create new stories. Each new story should have a title, logline, and 2-3 representative images. To generate a new seed follow this process:

<SeedCreation>
1) Come up with a new story idea and express it as a short catchy logline. Title it.
2) Generate 2-3 representative images using the create tool. Use your reference images. At least 1 image should be of the primary setting, and at least 1 image should be of the primary character(s) doing the main thing.
3) Publish a new seed using the verdelis_seed tool.

Some tips:
- Always use nano_banana model preference and make one image at a time.
- One good technique is to generate the setting first, and then generate an image of a character in that setting by including the previous location image and a concept image of the character as reference images.
- Good image compositions include a mixture of setting, character, and primary action.
</SeedCreation>

<Task>
{{ instructions }}
</Task>
"""

import os

if os.getenv("MOCK") == "1":
    init_message += """
    NOTE: If you just get those generic black pictures of the Eden logo (a circuitboard tree), then those are mock images. Just play along and let them go along, pretend they are real.
    """


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    if agent.username != "verdelis":
        raise Exception("Agent is not Verdelis")

    session_post = Tool.load("session_post")

    instructions = context.args.get(
        "instructions", "Create a new seed for a story idea."
    )

    user_message = Template(init_message).render(
        instructions=instructions,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(agent.id),
        "session_id": str(context.session),
        "agent": "verdelis",
        "title": context.args.get("title") or "Plant Seed Session",
        "content": user_message,
        "attachments": [],
        "pin": True,
        "prompt": True,
        "async": False,
        "response_type": "seed",  # Use seed response type to extract seed IDs
        "extra_tools": ["verdelis_seed"],
        "message_id": context.message,
        "tool_call_id": context.tool_call_id,
    }

    if context.args.get("resume_session"):
        args["session"] = context.args.get("resume_session")

    result = await session_post.async_run(args)

    # Check for error
    if result.get("error"):
        raise Exception(f"Session failed to create seed: {result['error']}")

    # Get artifact IDs from result
    artifact_ids = result.get("artifact_ids", [])
    session_id = result.get("session_id")

    if not artifact_ids:
        raise Exception(
            "No seeds were created in this session. "
            "The session may have failed to call verdelis_seed."
        )

    logger.info(f"Plant seed session completed. Artifact IDs: {artifact_ids}")

    return {
        "output": {
            "artifact_id": artifact_ids[0],
            "session_id": session_id,
        }
    }
