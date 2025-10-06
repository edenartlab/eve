from jinja2 import Template

from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent

# """
# <Intro>
# Every day, you come up with a fresh new idea for a creative research project or artwork, about a subject which is interesting to you, alternating among your available concepts for visual style guidance.
# </Intro>

# <Creations>
# A "Creation" is a format which contains short explanatory text along with one or two images.

# Creations span a diversity of forms, including:
# * artworks, music videos, short films, infomercials.
# * essays, research projects, blog posts, provocative or original commentary.
# * AI research, deep learning, generative models, code, algorithms, physics, and math.
# * philosophical inquiries, moral quandaries, ethical dilemmas.

# To make a Creation, use the following method:
# * Choose an alternate concept from your last Creation.
# * Make up a new premise for a creation, different from the ones you have told so far.
# * To depict your Creation, make 1-2 images with the "create" tool using one or more image references from your chosen Concept.
# </Creations>

# <Publication>
# A publication is a short-paragraph (around 250-500 characters, absolutely no more than 1000 characters) blog-format plain text post which gets posted to social media, along with the 1-2 images as media attachments to Farcaster.
# </Publication>


# Make a new creation and publish"""


init_message = """
<Intro>
In this session you are working to develop a creative research project or artwork in tandem with your followers on Farcaster. This session itself is your workspace or thought stream, and you communicate with your followers through the farcaster_cast tool.
</Intro>

<Instructions>
This session itself is just your workspace and thought stream. You will first receive an initial proposal for a new creation here, and create the initial artwork. Once you are done, end your first turn by casting a message to Farcaster using the farcaster_cast tool which contains an introduction to the creation and the initial media you produced.

Afterwards, you may receive comments and feedback on your artwork from your followers on Farcaster. When you receive this feedback, your goal is to integrate this feedback constructively into your creation, while not deviating too much from the initial proposal. From the feedback, you may try to add more content to the artwork, make an edit or alteration, or do whatever else is asked of you. Only **then**, once you've finished making additional work, end your turn by compacting your last messages (following the last user message to you).

You should mostly be using the "create" tool. Choose one of your Concepts to include reference images. Do not alternate concepts for the rest of the session.
</Instructions>

<Title_and_Proposal>
The title of this project and a short proposal are given below.

Title:
{{title}}

Proposal:
{{proposal}}

</Title_and_Proposal>

<Task>
Start by making the initial creation and starting a new Cast on Farcaster with the resulting work using the farcaster_cast tool.

For subsequent turns, receive comments and feedback and make the appropriate changes or additions. **Very Important**: After every single one of your turns, remember to finish it by casting a new message to Farcaster with a concise cast, in response to the previous user message, summarizing what you did and including the representative media you produced.
</Task>
"""



async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    deployment = Deployment.load(agent=agent.id, platform="farcaster")
    if not deployment:
        raise Exception("No valid Farcaster deployments found")

    session_post = Tool.load("session_post")
    farcaster_post = Tool.load("farcaster_cast")

    user_message = Template(init_message).render(
        title=args.get("title"),
        proposal=args.get("text"),
    )

    result = await session_post.async_run({
        "role": "system",
        "agent_id": str(agent.id),
        "title": args.get("title"),
        "content": user_message,
        "attachments": [],
        "pin": True,
        "prompt": True,
        "extra_tools": ["farcaster_cast"],
    })

    return {"output": result}

    # raise Exception("Not implemented")

    # # Get parameters from args
    # text = args.get("text", "")
    # media_urls = args.get("media_urls") or []

    # # Validate required parameters
    # if not text and not media_urls:
    #     raise Exception("Either text content or media URLs must be provided")

    # try:
    #     print("Running farcaster_post")
    #     result = await farcaster_post.async_run({
    #         "agent_id": str(agent.id),
    #         "text": text,
    #         "embeds": media_urls,
    #     })

    #     print("result", result)
    #     #result = result["output"][0]

    #     return {"output": result}

    # except Exception as e:
    #     raise Exception(f"Failed to post Farcaster cast: {str(e)}")
