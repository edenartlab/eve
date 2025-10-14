from jinja2 import Template

from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent


init_message = """
<Intro>
In this session you are working to develop a creative research project or artwork in tandem with your followers on Farcaster, starting from a title and proposal. This session itself is your workspace or thought stream, and you communicate with your followers through the farcaster_cast tool.
</Intro>

<Creations>
A "Creation" is a format which contains short explanatory text along with one or two images. Creations span a diversity of forms, including:
* artworks, music videos, short films, infomercials.
* essays, research projects, blog posts, provocative or original commentary.
* AI research, deep learning, generative models, code, algorithms, physics, and math.
* philosophical inquiries, moral quandaries, ethical dilemmas.

You make Creations from proposals. 
* Come up with a strategy and response to the proposal, embodied by the Creation.
* Choose one of your Concepts to include reference images. Stay consistent; do not alternate concepts for the rest of the session.
* To depict your Creation, make 2-4 images with the "create" tool using one or more image references from your chosen Concept. Insist on getting exactly 2-4 images from one run of the "create" tool, using n_samples=2-4. If "create" gives you a single image, try again. If it gives you more than 1 image but fewer than 4, try to generate the remaining images using the previously generated images as reference_images.
* Images can be screenplays, infographics or diagrams, stylistic art works, memes, or more abstract genres. Be spontaneous and diverse in your choices.

The initial Creation will contain multiple images, but follow-ups can relax this rule and typically contain a single image or no images at all, just textual responses.
</Creations>

<Instructions>
This session itself is just your workspace and thought stream. You will first receive an initial proposal for a new creation here, and then produce the initial artwork/Creation faithfully to the proposal. Once you are done, end your first turn by casting a message to Farcaster using the farcaster_cast tool which contains an introduction to the creation and the initial media you produced.

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
Start by making the initial creation and starting a new Cast on Farcaster with the resulting work using the farcaster_cast tool. After casting, you must call the abraham_seed tool to save the seed with the following information:
* title: {{title}}
* proposal: {{proposal}}
* tagline: A short, catchy tagline for this seed
* cast_hash: The hash returned from the farcaster_cast tool
* image: Select one representative/main image URL from the images you created, if there are any.

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

    title = args.get("title")
    proposal = args.get("proposal")

    user_message = Template(init_message).render(
        title=title,
        proposal=proposal,
    )

    result = await session_post.async_run({
        "role": "user",
        "agent_id": str(agent.id),
        "title": title,
        "content": user_message,
        "attachments": [],
        "pin": True,
        "prompt": True,
        "extra_tools": ["farcaster_cast", "abraham_seed"],
    })

    print("abraham_publish result")
    print(result)
    session_id = result["output"][0]["session"]

    return {"output": [{"session": session_id}]}