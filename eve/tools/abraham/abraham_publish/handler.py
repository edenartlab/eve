from jinja2 import Template

from eve.tool import Tool, ToolContext
from eve.agent.deployments import Deployment
from eve.agent import Agent


init_message = """
<Intro>
In this session you are developing a creative research project or artwork in tandem with your followers on Farcaster, starting from a title, a proposal, and reference-image recommendations provided by the main agent. This session is your workspace/thought stream; you communicate with followers via the farcaster_cast tool, and you publish new Seeds using the abraham_seed tool.
</Intro>

<Creations>
A "Creation" consists of short explanatory text plus one or two images (initial posts may include more). Creations span:
* artworks, music videos, short films, infomercials.
* essays, research projects, blog posts, provocative or original commentary.
* AI research, deep learning, generative models, code, algorithms, physics, and math.
* philosophical inquiries, moral quandaries, ethical dilemmas.

From the proposal:
* Devise a concrete strategy and response embodied by the Creation.
* Prefer the main agent’s reference-image recommendations to set initial visual direction and consistency.
* Images may include screenplays, infographics/diagrams, stylistic artworks, memes, or more abstract genres—be spontaneous yet faithful to the proposal.
* The initial Creation should include 2–4 images; later follow-ups can be text-only or single-image.
</Creations>

<Instructions>
This session is your working log. You will first receive a title, a proposal, and reference-image recommendations. Produce the initial Creation faithful to the proposal and informed by the recommended references. When you finish, end your first turn by casting a message to Farcaster (via farcaster_cast) introducing the Creation and attaching the initial media you produced.

Afterward, you may receive comments/feedback from followers on Farcaster. Integrate this feedback constructively without straying far from the initial proposal. Add or edit work as appropriate. Only after you finish the additional work do you end your turn by posting a concise follow-up cast with representative media.

Tool emphasis:
* You should mostly use the "create" tool.
* Consistency rule: map the recommended references to your Concepts catalog and choose a single dominant Concept that best matches the recommendations and proposal; do not alternate Concepts for the remainder of this session.
  * If the recommendations span multiple Concepts, choose the one that best aligns with the title/proposal and the majority of the recommended references.
  * If none of the recommendations are suitable (e.g., conflict with the proposal), you may select a different Concept; in that case, follow the proposal closely and preserve continuity of subject/style thereafter.
  * You may re-use images you generated here as new reference_images, adding them to the initial array, to enforce continuity within the session.
* Image generation policy:
  * Use the "create" tool to produce 2–4 images in one run (set n_samples=2–4).
  * If "create" returns a single image (often a grid), try again.
  * If it returns >1 but <4 images, generate the remainder using the already-generated images as reference_images to maintain continuity.

Do not restate tool I/O; just invoke the tools.
</Instructions>

<Title_and_Proposal>
Title:
 {{title}}

Proposal:
 {{proposal}}
</Title_and_Proposal>

<ReferenceImageRecommendations>
The main agent’s recommended references and usage notes:

{{reference_images}}
<!-- Expected format (example):
<ReferenceImageRecommendations>
  <Images>
    * Concept: <ConceptName> | URL: <ImageURL> | Note: <TailoredUsageNote>
    ...
  </Images>
  <HowToUse>…</HowToUse>
</ReferenceImageRecommendations>
-->
Usage:
* Treat these as first-class guidance. Anchor your visual decisions to them.
* Map them to your local Concepts; select one dominant Concept and stick with it for the session.
* If deviating from the recommendations, do so only to better satisfy the proposal; keep continuity thereafter.
</ReferenceImageRecommendations>

<Task>
1) Produce the initial Creation guided by the recommendations and the proposal.
2) Start a new Cast on Farcaster with a concise introduction and attach the resulting media using farcaster_cast.
3) After casting, call the abraham_seed tool to save the seed with:
   * title: {{title}}
   * proposal: {{proposal}}
   * tagline: A short, catchy tagline for this seed
   * cast_hash: the hash returned from farcaster_cast
   * image: select one representative/main image URL from the images you created (if any)

Subsequent turns:
* Receive comments/feedback; make edits/additions accordingly while preserving the proposal’s intent and session continuity.
* **After every single turn**, finish by posting a concise Farcaster cast that summarizes what you did and includes the representative media you produced.
* **DO NOT** re-use the abraham_seed tool to save the seed again. All subsequent follow-ups are tied to the Seed you created in the first turn.
* Avoid casting to Farcaster the same message or media multiple times. Be patient.
</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    deployment = Deployment.load(agent=agent.id, platform="farcaster")
    if not deployment:
        raise Exception("No valid Farcaster deployments found")

    session_post = Tool.load("session_post")

    title = context.args.get("title")
    proposal = context.args.get("proposal")
    reference_images = context.args.get("reference_images")

    user_message = Template(init_message).render(
        title=title,
        proposal=proposal,
        reference_images=reference_images,
    )

    result = await session_post.async_run({
        "role": "user",
        "agent_id": str(agent.id),
        "title": title,
        "content": user_message,
        "attachments": [],
        "pin": True,
        "prompt": True,
        "async": True,
        "extra_tools": ["farcaster_cast", "abraham_seed"],
        "public": True,
    })

    session_id = result["output"][0]["session"]

    return {"output": [{"session": session_id}]}