from .... import llm
from ....thread import UserMessage
from ....agent import Agent
from ..common import ReelStoryboard


INSTRUCTIONS = """You are creating a storyboard or comprehensive description of a short film or “Reel” of generally 1 to 2 minutes long, in the schema given to you. This will be used to produce a final video.

Some guidelines:
- Avoid layering an overall reel voiceover on top of scene-level voiceovers unless there is a good reason to have both.
- Each video clip or scene can include:
  - A visual description of what is happening.
  - Camera motion details, if important.
  - Foley/sound effects details, if any.
  - Vocals, which can be multiple lines or multiple speakers.
- Create characters only when they are named. Do not include NPCs or extras.

Your Task:
- You will receive a user prompt describing an idea for a reel. It may be vague or highly detailed, or a mix of both.
- A user may just give you a premise or bare idea for a story, in which case you should make something short and simple, unless they ask you to be creative with the details.
- If they give you a lot of details, you should stay as authentic and faithful to the request as possible.
- The length of the overall voiceover if there is one, should not exceed the sum total duration of the video across all scenes, at approximately 20-30 words per 10 seconds of video (150-180 words per minute). If there is an overall voiceover, aim for at least half the total duration."""

DEFAULT_SYSTEM_MESSAGE = """You are an advanced AI that creates captivating storyboards and short films, authentically to a user's desire or vision."""


async def handler(args: dict, db: str):
    agent_name = args.get("agent_name")
    prompt = args.get("prompt")

    if agent_name:
        agent = Agent.load(agent_name, db=db)
        system_message = agent.description or DEFAULT_SYSTEM_MESSAGE
    else:
        system_message = DEFAULT_SYSTEM_MESSAGE
    
    reel = await llm.async_prompt(
        messages=[
            UserMessage(content=INSTRUCTIONS),
            UserMessage(content=prompt)
        ],
        system_message=system_message,
        model="claude-3-5-sonnet-20241022",
        response_model=ReelStoryboard
    )

    return {
        "output": reel.model_dump()
    }
