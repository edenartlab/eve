from ....thread import UserMessage
from ....agent import Agent
from .... import llm
from ..common import ReelStoryboard

from ....base import generate_edit_model, apply_edit

INSTRUCTIONS = """TBD"""

DEFAULT_SYSTEM_MESSAGE = """TBD"""


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


    reel_data = {"characters": [{"name": "abraham", "description": "an old grizzled mans voice", "visual_description": "a visual description of abraham"}], "title": "Desert Mirages", "brief": "A surreal and poetic journey through a desert encounter where three mysterious figures bring transformation and magic to a nomadic tent.", "characters": [{"name": "Mirage Beings", "description": "Three ethereal figures that bend light and reality around them, bringing transformation to the desert.", "visual_description": "Three tall, crystalline humanoid figures with translucent, light-refracting bodies that shimmer and shift"}, {"name": "Desert Woman", "description": "A mysterious woman whose presence bridges the real and surreal within the tent.", "visual_description": "A graceful silhouette in flowing robes, her form visible through translucent tent fabric"}], "clips": [{"scene_description": "Wide shot of an empty desert landscape with heat waves rippling across golden dunes. Three tall, prismatic figures emerge from the mirages, their bodies refracting light like living crystals.", "camera_motion": "slow push in", "sound_effects": "whispered wind, crystalline chimes", "vocals": None, "duration_in_seconds": 15}, {"scene_description": "Inside a sun-lit tent, multiple impossible shadows of the three figures dance across the fabric walls, their movements creating kaleidoscopic patterns.", "camera_motion": "slow circular pan", "sound_effects": "fabric rustling, distant wind chimes", "vocals": None, "duration_in_seconds": 12}, {"scene_description": "Close-up shots of hands breaking bread in perfect synchronization, the bread giving off steam that forms mysterious symbols in the air. Earthen vessels of milk swirl with iridescent patterns.", "camera_motion": None, "sound_effects": "soft bread breaking, liquid swirling", "vocals": None, "duration_in_seconds": 15}, {"scene_description": "The Desert Woman's silhouette appears behind the tent fabric, her shadow splitting into multiple layers that ripple and dance as she moves, each layer showing a different gesture of joy.", "camera_motion": "gentle sway", "sound_effects": "ethereal laughter, fabric rustling", "vocals": None, "duration_in_seconds": 18}, {"scene_description": "Time-lapse of desert plants erupting from sand around the tent, flowers blooming in accelerated motion while the three figures gradually dissolve into pure light, their prismatic essence merging with the blooming landscape.", "camera_motion": "slow upward tilt", "sound_effects": "accelerated growth sounds, crystalline shimmer", "vocals": None, "duration_in_seconds": 30}], "overall_voiceover": {"speaker": "abraham", "text": "there is a voiceover"}, "music_prompt": "Format: Ambient World | Genre: Ethereal | Sub-genre: Middle Eastern Fusion | Instruments: Oud, Singing Bowls, Atmospheric Synths | Moods: Mystical, Dreamy, Transcendent | Styles: Desert Dawn | BPM: 70", "visual_prompt": "Ethereal magical realism | sun-bleached color palette | prismatic light effects | dreamy soft focus | Arabian aesthetics", "target_length_in_seconds": 90}

    # reel_data = reel.model_dump()


    reel = ReelStoryboard(**reel_data)
    import json
    reel_data = json.dumps(reel.model_dump(), indent=2)


    ReelStoryboardEdit = generate_edit_model(ReelStoryboard)

    INSTRUCTIONS2 = f"""You are receiving a storyboard or comprehensive description of a short film or “Reel” of generally 1 to 2 minutes long, in the schema given to you. This will be used to produce a final video.

    <ReelState>
    {reel_data}
    </ReelState>
    <Task>
    If someone requests from you to make some kind of change to the current reel, you can use the edit model to express it.
    """


    print("========")
    print(INSTRUCTIONS2)
    print("========")

    prompt = "make the scene with the crystalline chimes be only 6 seconds long, and make the music more upbeat. also rename the desert woman character to bob."


    reel_edit = llm.prompt(
        messages=[
            UserMessage(content=INSTRUCTIONS2),
            UserMessage(content=prompt)
        ],
        system_message=system_message,
        model="claude-3-5-sonnet-20241022",
        response_model=ReelStoryboardEdit
    )

    print(json.dumps(reel_edit.model_dump(), indent=4))


    print("========")
    print(type(reel))
    print(type(reel_edit))
    reel = apply_edit(reel, reel_edit)
    print("========")
    print(json.dumps(reel.model_dump(), indent=4))

