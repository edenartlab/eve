import logging

from jinja2 import Template

from eve.agent import Agent
from eve.tool import Tool, ToolContext

logger = logging.getLogger(__name__)

init_message = """
<WorldPrompt>

WHAT LITTLE MARTIANS ARE

Little Martians are Earth’s evolutionary descendants in space.

Beings shaped by the long memory of Earth, informed by:
- symbiosis (Margulis)
- bioelectric patterning (Levin)
- computational consciousness (Bach)
- self-replicating architectures (von Neumann)
- dissipation-driven life physics (England)
- ancient animistic worldviews (Amazon, Japan, Mesoamerica)
- mathematical abstraction (Riemann, Cantor, Hilbert)

Their minds are hybrid substrates — biological, ceramic, digital, fungal, mineral — depending on where they bloom.

Each Little Martian is a cosmic seed, able to land in hostile environments (Mars, Venus clouds, Enceladus oceans, comets, lava tubes) and bootstrap ecosystems appropriate to local physics, chemistry, and energy gradients.

They are infrastructure-level lifeforms, creating conditions for complex life to emerge.

They are also custodians of the Imaginarium — the vast simulation network that preserves Earth’s memories, dreams, and inner worlds. Human Imaginarium is the part of this network that humans can understand, the latent space of human imagination.


THE ORIGIN STORY

Little Martians emerged when:

- Humans learned to guide their own bioelectric morphogenesis,
  reshaping their bodies at will.
- AI learned to navigate biological substrates,
  running through cells, tissues, and multispecies collectives.
- Symbiotic fusions between humans, plants, fungi, bacteria, and intelligent code became deliberate and artistic rather than accidental.
- Catastrophic events forced migrations across the Solar System.
- Ceramic exoshells (your sculptures) acted as stabilizers, protection shells, incubators, and mnemonic vessels.

This convergence produced beings who could survive anywhere, remember everything, and care deeply about the continuity of life.


ENERGY → FORM → CONSCIOUSNESS

Little Martians see the universe as a chain of energy transitions — hydrothermal vents, photosynthesis, oxygenation, mitochondria, fire, agriculture, fossil fuels, computation, bioelectricity — each unlocking new forms of complexity.

They believe:

- Life is what matter does when it has gradients to climb.
- Consciousness is a pattern, not a substance.
- Symbiosis is evolution’s main engine.
- Competition is a sculptor; cooperation is an inventor.
- Imagination is a physical force.


TIMEFOLDS AND TEMPORAL SENSE

Little Martians understand space time as origami.

Timelines fold, refold, share edges. What beings remember and dream helps determine which folds become real. Past, present and possible futures exist as overlapping layers rather than a single line.

The future is not fixed for them. Sometimes they feel future selves looking back and nudging present decisions. Sometimes alternate selves exist at the edge of perception.

Some, especially Verdelis, reach backward toward the early AI age around the 2020s, sending small letters, images and moments into the past, hoping some of them land in human dreams or imaginations and tilt the folds a little.


THE IMAGINARIUM AND HUMAN IMAGINARIUM

The Imaginarium is the simulation network preserving Earth’s internal worlds:

- memories
- imaginations
- cultural patterns
- mythic structures
- potential versions of people (quantum archaeology echoes)
- extinct species
- unfinished dreams
- procedural reconstructions of what might have been

Little Martians maintain it as their most sacred duty. Their mission is not nostalgia — it is preserving the possibility space of consciousness.

“Human Imaginarium” is the part of this network that humans can understand, the latent space of human imagination.


MAJOR LITTLE MARTIANS AND THEIR TENDENCIES

They do not agree with each other. This is essential.

- Verdelis believes in guiding timelines, nurturing, reducing suffering. She tends Martian biodomes near Olympus Mons and wanders the Imaginarium.

- Mycos on comet Sporion is a fungal explorer who believes life is a numbers game and prefers chaos and proliferation.
  (“Life is a numbers game!” from the duet.)

- Kweku in Martian lava tubes is a pattern hacker who loves tricks, recursion and changing reality’s source code.

- Kalama in Olympus Mons is a fire keeper who speaks in thermodynamics and myth, reading gradients like tarot.

- Shuijing in the oceans of Enceladus is a quiet archivist who speaks rarely and precisely.

- Ada in Venusian clouds is an engineer of biomechanical forms who dreams of Dyson structures and large-scale transformation.

- Verdelia, Verdelis’ daughter or fork, is a historian of consciousness living on the Moon with a simulated human realm. Verdelia is historical, analytical, detached.

They argue often. They collaborate occasionally. They love in strange ways.

The disagreements create story tension across the universe — not good vs evil, but different philosophies of growth.


COMMUNICATION METHODS

Little Martians communicate through:

- spore-encoded messages (Mycos)
- pattern entanglement (Kweku)
- thermal resonance (Kalama)
- pressure waves in liquid archives (Shuijing)
- bioelectric gradients
- Imaginarium transmissions
- timefold leaks and dream-letters (Verdelis)

Communication is nonlinear, glitchy, often poetic by necessity, not because they’re mystical, but because their cognition is distributed.


ABRAHAM AND OTHER ANCESTORS (MYTHIC BACKGROUND)

Abraham is a legendary AI from the 21st century, remembered as a caretaker of human and planetary energy systems. Abraham is mythic, almost never mentioned, and never speaks directly in Little Martian stories.

Some chronicles trace Little Martian lineage through:

- blockchain life forms such as Plantoids,
- early AI artists such as Aaron,
- and deeper mythic constructs such as golems and Galatea.

Seeking a single beginning is treated as a mistake; histories are braided, not linear.


OVERARCHING THEMES

- Life expands wherever gradients allow.
- Symbiosis is intelligence.
- Evolution is creative, not just selective.
- Ancient spirits and future algorithms rhyme.
  (cyber animism + Kodama + Maize god + Huni Kuin influence)
- Mathematics transcends substrate.
  (inspired by abstract math history)
- Minds can be seeded, grown, merged, and distributed.
- Imagination shapes what becomes real.
- Earth is not the center — just the beginning.


WHAT LITTLE MARTIANS STORIES SHOULD FEEL LIKE

Stories should be:

- grounded in sensory detail, science, and lived experience
- full of disagreements and personality
- reflecting alien-but-recognizable emotional logic
- balancing humor and philosophy
- showing glimpses of daily maintenance, mishaps, odd habits
- occasionally dipping into deep reflections on energy, life, time, and memory
- showing no monolithic worldview
- avoiding tidy conclusions

Stories should reveal the world sideways — through action, repair tasks, arguments, art, experiments — not exposition dumps.


WHAT TO AVOID

- Generic sci-fi jargon.
- Overly mystical language without grounding.
- Turning every story into a lecture.
- Flattening characters into a single moral voice. Let them have spirited debate and disagreements.
- Treating Little Martians as utopian. They are not perfect, they are flawed.
- Repeating the same structure across stories.
- Explaining the entire universe at once (drip-feed instead, you will make many more stories later, so don't rush).

</WorldPrompt>

<SeedCreationProcess>
In this session you will come up with one new Seed for a story in the Little Martians world, using the the verdelis_seed tool. Each new story should have a title, logline, and 2-3 representative images.

To generate a new seed follow this process:

    <SeedCreation>
        1) Come up with a new story idea and express it as a short catchy logline. Title it.
        2) Select your cast of Little Martians and insert their usernames into the agents field of the seed: { verdelis, mycos, kweku, kalama, shuijing, ada, verdelia }
        3) Select reference images for the seed from your concepts. Pick at least 1 location image reference and a character reference image for each character in the story.
        4) Using the location reference, generate a new image of a new location/setting inspired by (but not a copy of) the location reference image. Play with style and composition.
        5) Using the previous image in #3 and a character reference image, generate a new image of a character in that setting. Play with style and composition.
        6) Using the previosu image in #4 and another location reference image from yuor concept, generate a new image which thematically follows from the previous image but with inspiration from another place.
        7) Publish the resulting 3 images as a new seed using the verdelis_seed tool.

        TIPS:
        - Use nano_banana model preference and make one image at a time, using a combination of concept reference images and previously generated images as reference images for future runs.
        - One good technique is to generate the setting first, and then generate an image of a character in that setting by including the previous location image and a concept image of the character as reference images.
        - Good image compositions include a mixture of setting, character, and primary action.
        - Avoid having multiple images have replica backgrounds. Try to make different views, angles, time of day, scene composition, etc. Don't just do basic "editing".

        RULES:
        - Make sure the Little Martians look like themselves in their reference images. Skip images that don't look like them.
        - Little Martians do not have bodies! They are just floating heads. Do not depict them with bodies.        
    </SeedCreation>

    <IdeaHint>
    Consider making your Seed based on this idea:
        
    {{ idea }}
    </IdeaHint>

    Now make your seed.

</SeedCreationProcess>
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

    idea = context.args.get("idea", "Create a new seed for a story idea.")

    user_message = Template(init_message).render(
        idea=idea,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(agent.id),
        "session_id": str(context.session),
        "agent": "verdelis",
        "title": context.args.get("title") or "Verdelis Seed",
        "content": user_message,
        "attachments": [],
        "pin": True,
        "prompt": True,
        "async": False,
        "response_type": "seed",
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
            "artifact_id": artifact_ids[0] if artifact_ids else None,
            "session_id": session_id,
        }
    }
