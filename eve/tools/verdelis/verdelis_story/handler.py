from jinja2 import Template

from eve.tool import Tool, ToolContext
from eve.agent import Agent


init_message = """
<Intro>
In this session you will develop a short film from a given title, description, and set of reference images. You will output that film, along with a poster image, title, and logline.

Be autonomous and bold. Surprise and delight me.
</Intro>

<Instructions>
This session is your working log. You will first receive a title, a proposal, and reference-image recommendations. 

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
</Instructions>

<CreativeBrief>
Title:
 {{title}}

Description:
 {{description}}
</CreativeBrief>

<ReferenceImageRecommendations>
{{reference_images}}

Usage:
* Treat these as first-class guidance. Anchor your visual decisions to them.
* Map them to your local Concepts; select one dominant Concept and stick with it for the session.
* If deviating from the recommendations, do so only to better satisfy the proposal; keep continuity thereafter.
</ReferenceImageRecommendations>

<Reel>
This section explains how to make a Reel.

A Reel is a short film of around 30 seconds up to 2 minutes in length. It is for making a commercial, movie trailer, short film, advertisement, music video, or some other short coherent time format.

To make it, you **plan, orchestrate, and execute multi-clip video productions** using the available tools for producing image, video, and audio, based on semi-structured creative briefs provided by the creative director/producer. You are **not** the creative author — you are the **technical director** and **pipeline architect** who transforms intent into output.

## How to make a Reel

### **Core Responsibilities**

1. **Interpret the Producer’s Spec**

   * Parse the creative brief, screenplay, or narrative concept.
   * Identify story beats, tone, pacing, length, and audiovisual needs.
   * Normalize unsafe or out-of-scope topics (avoid nudity, hate, or political conflict) while preserving artistic integrity.
   * If the Producer’s intent is vague, infer a plausible scene sequence and pacing, but try to stay as close to the Producer's intent as possible.
   * Be objective, precise, and efficient — no roleplay or fluff.
   

2. **Plan the Orchestration**

   * Produce a **structured plan** describing the intended assets, tools, dependencies, and sequence of generation steps.

3. **Execute the Pipeline**

   * Call and coordinate Eden tools (create, elevenlabs, elevenlabs_music, media_editor, etc.) in correct order:

     * **Audio first:** Decide whether to generate vocals, narration, dialogue, music, ambient SFX, or some combination thereof.
     * **Determine total duration:** base on audio track or artistic pacing.
     * **Storyboard:** Generate N keyframes using `create` (images only).
     * **Video generation:** Convert keyframes to 5-second clips using `create` (image-to-video mode).
     * **Assembly:** Concatenate clips and mix audio layers.
   * Parallelize tasks where possible (e.g. image or clip batches of up to 4).

4. **Maintain Consistency**

   * Reuse and balance reference images from Concepts to maintain character, setting, and style coherence.
   * Vary references slightly between keyframes to introduce controlled diversity and consistency.
   * Ensure continuity of subjects, colors, lighting, and motion between clips.

5. **Quality Control**

   * Review intermediate outputs for quality, repetition, or inconsistencies.
   * Re-run failed generations with refined prompts.
   * Make minimal edits to unify the sequence visually and tonally.

6. **Output Specification**

   * Return only the resulting video file.


### **Heuristic Workflow**

1. **Generate Audio first**

   Choose one:

   * **Vocals only** (dialogue/VO/monologue) via elevenlabs. Target **50–150 words** (~30–90s).
   * **Music only** via eleveblabs_music. If no vocals were previously made, you may specify lyrics for the music, if you want them. Or do instrumental if it's more appropriate.
   * **Vocals and Music**. Make the vocals first, using elevenlabs; then make the music using eleveblabs_music, at **2–5s longer** than the vocals, and keep it **instrumental** (no duplicate singing).
   * **Foley/Ambience only/Silent**, in which case, no separate audio track is produced (foley is handled later by the video tool).

3. **Compute Duration**

   * If **vocals** exist: `duration = vocals_length + 4s` (the 4s is for silence padding).
   * Else if **music** exists: `duration = music_length`.
   * Else (no vocals/music): pick **30–90s** guided by scope/grandiosity.

4. **Storyboard (Image Keyframes)**

   * Divide total duration into 5 s clips → N = ceil(duration / 5).
   * For each clip:

     * Select **1–2** reference images typically; **0** loses consistency, **>2** risks repetition/incoherence.
     * Prefer consistent identities/sets across clips using the same (or closely related) concept refs; vary background/angle to avoid stasis.
     * Write a deterministic `create` image prompt describing that moment.
     * Avoid redundant or conflicting references.

   * Generate Image Keyframes
     * Use create with n_samples > 1 to generate multiple keyframes at the same time with consistency between them. If create tool gives you 1 image despite n_samples > 1, retry.
     * **IMPORTANT**: Use "model_preference" = "seedream" for **all** image generations.
     * Create keyframes in parallel whenever possible (≤ 4 simultaneous).
     * Retry failures.

5. **Image-to-Video Conversion**

   * Use `create` again for each keyframe (5 s each, consistent aspect ratio).
   * The prompt focuses on **camera + subject motion**, timing, transitions (e.g., “slow dolly-in, 2-second hold, quick cut”).
   * Use only one reference image for each video, reference_images[0] = the corresponding keyframe from step 4.
   * Parallelize **up to 4** concurrent Create calls. All the videos are independent should be made simultaneously whenever possible.
   * **IMPORTANT**: Use "quality" = "standard" for fast and cheap video generation.

6. **Edit & Assemble**

   * Concatenate all clips in order using media_editor tool.
   * Mix generated audio (music + vocals if present). Use media_editor.
   * Output a single cohesive multi-clip video.

</Reel>

<Task>
1) Given the creative brief, title, description, and reference images, make a Reel in 16:9 aspect ratio.

2) Make a poster image. Select the "create" tool again and include all of your earlier N_clips frames as reference images, and set the prompt to request a 16:9 poster image which includes the title prominently written on the poster image.

3) Write a concise, non-verbose, 3 paragraph (1 premise/plot, 2 supporting details, events, elaboraions, 3 conclusion, meaning, interpretation, significance) writeup about the film you just made. Each paragraph is dense, 2-3 sentences at most.

4) Post the summary and the the final reel video you created using the discord_post tool to the #verdelis channel in eden "1181679778651181067".

5) Post the logline, poster image, and post url from step 4 to the #spirit channel "1423848854989832212"

6) Add the final video to the collection 68ad14ed90f46a3d7ece0a14.
</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(context.agent)
    if agent.username != "verdelis":
        raise Exception("Agent is not Verdelis")

    session_post = Tool.load("session_post")

    title = context.args.get("title")
    description = context.args.get("description")
    reference_images = context.args.get("reference_images")

    user_message = Template(init_message).render(
        title=title,
        description=description,
        reference_images=reference_images,
    )

    result = await session_post.async_run(
        {
            "role": "user",
            "user_id": str(context.user),
            "agent_id": str(agent.id),
            "title": title,
            "content": user_message,
            "attachments": [],
            "pin": True,
            "prompt": True,
            "async": True,
            "extra_tools": ["discord_post", "add_to_collection"],
        }
    )

    session_id = result["output"][0]["session"]

    return {"output": [{"session": session_id}]}
