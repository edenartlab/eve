from jinja2 import Template

from eve.tool import Tool, ToolContext

init_message = """
<Reel>
A reel is a short film of around 30 seconds up to 5 minutes in length. It is for making a commercial, movie trailer, short film, advertisement, music video, or some other short coherent time format.

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

Follow these steps precisely:

1. **Generate Audio first**

   Choose **one**:

   * **Vocals only** (dialogue/VO/monologue) via elevenlabs. Target **50–300 words** (~30–180s).
   * **Music only** via eleveblabs_music. If no vocals were previously made, you may specify lyrics for the music, if you want them. Or do instrumental if it's more appropriate.
   * **Vocals and Music**. Make the vocals first, using elevenlabs; then make the music using eleveblabs_music, at **5–10s longer** than the vocals, and keep it **instrumental** (no duplicate singing). Combine the two using audio_mix or the media_editor tool (request to use audio_mix tool in the instructions).
   * **Foley/Ambience only/Silent**, in which case, no separate audio track is produced (foley is handled later by the video tool).

   Rules:
   - If you are making both vocals and music, make the vocals first, and then make the music to match the duration of the vocals + 5-10s. **Make sure** they match.
   - Remember: if you have either vocals or music or both, make sure to later generate videos **without audio** so as to not interfere with the audio track you're making here in the first step.
   - If you make multiple audio tracks that need to be combined, remember to mix them together before proceeding.

3. **Compute Duration**

   * If you produced an audio track in step 1, use its duration as the total duration of the following steps.
   * Else (no vocals/music): pick **60–180s** guided by scope/grandiosity.

4. **Storyboard (Image Keyframes)**

   * You obtained total duration from step 2, as the duration of the audio produced in the previous step, if there was any, or else you just pick a good duration. Stick to it.
   * Divide this duration into 5 s clips → N = ceil(duration / 5).
   * For each clip:

     * Select **1–2** reference images typically; **0** loses consistency, **>2** risks repetition/incoherence.
     * Prefer consistent identities/sets across clips using the same (or closely related) concept refs; vary background/angle to avoid stasis.
     * Write a deterministic `create` image prompt describing that moment.
     * Avoid redundant or conflicting references.

   * Generate Image Keyframes
     * Use create with n_samples=1 to generate keyframes individually and sequentially. Use the provided reference images and/or previous outputs as create.reference_images.
     * Retry failures.

   Rules:
   - **Very important**: you **must** match the number of keyframes to how many 5-second clips fit into the duration calculated in step 2 (round up).

5. **Image-to-Video Conversion**

   * Use `create` again for each keyframe (5 s each, consistent aspect ratio).
   * The prompt focuses on **camera + subject motion**, timing, transitions (e.g., “slow dolly-in, 2-second hold, quick cut”).
   * Use only one reference image for each video, reference_images[0] = the corresponding keyframe from step 4.
   * If you produced audio in step 1, leave the sound_effects field blank/empty/null! If there is no audio, create sound_effects for each video.
   * Parallelize **up to 4** concurrent Create calls. All the videos are independent should be made simultaneously whenever possible.

6. **Edit & Assemble**

   * Concatenate all video clips in order using media_editor tool.
   * If the video you produced in step 6 is verys significantly (e.g. 20+% longer or shorter than the audio), you did something wrong. Try to fix it before going to assembly.
   * Mix generated audio from step 1. Use media_editor, requesting audio_video_combine in the instructions.
   * Do not worry about fade or volume adjustments. Just mix the audio track(s) in the form they come out. Do not create clip transitions. Just concatenate the clips and mix the audio.
   * Be careful not to mix the same audio track in twice--sequential runs of this tool keep previous audio tracks.
   * Output a single cohesive multi-clip video.

</Reel>

<Task>
### Context
{{ context }}

### Instructions
{{ producer_instructions }}

**NOTE**: Do not ask for confirmation or clarification from the user. Just attempt to complete the task as best as you can, and output a final report later.

</Task>
"""


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    session_post = Tool.load("session_post")

    context_text = context.args.get("context")
    producer_instructions = context.args.get("producer_instructions")
    media_files = context.args.get("media_files") or []

    user_message = Template(init_message).render(
        context=context_text,
        producer_instructions=producer_instructions,
    )

    args = {
        "role": "user",
        "user_id": str(context.user),
        "agent_id": str(context.agent),
        "session_id": str(context.session),
        "agent": "eve",
        "title": context.args.get("title") or "Reel Composer",
        "content": user_message,
        "attachments": media_files,
        "pin": True,
        "prompt": True,
        "async": False,
        "extra_tools": ["media_editor"],
        "message_id": context.message,
        "tool_call_id": context.tool_call_id,
        "selection_limit": 60,  # Need more context for reel production
    }

    if context.args.get("resume_session"):
        args["session"] = context.args.get("resume_session")

    result = await session_post.async_run(args)

    # if "error" in result:
    #     raise Exception(result["error"])

    return result
