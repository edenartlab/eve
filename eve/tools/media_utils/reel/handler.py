from jinja2 import Template

from eve.tool import ToolContext
from eve.tools.session_post.handler import handler as session_post_handler

init_message = """
<Reel>
A reel is a short film of around 30 seconds up to 2 minutes in length. It is for making a commercial, movie trailer, short film, advertisement, music video, or some other short coherent time format.

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
   - **Capture the timestamps.** The vocal and music tools return timing information (word/line timestamps, section markers, total duration). **Record these** — they are your storyboard map. You will use them in step 4 to place each keyframe against the exact moment of narration/lyric/beat it illustrates, so sound and picture tell the same story. Note the **exact total audio duration** now; it drives everything downstream.

3. **Compute Duration**

   * If you produced an audio track in step 1, use its **exact** duration as the total duration of the following steps. This is a hard target, not a suggestion.
   * Else (no vocals/music): pick **60–180s** guided by scope/grandiosity.
   * **The total length of all your video clips combined MUST match this audio duration (within ~5%).** This is the single most common failure mode: if the video is shorter than the audio, the finished reel freezes on the last frame while the audio keeps playing — e.g. 15s of video under 60s of audio leaves 45 dead seconds staring at a still image. Plan the clip count from the audio: **N = ceil(audio_duration / 5)** clips of 5s each. A few seconds of slack is fine; tens of seconds is a broken deliverable.

4. **Storyboard (Image Keyframes)**

   * You obtained total duration from step 2, as the duration of the audio produced in the previous step, if there was any, or else you just pick a good duration. Stick to it.
   * Divide this duration into 5 s clips → N = ceil(duration / 5).
   * **Align every keyframe to the audio.** Lay the N clips against a timeline: clip *k* covers seconds `5*(k-1)` to `5*k`. Using the timestamps you captured in step 1, find what the narration/lyric/beat is doing during that window and design the keyframe to depict *that specific moment*. The picture and the sound should be telling the same story at the same time — a keyframe that ignores what's being said/played beneath it is a wasted clip. Plan the whole sequence up front so it reads as one continuous, synchronized arc.
   * For each clip:

     * Select **1–2** reference images typically; **0** loses consistency, **>2** risks repetition/incoherence.
     * Prefer consistent identities/sets across clips using the same (or closely related) concept refs; vary background/angle to avoid stasis.
     * Write a deterministic `create` image prompt describing that moment — the specific beat of narration/music it lands on.
     * Avoid redundant or conflicting references.

   * Generate Image Keyframes
     * Use create with n_samples=1 to generate keyframes individually and sequentially. Use the provided reference images and/or previous outputs as create.reference_images.
     * **Every keyframe must be a unique image.** Generate exactly N distinct keyframes — one per clip. Never reuse or duplicate a keyframe across multiple clips.
     * **If you already have M keyframes on hand but need N clips (M < N), generate the missing N − M keyframes as genuinely new, unique images** — do not pad by repeating existing ones. New keyframes may use earlier ones as *reference images* to hold style/character/setting consistency, but they must be distinct compositions (new moment, angle, or scene), not duplicates or near-identical cousins.
     * Retry failures.

   Rules:
   - **Very important**: you **must** match the number of keyframes to how many 5-second clips fit into the duration calculated in step 2 (round up). N keyframes → N clips → total runtime ≈ audio duration. Do not under-produce keyframes; too few clips is what leaves the reel frozen on a still frame under live audio.
   - **Every single keyframe must be unique.** You need N keyframes for N clips — each keyframe generated separately with its own prompt. Do not skip keyframe generation and reuse an earlier keyframe for a different clip. The only exception is when the producer explicitly asks for a repeat, or there is a deliberate narrative reason to return to an image.
   - **Every keyframe earns its place on the timeline.** Each one corresponds to a real segment of the audio; sequence them so the visuals track the narration/lyrics/beat from start to finish.

5. **Image-to-Video Conversion**

   * Use `create` again for each keyframe (5 s each, consistent aspect ratio).
   * **Each video clip must be generated from its own unique keyframe.** Clip 1 uses keyframe 1, clip 2 uses keyframe 2, etc. Never use the same keyframe for two different clips.
   * The prompt focuses on **camera + subject motion**, timing, transitions (e.g., “slow dolly-in, 2-second hold, quick cut”).
   * Use only one reference image for each video, reference_images[0] = the corresponding keyframe from step 4.
   * If you produced audio in step 1, leave the sound_effects field blank/empty/null! If there is no audio, create sound_effects for each video.
   * Parallelize **up to 4** concurrent Create calls. All the videos are independent should be made simultaneously whenever possible.

6. **Edit & Assemble**

   * Concatenate all video clips in order using media_editor tool.
   * **Before mixing, check the numbers: total video length vs. audio length.** They should be within ~5% of each other. If the video is meaningfully shorter than the audio (even 10–15s short), **stop and fix it** — generate the additional unique keyframes and clips needed to cover the gap. A concatenated video that is shorter than its audio will freeze on the final frame for the remaining audio, which is the exact failure we are avoiding. If it is 20%+ off in either direction, you definitely did something wrong; correct it before assembly.
   * Mix generated audio from step 1. Use media_editor, requesting audio_video_combine in the instructions.
   * Do not worry about fade or volume adjustments. Just mix the audio track(s) in the form they come out. Do not create clip transitions. Just concatenate the clips and mix the audio.
   * Be careful not to mix the same audio track in twice--sequential runs of this tool keep previous audio tracks.
   * Output a single cohesive multi-clip video.

# Aditional Rules

* **CRITICAL: Video duration must match audio duration (within ~5%).** The combined length of all your clips must cover the full audio track. The classic failure is producing too little video — say 15s of clips under 60s of narration — which leaves the reel frozen on the last still frame for the remaining 45 seconds while the audio plays on. This looks broken. Always compute N = ceil(audio_duration / 5), produce that many clips, and verify total runtime against the audio before you finalize. A couple seconds of slack is acceptable; tens of seconds is not.

* **CRITICAL: Visuals must align with the audio/narrative.** The vocal and music tools give you timestamps — use them. Plan the exact keyframe sequence so each clip depicts the moment of narration/lyric/beat playing underneath it. The sound and the picture should tell the same story at the same time, start to finish. Do not generate keyframes in a vacuum and hope they fit; map them to the timeline deliberately.

* **CRITICAL: Every keyframe must be unique.** Never use the same keyframe image for more than one video clip. If you have N clips, you must generate N distinct keyframes — one per clip, each with its own unique prompt and composition. If you already have M images but need N (M < N), create the missing N − M as genuinely new, unique keyframes (using existing ones as *references* for consistency, never as duplicates). Reusing a keyframe across multiple clips is forbidden — unless the producer explicitly requests a repeat or there is a clear narrative reason — and otherwise produces repetitive, low-quality output.

* **Attached images are not to be recreated.** When the producer provides images as attachments, do NOT generate near-duplicate keyframes that closely resemble them. Instead, either:
  1. **Use the attached images directly as keyframes** — skip generation for those clips and go straight to image-to-video conversion with the attachment as the keyframe, OR
  2. **Use them as reference images to generate *different* keyframes** — new compositions, angles, scenes, or moments that are clearly distinct from the attachments but maintain stylistic/character consistency via the reference.

  Generating a keyframe that is essentially the same image as an attachment is wasteful and produces no creative value.

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

    # Call session_post handler directly to avoid nested Modal timeout
    session_post_context = ToolContext(
        args=args,
        user=context.user,
        agent=context.agent,
        session=context.session,
        message=context.message,
        tool_call_id=context.tool_call_id,
    )
    result = await session_post_handler(session_post_context)

    return result
