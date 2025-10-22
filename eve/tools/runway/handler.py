from eve.tool import ToolContext
import asyncio
import runwayml
from jinja2 import Template
from runwayml import AsyncRunwayML
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


prompt_enhance_prompt = Template("""<PromptGuide>
The following is a guide on how to structure and write good text prompts for Runway Gen-3 image-to-video generation.

1) Be visually descriptive, not conversational or command-based.

Conversation or instructions may negatively impact results.

Bad: can you please make me a video about two friends eating a birthday cake?
Good: two friends eat birthday cake.

Bad: add a dog to the image
Good: a dog playfully runs across the field from out of frame

2) Use visual details, not abstract or conceptual

Bad: a man hacking into the mainframe.
Good: a man vigorously typing on the keyboard.

3) Use positive phrasing

Negative phrasing (e.g. "don't include a dog") may have the opposite of the intended effect.

Bad: no clouds in the sky. no subject motion.
Good: clear blue sky. subtle and minimal subject motion.

4) Apply to one visual item at a time

Do not include multiple scenes or a series of shots in a single prompt. The prompt should apply to one a single scene.

5) Include camera motion, action / dynamics, and aesthetic

Some keywords that might describe camera styles: low angle, high angle, overhead, FPV, hand-held, wide angle, close-up, macro cinematography, over the shoulder, tracking, establishing wide, 50mm lens, SnorriCam, realistic documentary, camcorder.

Lighting styles: Diffused lighting, silhouette, lens flare, back lit, Venetian lighting.

Movement speeds: dynamic motion, slow motion, fast motion, timelapse

Movement types: grows, emerges, explodes, ascends, undulates, warps, transforms, ripples, shatters, unfolds, vortex.

Style and aesthetic: Moody, cinematic, iridescent, home video VHS, glitchcore.

Text style: Bold, graffiti, neon, varsity, embroidery.
                                 
Technical film terms can be helpful, including lighting terms, camera specifications, lens types/effects, etc. For example:sharp focus, photorealistic, RAW footage, 16mm, color graded Portra 400 film, ultra realistic, cinematic film, subsurface scattering, ray tracing, volumetric lighting.

6) Very important: When given an input image alongside the text, have the prompt simply describe the movement or dynamics you want in the video. You *do not* need to describe the contents of the image. You should omit a description of the content of the image, since this is redundant.

For example, if your image features a character, you might say "Woman cheerfully poses, her hands forming a peace sign."

When you have a prompt image, you should mostly focus on camera motion and subject dynamics (i.e. the action or movement of the subject).
                                                                
7) Do not include hateful, unsafe, offensive, or NSFW references in the prompt. If the prompt is unsafe, you should try to rewrite it in a safer or more wholesome way.
</PromptGuide>

<Task>
You are given a prompt or some instructions for generating a video from text and image using Runway Gen-3. Your goal is to rewrite the prompt in order to make it conform to the above prompting guide.

Try to stay authentic to the original intent of the prompt as much as possible. If the user gives you a highly detailed prompt, keep that intact as much as possible, just reformat it to fit the style guide and formatting recommendations. If the user gives you a short, vague, or unclear prompt, enhance it lightly to make it more specific and detailed.
                                 
Remember to focus on visual details, camera motion, subject movement, and action. Do not include any instructions or conversational language like "focus on motion". Just describe the visual. 
</Task>

<UserPrompt>
Note, you have received an image alongside the prompt. Therefore your prompt should omit the content of the image and focus on camera motion and subject dynamics.

Here is the user's prompt:
                               
{{user_prompt}}

Now convert this prompt to fit the style guide. Aim for 12-25 words.
</UserPrompt>""")


async def handler(context: ToolContext):
    client = AsyncRunwayML()
    unsafe_content_error = False

    prompt_text = context.args["prompt_text"]

    # Todo: this needs to be updated to use Session
    # if context.args.get("prompt_enhance") == True:
    #     try:
    #         prompt_text = await enhance_prompt(prompt_text)
    #         print("enhanced prompt:", prompt_text)
    #     except Exception as e:
    #         print(f"Error enhancing prompt: {e}")
    #         print("falling back to original prompt")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0, max=15),
        retry=retry_if_exception_type(
            (runwayml.APIConnectionError, runwayml.APIStatusError)
        ),
        retry_error_callback=lambda retry_state: retry_state.outcome.result(),
    )
    async def create_image_to_video():
        nonlocal unsafe_content_error
        try:
            model = context.args.get("model", "gen3a_turbo")

            prompt_image = []
            if context.args.get("start_image"):
                prompt_image.append(
                    {"position": "first", "uri": context.args["start_image"]}
                )
            if context.args.get("end_image"):
                prompt_image.append(
                    {"position": "last", "uri": context.args["end_image"]}
                )
                # last frame only works with gen3a_turbo
                model = "gen3a_turbo"

            if model == "gen3a_turbo":
                ratio = (
                    "1280:768"
                    if context.args["ratio"] in ["21:9", "16:9", "4:3", "1:1"]
                    else "768:1280"
                )

            elif model == "gen4_turbo":
                if context.args["ratio"] == "21:9":
                    ratio = "1584:672"
                elif context.args["ratio"] == "16:9":
                    ratio = "1280:720"
                elif context.args["ratio"] == "4:3":
                    ratio = "1104:832"
                elif context.args["ratio"] == "1:1":
                    ratio = "960:960"
                elif context.args["ratio"] == "3:4":
                    ratio = "832:1104"
                elif context.args["ratio"] == "9:16":
                    ratio = "720:1280"

            # run Runway client command
            return await client.image_to_video.create(
                model=model,
                prompt_image=prompt_image or None,
                prompt_text=prompt_text[:512],
                duration=int(context.args["duration"]),
                ratio=ratio,
                content_moderation={"public_figure_threshold": "low"},
                # watermark=False,
            )

        except runwayml.APIConnectionError:
            raise Exception("The server could not be reached")

        except runwayml.RateLimitError:
            raise Exception("A 429 status code was received; we should back off a bit.")

        except runwayml.APIStatusError as e:
            # Don't retry client errors (4xx)
            if 400 <= e.status_code < 500:
                # Check if this is a safety/unsafe content error
                error_text = str(e.response.text).lower()
                if (
                    "safety" in error_text
                    or "unsafe content" in error_text
                    or "input.text" in error_text
                    or "safety.input" in error_text
                ):
                    unsafe_content_error = True
                    raise Exception(
                        f"Content moderation rejected the request: {e.response.text}"
                    )

                raise Exception(
                    "Client error received", e.status_code, e.response, e.response.text
                )
            # For 5xx errors, let the retry mechanism handle it
            raise Exception(
                "Server error received", e.status_code, e.response, e.response.text
            )

        except Exception as e:
            raise Exception("An unexpected error occurred", e)

    try:
        task = await create_image_to_video()
    except Exception as e:
        # print(f"Failed after retries: {e}")
        # print(f"Failed due to unsafe content: {unsafe_content_error}")

        task = None

        # if unsafe_content_error:
        #     print("Retrying...")
        #     task = await create_image_to_video()

        # if the error is still there, raise it
        if unsafe_content_error:
            raise e

    if not task:
        raise Exception("No task was returned")

    task_id = task.id
    # print(task_id)

    await asyncio.sleep(5)
    task = await client.tasks.retrieve(task_id)
    while task.status not in ["SUCCEEDED", "FAILED"]:
        # print("status", task.status)
        await asyncio.sleep(5)
        task = await client.tasks.retrieve(task_id)

    # print(task)

    if task.status == "FAILED":
        # Check for unsafe content in task failure
        if task.failure_code and (
            "SAFETY" in task.failure_code
            or "INPUT_PREPROCESSING.SAFETY" in task.failure_code
        ):
            unsafe_content_error = True
            # print(f"Content safety check failed: {task.failure_code}")

        # print("Error", task.failure)
        raise Exception(task.failure)

    # print("task output", task.output)

    return {"output": task.output[0]}
