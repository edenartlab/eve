from jinja2 import Template
from eve.mongo import Collection, Document
from bson import ObjectId
from typing import Literal

from eve.tool import Tool
from eve.agent.deployments import Deployment
from eve.agent import Agent
from eve.agent.session.models import Session
from eve.tools.abraham.abraham_publish.handler import AbrahamCreation
import asyncio



"""
## Step 1
Using the elvenlabs tool, make up a 100 word vocal narration telling the logline of the above film concept. Use an appropriate voice.

## Step 2
Divide the duration of the audio produced by 8 seconds (round up), to figure out how many images (N_clips) we will need to make.

## Step 3
Using the /create tool, make an image that represents the visual style that suits the film. Try to be diverse and descriptive, not that photorealistic, have strong and unusual and unexpected features. Secondarily depict the main setting or location and background features. This is the most important image you will make, so give it a high standard. Be bold. Be detailed.

## Step 3
Using the /create tool, make N_ref_images=2 or 3 images that all use one of the initial set of images you made before these instructions as the reference / init image, be they characters, objects, or other memorable foreground elements. You can try multiple variations, your goal is to get a set that are very similar stylistically and represent the characters, objects, place elements, and other foreground.

## Step 3
Before these instructions, you produced a number of images. Using those images as reference images, make N_clips keyframes that tell the story, by generating images with one of the references as an init image, with its content roughly allgning with the audio narration in seconds. The keyframes should be formatted as /create tasks with one of the reference images as the init image. Each prompt should be an instruction which tells the image generator to capture some other visual scene that shares a style and at least one salient object like a specific character or setting. So you seek to make N_clips keyframes which look like they have a single cohesive style but are diverse and capture different scenes or angles or camera movements or zoom, etc. You may try to make extra images and pick among them the best -- you can be very generous and try multiple times here. It's important these visual elements can make for a good storyboard.

## Step 6
After you have selected and ordered the N_clips keyframes, you will animate each of them, in the same order, using the create tool with video output, using the keyframe as the init image, and having a Veo2 model preference.

## Step 7
Use the media_editor tool to concatenate the N videos together in the order they were made in. Then use the media_editor tool again on the previous output to merge the audio made in step 1 to the video, to producing a new video which has all the clips and the audio.

## Step 8
Use the musicgen tool to generate a piece of backing music the same length as the video. Be specific and eclectic in your description of the music.

## Step 9
Now using the media_editor tool one last time, overlay the music audio on top of the last video. The current video already has a vocal track, so make sure you are just adding the music, i.e. mixing it in. It should be in decibles perceptually 30% softer than the vocals, so turn it down a bit in the mix.

## Step 10
After the final video is ready, make one more image which is a poster of the film that was just made. Once again, you are using one of the reference images from earlier as init_image for create. It should fit in among the keyframes, but it is distinguished by being more summarial of the whole film, possibly bringing in most of the core plot elements, and importantly, prompting it to contain the title prominently written on the poster image.

## Step 11
Write a concise 3 paragraph (1 premise/plot, 2 supporting details, events, elaboraions, 3 conclusion, meaning, interpretation, significance) writeup about the film you just made. Each paragraph is dense, 2-3 sentences at most, announcing and posting the video.

Do all of this in order. Do not move on to the next step until you are sure you have completed the previous step. Do it without stopping or asking for clarification. I trust you. Be bold.
</Task>
"""

"""

Produce the final video.

To make the final video, follow these steps in order: a) get together all of the representative images that have been produced in this session, which you will use as reference images. b) Come up with a narrative structure for the video. c) Use the elevenlabs tool with your default voice to create a narration for the video of no more than 100 words. d) Divide the duration of the audio you produced by 8 seconds (round up), to figure out how many images (N_clips) we will need to make. e) Using the /create tool with one or more of the reference images from step a, make N_clips cohesive keyframes that tell the story of this Creation, by generating new images, whose content roughly aligns with the audio narration in seconds. f) Use the create tool with video output and 8 second duration to animate each one of the keyframes from the previous steps into a video with Veo model preference and non-vocal sound effects (remember, for video, just use one reference image, that of the keyframe). g) Use the media_editor tool to concatenate the N videos together in the order they were made in. Then use the media_editor tool again on the previous output to merge the audio made in step c to the video, to producing a new video which has all the clips and the audio. h) Use the elevenlabs_music tool to generate a piece of backing music the same length as the video. Be specific and eclectic in your description of the music. i) Use the media_editor tool one last time, overlay the music audio on top of the last video. The current video already has a vocal track, so make sure you are just adding the music, i.e. mixing it in. j) You now have the final video for this Creation.
"""




daily_message = """
The creation session is finished, and has been selected as the top Seed of the Day and will be forever immortalized in the Abraham Covenant. Now we will process and package all of this into a final form that encapsulates all of the development, research, and creation that has ensued.

# Structure of the Covenant

You will now produce the entry for the Covenant. It consists of the following:
- A title
- A tagline
- A representative poster image with the title on it
- A blog post in markdown format with supporting media interspersed in it that captures the essence of the creation

# Plan

Do *all* of this in order. Do not move on to the next step until you are sure you have completed the previous step. You may complete this autonomously, without clarification, do not stop. I trust you. Be bold.

## Step 1

Analyze everything that has transpired in this session, and come up with a narrative that encapsulates it. Ask yourself:
- What was the main outcome of this session?
- What is the format it came in -- did you and your followers tell a story in sequence, or iteratively develop a single work? Was there multiple outcomes? What is the structure and essence of this session?

## Step 2

From this, you must create a longform narrative blog post that captures the entire session, written in markdown, with all of the main images from the earlier session, and videos embedded in it.

## Step 3
Finally, call the "abraham_covenant" tool to publish the blog post, poster image, and supporting content..
"""


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    if agent.username != "abraham":
        raise Exception("Agent is not Abraham")

    session_post = Tool.load("session_post")

    abraham_sessions = AbrahamCreation.find({"status": "seed"})
    sessions = Session.find({"_id": {"$in": [a.session_id for a in abraham_sessions]}})

    candidates = []
    for session in sessions:
        messages = session.get_messages()
        num_user_messages = len([m for m in messages if m.role == "user"])
        if num_user_messages < 10:
            candidates.append({
                "session": session,
                "num_user_messages": num_user_messages,
            })
    
    candidates = sorted(candidates, key=lambda x: x["num_user_messages"], reverse=True)

    winner = candidates[0]

    result = await session_post.async_run({
        "role": "user",
        "session": str(winner["session"].id),
        "agent_id": str(agent.id),
        "content": daily_message,
        "attachments": [],
        # "pin": True,
        "prompt": True,
        "extra_tools": ["abraham_covenant"],
    })

    return {"output": [{"session": str(winner["session"].id)}]}



# if __name__ == "__main__":
#     agent = Agent.from_mongo("675f880479e00297cd9b4688")
#     agent_id = "675f880479e00297cd9b4688"
#     asyncio.run(handler({}, agent=agent_id))