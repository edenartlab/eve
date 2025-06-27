import traceback
import random
import openai
import requests
import tempfile
import instructor
from typing import List
from datetime import datetime, timedelta, timezone
from bson.objectid import ObjectId
from pydantic import BaseModel, Field, ConfigDict
from PIL import Image
from io import BytesIO
import aiohttp
import os
import sentry_sdk

from eve import eden_utils
from eve.agent import Agent, refresh_agent
from eve.agent.deployments import Deployment
from eve.task import Task
from eve.tool import Tool
from eve.mongo import get_collection
from eve.models import Model
from eve.api.api_requests import ChatRequest, UpdateConfig


async def cancel_stuck_tasks():
    tasks = get_collection(Task.collection_name)

    expired_tasks = tasks.find(
        {
            "status": {"$nin": ["completed", "failed"]},
            "$or": [
                {
                    "tool": {"$nin": ["flux_trainer"]},
                    "createdAt": {
                        "$lt": datetime.now(timezone.utc) - timedelta(hours=3)
                    },
                },
                {
                    "tool": {"$in": ["flux_trainer"]},
                    "createdAt": {
                        "$lt": datetime.now(timezone.utc) - timedelta(hours=12)
                    },
                },
            ],
        }
    ).sort("createdAt", 1)

    # Convert cursor to list since we can't async iterate directly
    expired_tasks_list = list(expired_tasks)

    print("Cancelling expired tasks", expired_tasks_list)

    for task in expired_tasks_list:
        print(f"Cancelling expired task {task['_id']}")

        task = Task.from_schema(task)

        try:
            tool = Tool.load(key=task.tool)
            await tool.async_cancel(task, force=True)

        except Exception as e:
            print(f"Error canceling task {str(task.id)} {task.tool}", e)
            task.update(status="failed", error="Tool not found")
            sentry_sdk.capture_exception(e)
            traceback.print_exc()


def download_nsfw_models():
    from transformers import AutoModelForImageClassification, ViTImageProcessor

    AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
    ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")


async def run_nsfw_detection():
    import torch
    from PIL import Image
    from transformers import AutoModelForImageClassification, ViTImageProcessor

    model = AutoModelForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection",
        cache_dir="model-cache",
    )
    processor = ViTImageProcessor.from_pretrained("Falconsai/nsfw_image_detection")

    image_paths = [
        "https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/62946527441201f82e0e3d667fda480e176e9940a2e04f4e54c5230665dfc6f6.jpg",
        "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/bb88e857586a358ce3f02f92911588207fbddeabff62a3d6a479517a646f053c.jpg",
    ]

    images = [
        Image.open(eden_utils.download_file(url, f"{i}.jpg")).convert("RGB")
        for i, url in enumerate(image_paths)
    ]

    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

    print(logits)
    predicted_labels = logits.argmax(-1).tolist()
    print(predicted_labels)
    output = [
        model.config.id2label[predicted_label] for predicted_label in predicted_labels
    ]
    print(output)

    # Sort image paths based on safe logit values
    first_logits = logits[:, 0].tolist()
    sorted_pairs = sorted(
        zip(image_paths, first_logits), key=lambda x: x[1], reverse=True
    )
    sorted_image_paths = [pair[0] for pair in sorted_pairs]

    print("\nImage paths sorted by first logit value (highest to lowest):")
    for i, path in enumerate(sorted_image_paths):
        print(f"{i+1}. {path} (logit: {sorted_pairs[i][1]:.4f})")


async def generate_lora_thumbnails():
    tasks = get_collection(Task.collection_name)
    models = get_collection(Model.collection_name)

    models_with_no_thumbnails = models.find(
        {
            "base_model": "flux-dev",
            "thumbnail": {
                "$in": [
                    None,
                    "a7d63343652cc8a45831d5a580fc6be091e69217ef3e341e353c117ac94eaadd.jpg",
                ]
            },
        }
    ).sort("createdAt", 1)

    # models_with_no_thumbnails = list(models_with_no_thumbnails)

    for model in models_with_no_thumbnails:
        print(f"Making thumbnails for model {model['_id']}")

        try:
            # Flux-dev - generate thumbnail and upload
            if model.get("base_model") == "flux-dev":
                tool = Tool.load(key="flux_dev_lora")
                lora_mode = model.get("lora_mode")
                prompts = await generate_prompts(lora_mode)
                thumbnails = []

                for prompt in prompts:
                    print(f"Generating thumbnail: {prompt}")

                    async def generate_thumbnail():
                        result = await tool.async_run(
                            {
                                "prompt": prompt,
                                "lora": str(model["_id"]),
                                "lora_strength": 1.0,
                                "aspect_ratio": "1:1",
                            }
                        )
                        result = eden_utils.prepare_result(result)
                        output = result.get("output")
                        if output:
                            url = output[0].get("url")
                            response = requests.get(url)
                            img = Image.open(BytesIO(response.content))
                            return img

                    thumbnail = await eden_utils.async_exponential_backoff(
                        generate_thumbnail,
                        max_attempts=3,
                        initial_delay=1,
                    )
                    thumbnails.append(thumbnail)

                assert (
                    len(thumbnails) == 4
                ), f"Expected 4 thumbnails, got {len(thumbnails)}"

                print("Thumbnails", thumbnails)

                # Create blank canvas for 2x2 grid
                dim = thumbnails[0].size[0]  # All images are same size
                grid = Image.new("RGB", (dim * 2, dim * 2))

                # Paste images into grid
                grid.paste(thumbnails[0], (0, 0))
                grid.paste(thumbnails[1], (dim, 0))
                grid.paste(thumbnails[2], (0, dim))
                grid.paste(thumbnails[3], (dim, dim))
                grid = grid.resize((2048, 2048), Image.Resampling.LANCZOS)

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as f:
                    grid.save(f.name)
                    output = eden_utils.upload_result(
                        f.name, save_thumbnails=True, save_blurhash=True
                    )

                thumbnail = output.get("filename")

            # SDXL - just reupload existing thumbnail
            else:
                output = eden_utils.upload_result(
                    model.get("thumbnail"), save_thumbnails=True, save_blurhash=True
                )
                thumbnail = output.get("filename")

            print("final", thumbnail)

            if thumbnail:
                task_id = ObjectId(model["task"])
                models.update_one(
                    {"_id": model["_id"]},
                    {"$set": {"thumbnail": thumbnail, "thumbnail_prompts": prompts}},
                )
                tasks.update_one(
                    {
                        "_id": task_id,
                        "status": "completed",
                        "tool": "flux_trainer",
                    },
                    {"$set": {"result.0.output.0.thumbnail": thumbnail}},
                )
                print(f"updated task {task_id}")

        except Exception as e:
            print("Error generating thumbnails", e)
            sentry_sdk.capture_exception(e)
            traceback.print_exc()


async def generate_prompts(lora_mode: str = None):
    if lora_mode == "face":
        sampled_prompts = random.sample(FACE_PROMPTS, 16)
    elif lora_mode == "object":
        sampled_prompts = random.sample(OBJECT_PROMPTS, 16)
    elif lora_mode == "style":
        sampled_prompts = random.sample(STYLE_PROMPTS, 16)
    else:
        sampled_prompts = random.sample(ALL_PROMPTS, 16)

    class Prompts(BaseModel):
        """Exactly FOUR prompts for image generation models about <Concept>"""

        prompts: List[str] = Field(
            ..., description="A list of 4 image prompts about <Concept>"
        )

        model_config = ConfigDict(
            json_schema_extra={
                "examples": [
                    {"prompts": sampled_prompts[i : i + 4]} for i in range(0, 16, 4)
                ]
            }
        )

    prompt = """Come up with exactly FOUR (4, no more, no less) detailed and visually rich prompts about <Concept>. These will go to image generation models to be generated. Prompts must contain the word <Concept> at least once, including the angle brackets."""

    if lora_mode == "face":
        prompt += " The concept refers to a specific person."
    elif lora_mode == "object":
        prompt += " The concept refers to a specific object or thing."
    elif lora_mode == "style":
        prompt += " The concept refers to a specific style or aesthetic."

    try:
        client = instructor.from_openai(openai.AsyncOpenAI())
        result = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an artist."},
                {"role": "user", "content": prompt},
            ],
            response_model=Prompts,
        )
        prompts = result.prompts

    except Exception:
        print("failed to sample new prompts, falling back to old prompts")
        prompts = random.sample(sampled_prompts, 4)

    return prompts


FACE_PROMPTS = [
    "an illustrated black and white fashion sketch full figure drawing of <Concept> from a vogue magazine cover",
    "an illustrated black and white completely desaturated line drawing fashion sketch 2d isolated full figure drawing of <Concept> alone single character figure from a vogue magazine cover black and white no color fully desaturated sewing pattern vector drawing ",
    "<Concept> in the style of Lisa Frank, Rainbow 2D lisa frank trapper keeper cover colorful patterns illustration comic cartoon hard edge style, a dolphin jumping over water with rainbow in background lisa frank",
    "A vibrant, rainbow-colored sticker in the style of Lisa Frank with shiny, heavy glitter accents <Concept> in the style of Lisa Frank, Rainbow 2D lisa frank trapper keeper cover colorful patterns illustration comic cartoon hard edge style, a dolphin jumping over water with rainbow in background lisa frank",
    "A majestic portrait of <Concept> as a Roman Emperor in 50 BC, mounted atop a powerful white stallion adorned with ornate golden reins and a deep purple saddle cloth embroidered with imperial symbols. He wears a flowing crimson cape, gleaming gold-plated armor, and a laurel wreath crown upon his head. The magnificent Colosseum towers behind him in pristine condition, its travertine limestone walls glowing warmly in the late afternoon Mediterranean sun. Roman legionnaires stand at attention with their rectangular shields and red cloaks, while senators in togas gather on the steps. ",
    "a depiction of <Concept> as a lego figure, 3d model plastic lego man, full body framed 3d render showing the full figure, lego hands and feet, standing alone, one character",
    "Image of <Concept> in the style of iconic Grand Theft Auto V loading screen, leaning against a classic car with sunset over palm trees. Overall image should give impression of professional artwork full of vibrant colors, perfect light and thick outlines, embodying the essence of GTA's thematic imagery.",
    "Classic American traditional tattoo art of the face of <Concept> as a tiny, mildly infected tattoo on a man's flexing bicep, Bold lines, desaturated faded desaturated classic tattoo art, retro flair, symbolic, tattoo pattern anchor and roses",
    "A marble statue of <Concept>, completely solid marble, all white with fine marble patterns",
    "A Hand-drawn 80s sci-fi paperback novel cover featuring characters and a futuristic vehicle, prominently featuring <Concept> as the main protaganist accompanied by a diverse motley crew of exactly three other distinct characters that don't look anything like the lead, featuring an alien, a child, and a femme fatale. The style has a strong classic 1980s aesthetic hand painted acrylic on paper look, with soft mildly desaturated features, kind of like the cover of a paperback novel. The coor palette is dark and mysterious, with lots of negative space and an outerspace star speckled background with bold accents.",
    "A Hand-drawn 80s sci-fi movie poster about an action movie with characters and vehicles, prominently featuring <Concept> as the main protaganist accompanied by a diverse motley crew of exactly three other distinct characters that don't look anything like the lead, featuring an alien, a child, and a femme fatale. The style has a strong classic 1980s aesthetic hand drawn illustrated look, with soft mildly desaturated features, kind of like the cover of a paperback novel. The coor palette is dark and mysterious, with lots of negative space and an outerspace star speckled background with bold accents",
    "A mysterious figure, <Concept> in a hoodie stands surrounded by glowing green matrix computer code, 1980s hacker movie aesthetic, mr robot",
    "A masterfully composed artwork depicting <Concept> standing amid an ancient forest of towering redwoods, dappled golden sunlight filtering through the emerald canopy above. His tribal markings and battle-scarred armor blend with the organic patterns of moss-covered trees, while luminous forest spirits dance ethereally in the misty background, creating an enchanted atmosphere.",
    "A sharp, high resolution photograph of hundreds of people marching down the streets of New York in heavy armor. <Concept> is walking in front. While everyone else is panicking and running, they are steadily marching down the street, holding their assault rifles pointed down at the ground. Swat teams and FBI agents watch in disbelief as the army takes over the city. Photorealistic action shot, dramatic lighting, shot on Hasselblad H6D-400C.",
    "a portrait of <Concept> as a futuristic dystopian bond villain, eye patch and prominent scar on the face, dune, slytherin cinematic blade runner cyberpunk lighting high contrast cinema still film grain cruel grin crushed blacks bokeh fleurescent shallow depth of field",
    "a medium framed composition of an ice sculpture bust of <Concept> on a banquet table, exquisitely carved with intricate details, illuminated by soft blue and white LED lights, set against a clean backdrop, surrounded by gentle mist for ethereal effect, realistic style, high contrast lighting, focus on texture and translucency, cold atmosphere, dramatic composition, photography style shot, detailed craftsmanship, atmospheric mood",
    "cinema still portrait of <Concept> as a dark sith lord, ominous red lightsaber glowing, dramatic cinematic lighting, dark and smoky background, intricate black darth vader armor with crimson accents, flowing dark cape, intense and menacing expression, space ship interior background high contrast, chiaroscuro style, epic and dramatic composition, dark sci-fi fantasy atmosphere, highly detailed,hyperrealistic, dark background with red lighting, lots of negative space",
    "<Concept> as a dried-out crusty old mermaid, wrinkled and weathered skin, tangled and brittle seaweed-like hair, smoking a smoldering cigarette underwater with tiny bubbles rising, jagged and cracked tail with faded iridescent scales, adorned with a tarnished coral crown, holding a rusted trident, surrounded by murky greenish water, faint sunlight beams filtering through, floating remnants of debris and sea creatures bowing in the background, regal yet decrepit presence, eerie and otherworldly atmosphere, dark fantasy aesthetic, hyperdetailed, cinematic composition",
    "a hyper-realistic glamour portrait of <Concept> as an old wizard wearing fancy robes and a pointy hat",
    "<Concept> as a simpsons character cartoon simpsons style illustration in the style of the simpsons",
    "<Concept> as a southpark character, a simplistic, cutout animation style with basic geometric shapes, using primary colors, giving the impression of crudely drawn paper cutouts",
]

OBJECT_PROMPTS = [
    "an illustrated black and white fashion sketch full figure drawing of TOK from a vogue magazine cover",
    "an illustrated black and white completely desaturated line drawing fashion sketch 2d isolated full figure drawing of TOK alone single character figure from a vogue magazine cover black and white no color fully desaturated sewing pattern vector drawing ",
    "TOK in the style of Lisa Frank, Rainbow 2D lisa frank trapper keeper cover colorful patterns illustration comic cartoon hard edge style, a dolphin jumping over water with rainbow in background lisa frank",
    "A vibrant, rainbow-colored sticker in the style of Lisa Frank with shiny, heavy glitter accents TOK in the style of Lisa Frank, Rainbow 2D lisa frank trapper keeper cover colorful patterns illustration comic cartoon hard edge style, a dolphin jumping over water with rainbow in background lisa frank",
    "A majestic portrait of TOK as a Roman Emperor in 50 BC, mounted atop a powerful white stallion adorned with ornate golden reins and a deep purple saddle cloth embroidered with imperial symbols. He wears a flowing crimson cape, gleaming gold-plated armor, and a laurel wreath crown upon his head. The magnificent Colosseum towers behind him in pristine condition, its travertine limestone walls glowing warmly in the late afternoon Mediterranean sun. Roman legionnaires stand at attention with their rectangular shields and red cloaks, while senators in togas gather on the steps. ",
    "a depiction of TOK as a lego figure, 3d model plastic lego man, full body framed 3d render showing the full figure, lego hands and feet, standing alone, one character",
    "Image of TOK in the style of iconic Grand Theft Auto V loading screen, leaning against a classic car with sunset over palm trees. Overall image should give impression of professional artwork full of vibrant colors, perfect light and thick outlines, embodying the essence of GTA's thematic imagery.",
    "Classic American traditional tattoo art of the face of TOK as a tiny, mildly infected tattoo on a man's flexing bicep, Bold lines, desaturated faded desaturated classic tattoo art, retro flair, symbolic, tattoo pattern anchor and roses",
    "A marble statue of TOK, completely solid marble, all white with fine marble patterns",
    "A Hand-drawn 80s sci-fi paperback novel cover featuring characters and a futuristic vehicle, prominently featuring TOK as the main protaganist accompanied by a diverse motley crew of exactly three other distinct characters that don't look anything like the lead, featuring an alien, a child, and a femme fatale. The style has a strong classic 1980s aesthetic hand painted acrylic on paper look, with soft mildly desaturated features, kind of like the cover of a paperback novel. The coor palette is dark and mysterious, with lots of negative space and an outerspace star speckled background with bold accents.",
    "A Hand-drawn 80s sci-fi movie poster about an action movie with characters and vehicles, prominently featuring TOK as the main protaganist accompanied by a diverse motley crew of exactly three other distinct characters that don't look anything like the lead, featuring an alien, a child, and a femme fatale. The style has a strong classic 1980s aesthetic hand drawn illustrated look, with soft mildly desaturated features, kind of like the cover of a paperback novel. The coor palette is dark and mysterious, with lots of negative space and an outerspace star speckled background with bold accents",
    "A mysterious figure, TOK in a hoodie stands surrounded by glowing green matrix computer code, 1980s hacker movie aesthetic, mr robot",
    "A masterfully composed artwork depicting TOK standing amid an ancient forest of towering redwoods, dappled golden sunlight filtering through the emerald canopy above. His tribal markings and battle-scarred armor blend with the organic patterns of moss-covered trees, while luminous forest spirits dance ethereally in the misty background, creating an enchanted atmosphere.",
    "A sharp, high resolution photograph of hundreds of people marching down the streets of New York in heavy armor. TOK is walking in front. While everyone else is panicking and running, they are steadily marching down the street, holding their assault rifles pointed down at the ground. Swat teams and FBI agents watch in disbelief as the army takes over the city. Photorealistic action shot, dramatic lighting, shot on Hasselblad H6D-400C.",
    "a portrait of TOK as a futuristic dystopian bond villain, eye patch and prominent scar on the face, dune, slytherin cinematic blade runner cyberpunk lighting high contrast cinema still film grain cruel grin crushed blacks bokeh fleurescent shallow depth of field",
    "a medium framed composition of an ice sculpture bust of TOK on a banquet table, exquisitely carved with intricate details, illuminated by soft blue and white LED lights, set against a clean backdrop, surrounded by gentle mist for ethereal effect, realistic style, high contrast lighting, focus on texture and translucency, cold atmosphere, dramatic composition, photography style shot, detailed craftsmanship, atmospheric mood",
    "cinema still portrait of TOK as a dark sith lord, ominous red lightsaber glowing, dramatic cinematic lighting, dark and smoky background, intricate black darth vader armor with crimson accents, flowing dark cape, intense and menacing expression, space ship interior background high contrast, chiaroscuro style, epic and dramatic composition, dark sci-fi fantasy atmosphere, highly detailed,hyperrealistic, dark background with red lighting, lots of negative space",
    "TOK as a dried-out crusty old mermaid, wrinkled and weathered skin, tangled and brittle seaweed-like hair, smoking a smoldering cigarette underwater with tiny bubbles rising, jagged and cracked tail with faded iridescent scales, adorned with a tarnished coral crown, holding a rusted trident, surrounded by murky greenish water, faint sunlight beams filtering through, floating remnants of debris and sea creatures bowing in the background, regal yet decrepit presence, eerie and otherworldly atmosphere, dark fantasy aesthetic, hyperdetailed, cinematic composition",
    "a hyper-realistic glamour portrait of TOK as an old wizard wearing fancy robes and a pointy hat",
    "TOK as a simpsons character cartoon simpsons style illustration in the style of the simpsons",
    "TOK as a southpark character, a simplistic, cutout animation style with basic geometric shapes, using primary colors, giving the impression of crudely drawn paper cutouts",
]

STYLE_PROMPTS = [
    "A majestic tree rooted in circuits, leaves shimmering with data streams, stands as a beacon where the digital dawn caresses the fog-laden, binary soil—a symphony of pixels and chlorophyll, <Concept>",
    "a luminous white lotus blossom floats on rippling waters, green petals, <Concept>",
    "the all seeing eye made of golden feathers, surrounded by waterfall, photorealistic, ethereal aesthetics, powerful, <Concept>",
    "In the heart of an ancient forest, a massive projection illuminates the darkness. A lone figure, a majestic mythical creature made of shimmering gold, materializes, casting a radiant glow amidst the towering trees. intricate geometric surfaces encasing an expanse of flora and fauna, <Concept>",
    "The Silent Of Silicon, a digital deer rendered in hyper-realistic 3D, eyes glowing in binary code, comfortably resting amidst rich motherboard-green foliage, accented under crisply fluorescent, simulated LED dawn, <Concept>",
    "A solitary tree standing tall amidst a sea of buildings, Urban nature photography, vibrant colors, juxtaposition of natural elements with urban landscapes, play of light and shadow, storytelling through compositions, <Concept>",
    "A labyrinth of mirrored hallways reflecting infinity, with vibrant celestial bodies floating in defiance of gravity, glowing softly with hues of purple and gold, <Concept>",
    "An ancient library with endless shelves, books bound in glowing runes, floating motes of light dancing in the air, all illuminated by a singular beam of moonlight streaming through a cracked dome, <Concept>",
    "A crystalline waterfall cascading in slow motion, refracting a prismatic spectrum of light, surrounded by lush bioluminescent vegetation that pulses rhythmically with a quiet energy, <Concept>",
    "A futuristic cityscape where skyscrapers are crafted from transparent glass and gold filigree, their shapes resembling abstract sculptures, the entire skyline illuminated by a crimson sun sinking into a sea of mist, <Concept>",
    "A lone figure standing on a massive golden disc suspended above a stormy ocean, tendrils of lightning reaching up from the waves to the heavens, a glowing portal pulsating in the sky, <Concept>",
    "An alien desert under a violet sky with three suns, the sand swirling with iridescent colors, strange crystalline formations scattered throughout, and a massive obelisk towering in the distance, <Concept>",
    "A colossal machine resembling a giant clockwork cathedral, its gears and cogs intricately detailed and encrusted with emerald moss, with beams of light piercing through the machinery, <Concept>",
    "A serene valley filled with floating islands, their surfaces covered in emerald-green forests and waterfalls cascading into the clouds below, illuminated by soft golden light, <Concept>",
    "A hauntingly beautiful ice cavern with jagged stalactites, illuminated by an ethereal blue glow emanating from frozen creatures embedded in the walls, <Concept>",
    "An enormous, intricately-carved tree that reaches into the clouds, its branches holding luminous orbs, glowing softly in the twilight, while mythical creatures glide between the boughs, <Concept>",
    "A futuristic botanical garden filled with plants made of glass and metal, their leaves softly glowing, with mechanical birds flying through artificial waterfalls, <Concept>",
    "A hidden underwater city with domed buildings made of coral and glass, illuminated by bioluminescent sea life, with rays of sunlight piercing through the deep blue ocean above, <Concept>",
    "An ancient temple suspended in the air, surrounded by swirling clouds, its walls inscribed with glowing hieroglyphs, and a golden beam of light connecting it to the earth below, <Concept>",
    "A sprawling digital network visualized as a glowing forest, with data flowing like streams of light between luminescent, crystalline trees, and holographic animals darting through the landscape, <Concept>",
    "A cosmic coliseum where the stars form the walls, with rings of orbiting planets circling above, and the arena floor a swirling nebula of colors and energy, <Concept>",
]

ALL_PROMPTS = FACE_PROMPTS + OBJECT_PROMPTS + STYLE_PROMPTS


# async def run_twitter_replybots(start_time=None):
#     # Get all Twitter deployments
#     deployments = Deployment.find({"platform": "twitter"})

#     for deployment in deployments:
#         try:
#             # Get the agent for this deployment
#             agent = Agent.from_mongo(deployment.agent)

#             # Initialize Twitter client
#             twitter = X(agent)

#             # Fetch mentions
#             mentions = twitter.fetch_mentions(start_time=start_time)
#             if not mentions.get("data"):
#                 continue

#             # Get newest tweet from each unique user
#             user_tweets = {}
#             for tweet in mentions.get("data", []):
#                 author_id = tweet["author_id"]
#                 if author_id not in user_tweets:
#                     user_tweets[author_id] = tweet

#             # Reply to each user's newest tweet
#             for tweet in user_tweets.values():
#                 try:
#                     twitter.post(
#                         tweet_text=f"👋 Hello! This is a placeholder reply from {agent.name}.",
#                         reply_to_tweet_id=tweet["id"],
#                     )
#                 except Exception as e:
#                     print(f"Error replying to tweet {tweet['id']}: {e}")
#                     sentry_sdk.capture_exception(e)

#         except Exception as e:
#             print(f"Error processing deployment {deployment.id}: {e}")
#             sentry_sdk.capture_exception(e)


async def run_twitter_automation():
    """
    Periodically generate tweets for Twitter deployments via chat endpoint
    """

    deployments = Deployment.find({"platform": "twitter"})
    api_url = os.getenv("EDEN_API_URL")

    async with aiohttp.ClientSession() as session:
        for deployment in deployments:
            try:
                agent = Agent.load(deployment.agent)

                # Create update config for the chat request
                update_config = UpdateConfig(
                    update_endpoint=f"{api_url}/updates/platform/twitter",
                    deployment_id=str(deployment.id),
                    twitter_tweet_to_reply_id=None,
                )

                # Create chat request
                chat_request = ChatRequest(
                    user_id=str(agent.id),
                    agent_id=str(agent.id),
                    user_message={
                        "content": "Please generate a tweet for my Twitter audience."
                    },
                    update_config=update_config,
                    force_reply=True,
                )

                # Post to chat endpoint
                async with session.post(
                    f"{api_url}/chat",
                    json=chat_request.model_dump(),
                    headers={"Authorization": f"Bearer {os.getenv('EDEN_API_KEY')}"},
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Chat request failed: {error_text}")

            except Exception as e:
                print(f"Error processing deployment {deployment.id}: {e}")
                sentry_sdk.capture_exception(e)


async def rotate_agent_metadata(since_hours=6):
    """
    Rotate agent suggestions, greetings, and knowledge descriptions for agents whose updatedAt is younger than 6 hours or null (new agents)
    """

    agents = get_collection(Agent.collection_name)

    filter = {}
    if since_hours:
        filter["type"] = "agent"
        filter["$or"] = [
            {"updatedAt": None},
            {
                "updatedAt": {
                    "$gt": datetime.now(timezone.utc) - timedelta(hours=since_hours)
                }
            },
        ]

    for agent in agents.find(filter):
        print(agent)
        agent = Agent(**agent)
        print("rotate", agent.username)
        print(type(agent))
        print(agent)
        updated_at = (agent.updatedAt or agent.createdAt).replace(tzinfo=timezone.utc)
        refreshed_at = agent.refreshed_at
        if refreshed_at:
            refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)

        if refreshed_at and (updated_at - refreshed_at).total_seconds() <= 0:
            continue

        try:
            await refresh_agent(agent)
        except Exception as e:
            print(f"Error refreshing agent {agent.username}: {e}")
            sentry_sdk.capture_exception(e)
