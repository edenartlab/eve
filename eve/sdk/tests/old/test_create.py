from eden.client import EdenClient
from loguru import logger

client = EdenClient(stage=True)

# result = client.create("audiocraft", {
#     "text_input": "futuristic synth sounds",
# })

# result = client.create("txt2img", {
#     "prompt": "<concept> on a disco floor",
#     "lora": "66c2bf435ec7b6d95deb7223",
#     "lora_strength": 0.3,
#     "width": 1280,
#     "height": 768,
# })

# result = client.create("stable_audio", {
#     "prompt": "drum and bass",
# })

result = client.create("flux_dev", {
    "prompt": "Alien forest with bioluminescent flora.",
    "lora_strength": 0.3,
    "seed": 19,
    "n_samples": 4,
})


logger.info(result)
