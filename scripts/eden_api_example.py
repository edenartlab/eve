import json
import os
import time

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
EDEN_API_KEY = os.environ.get("EDEN_API_KEY")


config = {
    "interpolation_texts": [
        "Drawing by M. C. Escher with a strong solar-punk flavor representing: A scene of vibrant greens and blues, a garden where life thrives,A butterfly's delicate flutter, its energy vividly alive.A visual symphony, under the sun's watchful eye,Nature's strength and beauty reign, its power we can't deny. Hyper realistic, detailed, intricate, best quality, hyper detailed, ultra realistic, sharp focus, delicate and refined.",
        "Drawing by M. C. Escher with a strong solar-punk flavor representing: A garden filled with lush green foliage, a butterfly gracefully hovers above. Hyper realistic, detailed, intricate, best quality, hyper detailed, ultra realistic, sharp focus, delicate and refined.",
        "Drawing by M. C. Escher with a strong solar-punk flavor representing: Its wings beat with vitality, carrying life and vigor high up into the sky. Hyper realistic, detailed, intricate, best quality, hyper detailed, ultra realistic, sharp focus, delicate and refined.",
        "Drawing by M. C. Escher with a strong solar-punk flavor representing: The delicate creature reminds us of the vastness of existence, as it flutters with vibrance. Hyper realistic, detailed, intricate, best quality, hyper detailed, ultra realistic, sharp focus, delicate and refined.",
        "Drawing by M. C. Escher with a strong solar-punk flavor representing: A display of vibrant colors dances beneath the warm rays of the sun, showcasing nature's unrivaled strength and resilience. Hyper realistic, detailed, intricate, best quality, hyper detailed, ultra realistic, sharp focus, delicate and refined.",
        "Drawing by M. C. Escher with a strong solar-punk flavor representing: This butterfly is a symbol of hope, embodying our own aspirations and ambitions. Hyper realistic, detailed, intricate, best quality, hyper detailed, ultra realistic, sharp focus, delicate and refined.",
        "Drawing by M. C. Escher with a strong solar-punk flavor representing: Let us take inspiration from its flight, and strive to capture the boundless energy and vitality it possesses. Hyper realistic, detailed, intricate, best quality, hyper detailed, ultra realistic, sharp focus, delicate and refined.",
        "Drawing by M. C. Escher with a strong solar-punk flavor representing: The greens and blues of the Earth and sky hold endless possibilities, waiting for us to seize. Hyper realistic, detailed, intricate, best quality, hyper detailed, ultra realistic, sharp focus, delicate and refined.",
    ],
    "interpolation_init_images": [
        "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/44050c3ab6e427ca6fa851f1a66cfe7dcacd996818d05bd09395f1e3790ad91c.jpg",
        "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/44050c3ab6e427ca6fa851f1a66cfe7dcacd996818d05bd09395f1e3790ad91c.jpg",
        "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/44050c3ab6e427ca6fa851f1a66cfe7dcacd996818d05bd09395f1e3790ad91c.jpg",
        "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/44050c3ab6e427ca6fa851f1a66cfe7dcacd996818d05bd09395f1e3790ad91c.jpg",
        "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/44050c3ab6e427ca6fa851f1a66cfe7dcacd996818d05bd09395f1e3790ad91c.jpg",
        "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/44050c3ab6e427ca6fa851f1a66cfe7dcacd996818d05bd09395f1e3790ad91c.jpg",
        "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/44050c3ab6e427ca6fa851f1a66cfe7dcacd996818d05bd09395f1e3790ad91c.jpg",
        "https://edenartlab-prod-data.s3.us-east-1.amazonaws.com/44050c3ab6e427ca6fa851f1a66cfe7dcacd996818d05bd09395f1e3790ad91c.jpg",
    ],
    "interpolation_init_images_min_strength": 0,
    "interpolation_init_images_max_strength": 0,
    "interpolation_init_images_power": 0.5,
    "n_film": 2,
    "latent_blending_skip_f": [0.1, 0.9],
    "guidance_scale": 20,
    "width": 1024,
    "height": 1024,
    "stream": False,
    "steps": 20,
    "fps": 7,
    "n_frames": 73,
}


def run_eden_task(config):
    request = {"tool": "real2real", "args": config}
    response = requests.post(
        "https://api.eden.art/v2/tasks/create",
        json=request,
        headers={"x-api-key": EDEN_API_KEY},
    )
    result = response.json()

    if response.status_code == 500:
        error = result.get("error", "Unknown error")
        raise Exception("Error submitting task: " + error.get("detail"))
    if response.status_code != 200:
        raise Exception("Error submitting task: " + response.text)

    taskId = result["task"]["_id"]
    print(f"Submitted task: {taskId}")

    status_bar = tqdm(desc="Task status", leave=False)
    while True:
        response = requests.get(
            "https://api.eden.art/v2/tasks/" + taskId,
            headers={"x-api-key": EDEN_API_KEY},
        )
        result = response.json()
        with open(f"Task_{taskId}.json", "w") as f:
            json.dump(result, f, indent=4)
        status = result["task"]["status"]
        if status == "completed":
            output_file = result["task"]["result"][0]["output"][0]["url"]
            print(f"✅ => {output_file}")
            break
        elif status == "failed" or status == "cancelled":
            raise Exception("❌ Task failed: " + result["task"]["error"])
        elif status == "running":
            status_bar.set_description("🏃 Task running")
        elif status == "pending":
            status_bar.set_description("⏳ Task pending")
        time.sleep(10)


if __name__ == "__main__":
    run_eden_task(config)
