import os
import requests
import asyncio
import tempfile
from ... import eden_utils


async def handler(args: dict, user: str = None, agent: str = None):
    HEDRA_API_KEY = os.getenv("HEDRA_API_KEY")
    HEDRA_BASE_URL = "https://mercury.dev.dream-ai.com/api"

    # Create temp files with appropriate extensions
    temp_image = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_audio = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    
    try:
        # Download files using eden_utils
        image_path = eden_utils.download_file(args["image"], temp_image.name, overwrite=True)
        audio_path = eden_utils.download_file(args["audio"], temp_audio.name, overwrite=True)
        
        # Upload audio file
        audio_response = requests.post(
            f"{HEDRA_BASE_URL}/v1/audio", 
            headers={'X-API-KEY': HEDRA_API_KEY}, 
            files={'file': open(audio_path, 'rb')}
        )
        if not audio_response.ok:
            raise Exception(f"Failed to upload audio: {audio_response.text}")
        
        # Upload image file
        image_response = requests.post(
            f"{HEDRA_BASE_URL}/v1/portrait", 
            headers={'X-API-KEY': HEDRA_API_KEY}, 
            files={'file': open(image_path, 'rb')}
        )
        if not image_response.ok:
            raise Exception(f"Failed to upload image: {image_response.text}")

        # print({
        #     "avatarImage": image_response.json()["url"],
        #     "audioSource": "audio",
        #     "voiceUrl": audio_response.json()["url"],
        #     "aspectRatio": args["aspectRatio"]
        # })

        # Do portrait
        video_response = requests.post(
            f"{HEDRA_BASE_URL}/v1/characters", 
            headers={'X-API-KEY': HEDRA_API_KEY}, 
            json={
                "avatarImage": image_response.json()["url"],
                "audioSource": "audio",
                "voiceUrl": audio_response.json()["url"],
                "aspectRatio": args["aspectRatio"]
            }
        )
        if not video_response.ok:
            raise Exception(f"Failed to create character: {video_response.text}")

        print(video_response.json())
        project_id = video_response.json()["jobId"]
        
        # Poll for completion
        while True:
            project_status = requests.get(
                f"{HEDRA_BASE_URL}/v1/projects/{project_id}", 
                headers={'X-API-KEY': HEDRA_API_KEY}
            )
            
            if not project_status.ok:
                raise Exception(f"Failed to get project status: {project_status.text}")

            status = project_status.json()["status"]
            
            if status == "Completed":
                return {
                    "output": project_status.json()["videoUrl"]
                }
            elif status == "Failed" or status == "Cancelled" or status == "Error":
                raise Exception(f"Project failed: {project_status.json().get('error', 'Unknown error')}")
            elif status == "InProgress":
                print("Project in progress")
            else:
                print(f"Project status: {status}")
            
            await asyncio.sleep(10)
            
    finally:
        os.unlink(temp_image.name)
        os.unlink(temp_audio.name)
