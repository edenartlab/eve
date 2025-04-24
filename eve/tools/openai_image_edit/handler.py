import base64
import os
from openai import AsyncOpenAI
import asyncio
import tempfile
from io import BytesIO
from PIL import Image

from eve.eden_utils import download_file, PIL_to_bytes

# Maximum dimension (width or height) for resizing
MAX_DIMENSION = 2048

async def preprocess_image(file_input, is_mask=False):
    """
    Downloads if URL, resizes image, converts to WEBP bytes.
    Cleans up temporary files.
    """
    is_temp = False
    temp_file_path = None

    try:
        if isinstance(file_input, str) and file_input.startswith(("http://", "https://")):
            print(f"Input is URL, downloading: {file_input}")
            # Create a temporary file to store the download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
                temp_file_path = temp_file.name
            # Run synchronous download in a thread
            await asyncio.to_thread(download_file, file_input, temp_file_path, overwrite=True)
            image_path = temp_file_path
            is_temp = True
            print(f"Downloaded to temporary file: {image_path}")
        elif isinstance(file_input, str) and os.path.exists(file_input):
            image_path = file_input
            print(f"Input is local file: {image_path}")
        else:
            raise ValueError(f"Invalid image input type or path: {file_input}")

        # Define synchronous processing function to run in thread
        def _process_sync(path, mask_flag):
            print(f"Processing image: {path}")
            img = Image.open(path)
            w, h = img.size
            
            # Calculate new size
            if max(w, h) > MAX_DIMENSION:
                ratio = MAX_DIMENSION / max(w, h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                print(f"Resizing from {w}x{h} to {new_w}x{new_h}")
                img.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                 print(f"Image size {w}x{h} is within limits, no resize needed.")

            if mask_flag:
                print("Converting mask to RGBA")
                img = img.convert('RGBA')
            
            print("Converting to WEBP format")
            img_bytes = PIL_to_bytes(img, ext='WEBP', quality=90) # Use WEBP
            print(f"Finished processing, final size: {len(img_bytes)} bytes")
            return img_bytes

        # Run synchronous PIL processing in a thread
        processed_bytes = await asyncio.to_thread(_process_sync, image_path, is_mask)
        return processed_bytes

    except FileNotFoundError as e:
        print(f"Error: Input file not found during preprocessing: {e}")
        raise ValueError(f"File not found: {getattr(e, 'filename', file_input)}") from e
    except Exception as e:
        print(f"Error during image preprocessing for {file_input}: {e}")
        raise e # Re-raise the exception after logging
    finally:
        # Clean up the temporary file if one was created
        if is_temp and temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except OSError as e:
                print(f"Error cleaning up temporary file {temp_file_path}: {e}")

async def handler(args: dict, user: str = None, agent: str = None):
    """
    Handles the image editing request using the OpenAI API (gpt-image-1 only).
    Supports single or multiple input images.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = AsyncOpenAI(api_key=api_key)

    valid_args = {k: v for k, v in args.items() if v is not None}
    valid_args["model"] = "gpt-image-1" # Explicitly set the model

    if user and 'user' not in valid_args:
         valid_args['user'] = str(user)

    image_paths = valid_args.pop('image', []) # Expecting a list of paths
    mask_path = valid_args.pop('mask', None)

    if not image_paths:
         raise ValueError("The 'image' parameter (list of image file paths) is required.")

    # Ensure image_paths is a list, even if only one is provided via API YAML conversion
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    try:
        # Prepare preprocessing tasks for all files (images and potentially mask)
        print("Starting image preprocessing...")
        image_preprocess_tasks = [preprocess_image(path, is_mask=False) for path in image_paths]
        
        all_preprocess_tasks = list(image_preprocess_tasks)
        mask_preprocess_task = None
        if mask_path:
            print(f"Mask path provided: {mask_path}")
            mask_preprocess_task = preprocess_image(mask_path, is_mask=True)
            all_preprocess_tasks.append(mask_preprocess_task)

        # Gather all preprocessed file contents (bytes) concurrently
        all_processed_bytes = await asyncio.gather(*all_preprocess_tasks)

        # Assign preprocessed image bytes
        image_contents = all_processed_bytes[:len(image_preprocess_tasks)]
        valid_args['image'] = image_contents

        # Assign preprocessed mask bytes if it was processed
        if mask_preprocess_task:
            mask_content = all_processed_bytes[-1] # Mask content is the last one if it exists
            valid_args['mask'] = mask_content

        response = await client.images.edit(**valid_args)

        output = []
        for i, item in enumerate(response.data):
            image_bytes = base64.b64decode(item.b64_json)
            temp_file_name = f"image_{i}.jpg"
            with open(temp_file_name, "wb") as f:
                f.write(image_bytes)
            output.append(temp_file_name)

        return {"output": output}

    except Exception as e:
        print(f"Error in handler: {e}") # Catch errors from preprocessing or API call
        raise e 