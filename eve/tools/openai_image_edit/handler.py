import base64
import os
import tempfile
import openai
from PIL import Image
from io import BytesIO

from eve.utils import download_file, PIL_to_bytes

# Maximum dimension (width or height) for resizing
MAX_DIMENSION = 2048

SAFETY_CODES = {"moderation_blocked", "content_policy_violation"}


def classify_openai_error(e: Exception):
    """
    Returns (category, info) where category is one of:
      - "safety"
      - "bad_request"
      - "openai_internal"
      - "other_openai_error"
      - "client_runtime"
    Keeps info minimal (status, request_id, code, message).
    """
    # Defaults if the SDK exception doesn't have these
    status = getattr(e, "status_code", None)
    request_id = getattr(e, "request_id", None)
    message = str(e)
    code = None
    safety_violations = None

    # Try to read the standard error envelope (best-effort, no extra dependencies)
    try:
        resp = getattr(e, "response", None)
        if resp is not None:
            err = (resp.json() or {}).get("error", {})  # robust to missing body
            code = err.get("code") or code
            message = err.get("message", message)
            safety_violations = err.get("safety_violations")
    except Exception:
        pass  # never let error parsing throw

    # --- Lanes ---
    if isinstance(e, openai.BadRequestError):
        # Safety lane: common cases include code="moderation_blocked"
        if (code in SAFETY_CODES) or safety_violations or ("safety" in (message or "").lower()):
            return "safety", {"status": status, "request_id": request_id, "code": code, "message": message}
        # Other 4xx are your inputs/params
        return "bad_request", {"status": status, "request_id": request_id, "code": code, "message": message}

    if isinstance(e, openai.InternalServerError) or (isinstance(e, openai.APIError) and status and 500 <= status < 600):
        return "openai_internal", {"status": status, "request_id": request_id, "message": message}

    # Nice to keep these around for telemetry/retry logic
    if isinstance(e, openai.RateLimitError):
        return "other_openai_error", {"status": status, "request_id": request_id, "subtype": "rate_limit", "message": message}
    if isinstance(e, openai.APITimeoutError) or isinstance(e, openai.APIConnectionError):
        return "other_openai_error", {"status": status, "request_id": request_id, "subtype": "network", "message": message}
    if isinstance(e, openai.APIError):
        return "other_openai_error", {"status": status, "request_id": request_id, "message": message}

    # Anything else is likely your code/IO/etc.
    return "client_runtime", {"message": message}


class BytesIOWithName(BytesIO):
    """A BytesIO subclass that has a name attribute for MIME type detection."""
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name

def preprocess_image(file_input, is_mask=False):
    """
    Downloads if URL, resizes image, converts to WEBP bytes.
    Returns the processed image bytes.
    """
    temp_file_path = None
    
    try:
        if isinstance(file_input, str) and file_input.startswith(("http://", "https://")):
            print(f"Input is URL, downloading: {file_input}")
            # Create a temporary file to store the download
            file_type = file_input.split(".")[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
                temp_file_path = temp_file.name
            # Download the file
            download_file(file_input, temp_file_path, overwrite=True)
            image_path = temp_file_path
            print(f"Downloaded to temporary file: {image_path}")
        elif isinstance(file_input, str) and os.path.exists(file_input):
            image_path = file_input
            print(f"Input is local file: {image_path}")
        else:
            raise ValueError(f"Invalid image input type or path: {file_input}")

        # Process the image
        print(f"Processing image: {image_path}")
        img = Image.open(image_path)
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

        if is_mask:
            print("Converting mask to RGBA")
            img = img.convert('RGBA')
        
        print("Converting to WEBP format")
        img_bytes = PIL_to_bytes(img, ext='WEBP', quality=90)
        print(f"Finished processing, final size: {len(img_bytes)} bytes")
        return img_bytes

    except FileNotFoundError as e:
        print(f"Error: Input file not found during preprocessing: {e}")
        raise ValueError(f"File not found: {getattr(e, 'filename', file_input)}") from e
    except Exception as e:
        print(f"Error during image preprocessing for {file_input}: {e}")
        raise e
    finally:
        # Clean up the temporary file if one was created
        if temp_file_path and os.path.exists(temp_file_path):
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

    client = openai.OpenAI(api_key=api_key)

    # Extract and validate required parameters
    image_paths = args.get('image', [])
    mask_path = args.get('mask')
    prompt = args.get('prompt')
    
    if not image_paths:
        raise ValueError("The 'image' parameter (list of image file paths) is required.")
    
    if not prompt:
        raise ValueError("The 'prompt' parameter is required.")

    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    try:
        # Process the first image (OpenAI edit API only supports one image)
        print("Processing main image...")
        image_bytes = preprocess_image(image_paths[0], is_mask=False)
        
        # Process mask if provided
        mask_bytes = None
        if mask_path:
            print("Processing mask...")
            mask_bytes = preprocess_image(mask_path, is_mask=True)

        # Prepare API call parameters
        api_params = {
            "model": "gpt-image-1",
            "image": BytesIOWithName(image_bytes, "image.webp"),
            "prompt": prompt,
            # "moderation": "low"
        }
        
        if mask_bytes:
            api_params["mask"] = BytesIOWithName(mask_bytes, "mask.webp")

        # Add optional parameters
        if "n" in args:
            api_params["n"] = args["n"]
        elif "n_samples" in args:
            api_params["n"] = args["n_samples"]
            
        if "size" in args:
            api_params["size"] = args["size"]
            
        if "input_fidelity" in args:
            api_params["input_fidelity"] = args["input_fidelity"]
            
        # if "output_compression" in args:
        #     api_params["output_compression"] = args["output_compression"]
            
        # if "output_format" in args:
        #     api_params["output_format"] = args["output_format"]
            
        if user:
            api_params["user"] = str(user)

        # Make the API call
        print("Making OpenAI API call...")
        response = client.images.edit(**api_params)

        # Process the response
        output = []
        for i, item in enumerate(response.data):
            image_bytes = base64.b64decode(item.b64_json)
            output_filename = f"edited_image_{i}.png"
            with open(output_filename, "wb") as f:
                f.write(image_bytes)
            output.append(output_filename)
            print(f"Saved edited image: {output_filename}")

        return {"output": output}

    except Exception as e:
        category, info = classify_openai_error(e)

        if category == "safety":
            # TODO: your safety-specific handling
            error_result = {"error": {"category": category, **info}}

        elif category == "openai_internal":
            # TODO: your retry/backoff/circuit-breaker path
            error_result = {"error": {"category": category, **info}}

        elif category == "bad_request":
            # TODO: your parameter-fix path
            error_result = {"error": {"category": category, **info}}

        else:
            # Optional: log/telemetry for other OpenAI or client runtime issues
            error_result = {"error": {"category": category, **info}}

        
        print("!!!! ================ ERROR ==================== !!!!")
        print("OpenAI error result: ", error_result)
        print("!!!! ================ ERROR ==================== !!!!")

        raise e
