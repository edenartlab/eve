import base64
import json
import os
import tempfile
from io import BytesIO
from urllib.parse import urlparse

import openai
from PIL import Image

from eve.tool import ToolContext
from eve.utils import download_file

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
        if (
            (code in SAFETY_CODES)
            or safety_violations
            or ("safety" in (message or "").lower())
        ):
            return "safety", {
                "status": status,
                "request_id": request_id,
                "code": code,
                "message": message,
            }
        # Other 4xx are your inputs/params
        return "bad_request", {
            "status": status,
            "request_id": request_id,
            "code": code,
            "message": message,
        }

    if isinstance(e, openai.InternalServerError) or (
        isinstance(e, openai.APIError) and status and 500 <= status < 600
    ):
        return "openai_internal", {
            "status": status,
            "request_id": request_id,
            "message": message,
        }

    # Nice to keep these around for telemetry/retry logic
    if isinstance(e, openai.RateLimitError):
        return "other_openai_error", {
            "status": status,
            "request_id": request_id,
            "subtype": "rate_limit",
            "message": message,
        }
    if isinstance(e, openai.APITimeoutError) or isinstance(
        e, openai.APIConnectionError
    ):
        return "other_openai_error", {
            "status": status,
            "request_id": request_id,
            "subtype": "network",
            "message": message,
        }
    if isinstance(e, openai.APIError):
        return "other_openai_error", {
            "status": status,
            "request_id": request_id,
            "message": message,
        }

    # Anything else is likely your code/IO/etc.
    return "client_runtime", {"message": message}


def preprocess_image(file_input, is_mask=False):
    """
    Downloads if URL, resizes image, converts to WEBP (PNG for masks),
    and returns a tuple of (image_bytes, filename, mime_type).
    """
    temp_file_path = None

    try:
        if isinstance(file_input, str) and file_input.startswith(
            ("http://", "https://")
        ):
            # Create a temporary file to store the download
            parsed_url = urlparse(file_input)
            path = parsed_url.path or ""
            _, ext = os.path.splitext(path)
            file_type = ext.lstrip(".") or "tmp"
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_type}"
            ) as temp_file:
                temp_file_path = temp_file.name
            # Download the file
            download_file(file_input, temp_file_path, overwrite=True)
            image_path = temp_file_path
        elif isinstance(file_input, str) and os.path.exists(file_input):
            image_path = file_input
        else:
            raise ValueError(f"Invalid image input type or path: {file_input}")

        # Process the image
        with Image.open(image_path) as img:
            w, h = img.size

            # Calculate new size
            if max(w, h) > MAX_DIMENSION:
                ratio = MAX_DIMENSION / max(w, h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                img.thumbnail((new_w, new_h), Image.Resampling.LANCZOS)

            if is_mask:
                img = img.convert("RGBA")
            else:
                if "A" in img.getbands():
                    img = img.convert("RGBA")
                elif img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

            buffer = BytesIO()
            if is_mask:
                target_format = "PNG"
                filename = "mask.png"
                mime_type = "image/png"
                save_kwargs = {}
            else:
                target_format = "WEBP"
                filename = "image.webp"
                mime_type = "image/webp"
                save_kwargs = {"quality": 90}

            img.save(buffer, format=target_format, **save_kwargs)
            buffer.seek(0)
            img_bytes = buffer.read()

        return img_bytes, filename, mime_type

    except FileNotFoundError as e:
        raise ValueError(f"File not found: {getattr(e, 'filename', file_input)}") from e
    except Exception as e:
        raise e
    finally:
        # Clean up the temporary file if one was created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                raise e


async def handler(context: ToolContext):
    """
    Handles the image editing request using the OpenAI API (gpt-image-1 only).
    Supports single or multiple input images.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = openai.OpenAI(api_key=api_key)

    # Extract and validate required parameters
    image_paths = context.args.get("image", [])
    mask_path = context.args.get("mask")
    prompt = context.args.get("prompt")

    if not image_paths:
        raise ValueError(
            "The 'image' parameter (list of image file paths) is required."
        )

    if not prompt:
        raise ValueError("The 'prompt' parameter is required.")

    # Ensure image_paths is a list
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    try:
        # Process the first image (OpenAI edit API only supports one image)
        image_bytes, image_filename, image_mime = preprocess_image(
            image_paths[0], is_mask=False
        )

        # Process mask if provided
        mask_bytes = None
        if mask_path:
            mask_bytes, mask_filename, mask_mime = preprocess_image(
                mask_path, is_mask=True
            )

        # Prepare API call parameters
        api_params = {
            "model": "gpt-image-1",
            "image": (image_filename, image_bytes, image_mime),
            "prompt": prompt,
            # "moderation": "low"
        }

        if mask_bytes:
            api_params["mask"] = (mask_filename, mask_bytes, mask_mime)

        # Add optional parameters
        if "n" in context.args:
            api_params["n"] = context.args["n"]
        elif "n_samples" in context.args:
            api_params["n"] = context.args["n_samples"]

        if "size" in context.args:
            api_params["size"] = context.args["size"]

        if "input_fidelity" in context.args:
            api_params["input_fidelity"] = context.args["input_fidelity"]

        # if "output_compression" in args:
        #     api_params["output_compression"] = context.args["output_compression"]

        # if "output_format" in args:
        #     api_params["output_format"] = context.args["output_format"]

        if context.user:
            api_params["user"] = str(context.user)

        # Make the API call
        response = client.images.edit(**api_params)

        # Process the response
        output = []
        for i, item in enumerate(response.data):
            image_bytes = base64.b64decode(item.b64_json)
            output_filename = f"edited_image_{i}.png"
            with open(output_filename, "wb") as f:
                f.write(image_bytes)
            output.append(output_filename)

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

        error_message = error_result.get("error").get("message")
        if not error_message:
            error_message = json.dumps(error_result)

        raise Exception(error_message)
