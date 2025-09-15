import base64
import os
import openai


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


async def handler(args: dict, user: str = None, agent: str = None):
    """
    Handles the image generation request using the OpenAI API (gpt-image-1 only).
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = openai.AsyncOpenAI(api_key=api_key)

    # Prepare arguments for the OpenAI API call
    # Filter out any None values before sending
    valid_args = {k: v for k, v in args.items() if v is not None}

    # Hardcode some params:
    valid_args["model"] = "gpt-image-1"
    valid_args["moderation"] = "low" 

    if user and 'user' not in valid_args:
         valid_args['user'] = str(user)

    # Rename n_samples to n for OpenAI API
    if "n_samples" in valid_args:
        valid_args["n"] = valid_args.pop("n_samples")

    # OpenAI got rid of these params?
    valid_args.pop("background", None)
    valid_args.pop("output_format", None)
    valid_args.pop("output_compression", None)

    if "quality" in valid_args:
        valid_args["quality"] = args["quality"]

    # if valid_args['background'] == 'transparent':
    #     valid_args['output_format'] = 'png'

    # if valid_args['output_format'] == 'png':
    #     valid_args['output_compression'] = 100

    valid_args.pop("agent", None)

    try:
        print(f"Calling OpenAI Images API (gpt-image-1) with args: {valid_args}")
        response = await client.images.generate(**valid_args)

        output = []
        for i, item in enumerate(response.data):
            image_bytes = base64.b64decode(item.b64_json)
            temp_file_name = f"image_{i}.jpg"
            with open(temp_file_name, "wb") as f:
                f.write(image_bytes)
            output.append(temp_file_name)

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

        return error_result
