import os


def get_api_url() -> str:
    """
    Get the appropriate API URL based on environment variables.

    If LOCAL_API_URL is set and not empty, use it.
    Otherwise, fall back to EDEN_API_URL.

    Returns:
        str: The API URL to use for deployment requests
    """
    local_api_url = os.getenv("LOCAL_API_URL")
    if local_api_url and local_api_url != "":
        return local_api_url
    return os.getenv("EDEN_API_URL")
