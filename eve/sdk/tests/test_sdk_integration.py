import os
import pytest
from dotenv import load_dotenv
from eve.llm import UserMessage
from eve.sdk.eden.client import EdenClient, EdenApiUrls
import httpx

# Load environment variables from root .env
load_dotenv()

# Configure API URLs based on environment
EDEN_API_URL = os.getenv("EDEN_API_URL", "http://localhost:5050")
EDEN_TOOLS_API_URL = os.getenv("EDEN_TOOLS_API_URL", "http://localhost:8000")
EDEN_API_KEY = os.getenv("EDEN_API_KEY")


@pytest.fixture
def api_urls():
    return EdenApiUrls(api_url=EDEN_API_URL, tools_api_url=EDEN_TOOLS_API_URL)


@pytest.fixture
def client(api_urls):
    return EdenClient(api_urls=api_urls)


@pytest.fixture(autouse=True)
def increase_timeout():
    # Temporarily increase default timeout for all httpx clients during tests
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = 120.0
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = 120.0
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = 120.0
    yield
    # Reset to defaults after tests
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = 5.0
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = 5.0
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = 5.0


# @pytest.mark.integration
# @pytest.mark.asyncio
# async def test_async_create_image(api_urls):
#     async with EdenClient(api_urls=api_urls) as client:
#         result = await client.async_create(
#             workflow="txt2img",
#             args={
#                 "prompt": "A beautiful sunset over mountains",
#             },
#         )
#         print(result)
#         assert "image_url" in result
#         assert result["image_url"].startswith("https://")


# @pytest.mark.integration
# def test_sync_create_image(client):
#     result = client.create(
#         workflow="musicgen",
#         args={
#             "prompt": "drum n bass",
#             "duration": 1,
#         },
#     )
#     print(result)
#     assert result is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_chat(api_urls):
    async with EdenClient(api_urls=api_urls) as client:
        updates = []
        message = "Tell me about Eden's video generation capabilities"
        async for update in client.async_chat(
            user_message=message,
            thread_name="test_thread",
        ):
            updates.append(update)
            # Verify update structure
            assert isinstance(update, dict)

        assert len(updates) > 0


@pytest.mark.integration
def test_sync_chat(client):
    updates = client.chat(
        user_message="Tell me about Eden's video generation capabilities",
        thread_name="test_thread",
    )

    assert len(updates) > 0


# @pytest.mark.integration
# def test_custom_api_urls():
#     custom_urls = EdenApiUrls(
#         api_url="https://custom-api.eden.art",
#         tools_api_url="https://custom-tools.eden.art",
#     )
#     client = EdenClient(api_urls=custom_urls)
#     assert client.api_urls.api_url == "https://custom-api.eden.art"
#     assert client.api_urls.tools_api_url == "https://custom-tools.eden.art"


# @pytest.mark.integration
# def test_default_api_urls(client):
#     assert client.api_urls.api_url == EDEN_API_URL
#     assert client.api_urls.tools_api_url == EDEN_TOOLS_API_URL


# @pytest.mark.integration
# @pytest.mark.asyncio
# async def test_error_handling(api_urls):
#     async with EdenClient(api_urls=api_urls) as client:
#         with pytest.raises(Exception):
#             await client.async_create(workflow="invalid/workflow", args={})


# @pytest.mark.integration
# def test_invalid_api_key():
#     with pytest.raises(Exception):
#         EdenClient(api_key="invalid_key")
