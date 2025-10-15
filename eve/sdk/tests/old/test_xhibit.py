import json
import os
import asyncio
import httpx
from pydantic import SecretStr
from dataclasses import dataclass
from typing import Optional, AsyncGenerator, List


@dataclass
class EdenApiUrls:
    api_url: str
    tools_api_url: str


EDEN_API_URL = "https://api.eden.art"
EDEN_TOOLS_API_URL = "https://edenartlab--tools-fastapi-app.modal.run"


class EdenClient:
    def __init__(
        self,
        api_urls: EdenApiUrls = EdenApiUrls(
            api_url=EDEN_API_URL, tools_api_url=EDEN_TOOLS_API_URL
        ),
    ):
        self.api_urls = api_urls
        self.api_key = get_api_key()
        self._async_client = httpx.AsyncClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._async_client.aclose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self._async_client.aclose())

    async def _subscribe(self, task_id):
        url = f"{self.api_urls.api_url}/v2/tasks/events?taskId={task_id}"
        headers = {"X-Api-Key": self.api_key.get_secret_value()}

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream("GET", url, headers=headers) as response:
                    response.raise_for_status()
                    event_data = None
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("event:"):
                            event_data = line[6:].strip()
                        elif line.startswith("data:"):
                            if event_data == "task-update":
                                data = json.loads(line[6:])
                                yield data
        except httpx.HTTPStatusError as e:
            error_content = await e.response.aread()
            raise Exception(
                f"HTTP error occurred: {e.response.status_code} - {error_content.decode()}"
            )
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")

    def create(self, workflow: str, args: Optional[dict] = None) -> dict:
        return asyncio.run(self.async_create(workflow, args))

    def chat(self, user_message: str, thread_name: str) -> List[dict]:
        """Synchronous version that collects all updates into a list"""
        updates = []

        async def collect_updates():
            async for update in self.async_chat(user_message, thread_name):
                updates.append(update)

        asyncio.run(collect_updates())
        return updates

    async def async_create(self, workflow: str, args: Optional[dict] = None) -> dict:
        uri = f"{self.api_urls.tools_api_url}/create"
        headers = {"X-Api-Key": self.api_key.get_secret_value()}
        payload = {"workflow": workflow, "args": args or {}}

        async with httpx.AsyncClient() as client:
            response = await client.post(uri, json=payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            task_id = response_json.get("_id")

        async for event in self._subscribe(task_id):
            if event["status"] == "completed":
                return event["result"]
            if event["status"] == "failed":
                raise Exception(event.get("error", "Unknown error"))

    async def async_chat(
        self, user_message: str | dict, thread_name: str
    ) -> AsyncGenerator[dict, None]:
        uri = f"{self.api_urls.tools_api_url}/chat"
        headers = {
            "X-Api-Key": self.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }

        message = (
            {"content": user_message} if isinstance(user_message, str) else user_message
        )
        payload = {
            "user_message": message,
            "thread_name": thread_name,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(uri, json=payload, headers=headers)
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip() or not line.startswith("data: "):
                    continue

                event_data = json.loads(line[6:])
                if event_data["event"] == "done":
                    break

                yield event_data["data"]


def get_api_key() -> SecretStr:
    return SecretStr("d1617ae00280dcf216a74bbd6e9cd4b368e317db95386ca7")
    if os.getenv("EDEN_API_KEY"):
        return SecretStr(os.getenv("EDEN_API_KEY"))
    home_dir = os.path.expanduser("~")
    api_key_file = os.path.join(home_dir, ".eden")
    try:
        with open(api_key_file, "r") as file:
            api_key = file.read().strip()
        return SecretStr(api_key)
    except FileNotFoundError:
        raise Exception(
            "\033[91mNo EDEN_API_KEY found. Please set it in your environment or run `eden login` to save it in your home directory.\033[0m"
        )


eden_client = EdenClient()

args = {
    "prompt": "A professional photo of <concept> as a fashion model looking fresh, beautiful, youthful, easeful, bright.",
    "lora": "66904ec042b902d8eb3b41e6",
    "look_image": "https://storage.googleapis.com/public-assets-xander/Random/remove/xhibit/test2.jpeg",
    "face_image": "https://storage.googleapis.com/public-assets-xander/Random/remove/xhibit/face.jpeg",
    "resolution": 1152,
}

response = eden_client.create(workflow="xhibit_vton", args=args)
