from eden import EdenClient
from loguru import logger

eden_client = EdenClient()

thread_id = eden_client.get_or_create_thread("test_thread_anthro")
logger.info(thread_id)

response = eden_client.chat(
    thread_id=thread_id,
    message={
        "content": "make a picture of a dog with a dark grittier style",
        "settings": {},
        "attachments": [],
    },
)

logger.info(response)
