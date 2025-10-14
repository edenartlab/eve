from eden import EdenClient
from loguru import logger

eden_client = EdenClient()

args = {
    "prompt": "a terrified, tiny Donal Trump running away from a giant, evil kamala harris monster that is taking over the white house",
    # "seed": 0
}

response = eden_client.create(workflow="beeple_ai", args=args)

logger.info(response)
