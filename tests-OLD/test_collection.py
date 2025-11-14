from bson import ObjectId

from eve.agent import Agent
from eve.task import CreationsCollection
from eve.auth import get_my_eden_user


def test_collection():
    user = get_my_eden_user()

    agent = Agent.load("verdelis")
    tools = agent.get_tools()

    tool = tools["flux_dev_lora"]
    # task = tool.start_task(user.id, agent.id, {
    #     "prompt": "verdelis is in a library"
    # })

    # result = tool.wait(task)
    # print(result)

    result = {
        "status": "completed",
        "error": None,
        "result": [
            {
                "output": [
                    {
                        "filename": "dd5c14efbbdb7a4534088b71caa268d1f75eb9389749483b1cf27831a4b3608d.png",
                        "mediaAttributes": {
                            "mimeType": "image/png",
                            "width": 1024,
                            "height": 1024,
                            "aspectRatio": 1.0,
                            "blurhash": "UEB3H7oeD*I;~URkE2xt%LRkM{xa%fo0MyWV",
                        },
                        "creation": ObjectId("67db88e2854f5189ad5dcf21"),
                    }
                ]
            }
        ],
    }

    assert result["status"] == "completed"

    creation = result["result"][0]["output"][0]["creation"]

    creation = ObjectId("67db8b6f4c5de9444861ef52")

    collection = CreationsCollection.load(
        "test_collection", user.id, create_if_missing=True
    )

    collection.add_creation(creation)


test_collection()
