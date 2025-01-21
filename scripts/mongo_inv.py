from pymongo import MongoClient
import os
from bson import ObjectId
from dotenv import load_dotenv
from datetime import datetime, timezone

from eve.agent import Agent
from eve.models import Model

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "eden-prod"


def fetch_thread_tasks(thread_id):
    # Connect to MongoDB

    if not MONGO_URI or not MONGO_DB_NAME:
        raise ValueError(
            "MONGO_URI and MONGO_DB_NAME must be set in environment variables"
        )

    # Replace %5E with ^ in the URI if it's URL encoded
    mongo_uri = MONGO_URI.replace("%5E", "^")
    client = MongoClient(mongo_uri)
    db = client[MONGO_DB_NAME]

    # Query the thread
    thread = db.threads3.find_one({"_id": ObjectId(thread_id)})
    if not thread:
        print(f"Thread {thread_id} not found")
        return

    # Get agent info
    agent_id = thread.get("agent")
    if agent_id:
        agent = db.users3.find_one({"_id": ObjectId(agent_id)})
        if agent:
            print(f"Agent Model: {agent.get('model')}")
            print("-" * 50)

    # Process each message
    for message in thread.get("messages", []):
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            for tool_call in tool_calls:
                task_id = tool_call.get("task")
                if task_id:
                    # Query the task
                    task = db.tasks3.find_one({"_id": ObjectId(task_id)})
                    if task:
                        print(f"Task ID: {task_id}")
                        print(f"Args: {task.get('args')}")
                        print("-" * 50)


def check_lora_model_mismatches():
    if not MONGO_URI or not MONGO_DB_NAME:
        raise ValueError(
            "MONGO_URI and MONGO_DB_NAME must be set in environment variables"
        )

    mongo_uri = MONGO_URI.replace("%5E", "^")
    client = MongoClient(mongo_uri)
    db = client[MONGO_DB_NAME]

    threads = db.threads3.find().sort("_id", -1).limit(100)

    for thread in threads:
        thread_id = thread["_id"]
        agent_id = thread.get("agent")

        if not agent_id:
            continue

        # Get agent model
        agent = db.users3.find_one({"_id": ObjectId(agent_id)})
        if not agent:
            continue

        agent_model_id = agent.get("model")
        agent_model_name = "No Model Configured"

        if agent_model_id:
            model = db.models3.find_one({"_id": ObjectId(agent_model_id)})
            if model:
                agent_model_name = model.get("name", "Unknown")

        # Check each message's tasks
        for message in thread.get("messages", []):
            for tool_call in message.get("tool_calls", []):
                task_id = tool_call.get("task")
                if not task_id:
                    continue

                task = db.tasks3.find_one({"_id": ObjectId(task_id)})
                if not task:
                    continue

                args = task.get("args", {})
                if not isinstance(args, dict):
                    continue

                lora = args.get("lora")
                if lora:
                    # Get LoRA model name
                    lora_model = db.models3.find_one({"_id": ObjectId(lora)})
                    lora_name = (
                        lora_model.get("name", "Unknown") if lora_model else lora
                    )

                    # Print if there's a mismatch OR if agent has no model configured
                    if (
                        lora_name != agent_model_name
                        or agent_model_name == "No Model Configured"
                    ):
                        print(f"Thread: {thread_id}")
                        print(f"Agent Model: {agent_model_name}")
                        print(f"Task ID: {task_id}")
                        print(f"LoRA: {lora_name}")
                        if agent_model_name == "No Model Configured":
                            print(f"Full Task Args: {args}")
                            print(f"Tool Call: {tool_call}")
                        print("-" * 50)


if __name__ == "__main__":
    # thread_id = input("Enter thread ID: ")
    # fetch_thread_tasks(thread_id)
    # check_lora_model_mismatches()
    agent = Agent.from_mongo("6779d07f861030b3d3ac6b7a")
    # model = Model.from_mongo(None)
    # print(model)
