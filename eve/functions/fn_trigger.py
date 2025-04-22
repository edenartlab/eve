import os
import requests
import asyncio
from datetime import datetime, timezone
from eve.trigger import delete_trigger

trigger_message = """<AdminMessage>
You have received a request from an admin to run a scheduled task. The instructions for the task are below. In your response, do not ask for clarification, just do the task. Do not acknowledge receipt of this message, as no one else in the chat can see it and the admin is absent. Simply follow whatever instructions are below.
</AdminMessage>
<Task>
{task}
</Task>"""


async def trigger_fn():
    trigger_id = os.getenv("TRIGGER_ID")
    api_url = os.getenv("EDEN_API_URL")

    response = requests.get(
        f"{api_url}/triggers/{trigger_id}",
        headers={"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"},
    )

    if not response.ok:
        raise Exception(
            f"Failed to get trigger info: {response.status_code} - {response.text}"
        )

    trigger = response.json()

    user_message = {
        "content": trigger_message.format(task=trigger["message"]),
    }

    chat_request = {
        "user_id": trigger["user"],
        "agent_id": trigger["agent"],
        "thread_id": trigger["thread"],
        "user_message": user_message,
        "update_config": trigger["update_config"],
        "force_reply": True,
    }

    response = requests.post(
        f"{api_url}/chat",
        json=chat_request,
        headers={"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"},
    )

    if not response.ok:
        raise Exception(
            f"Error making chat request: {response.status_code} - {response.text}"
        )

    print(f"Chat request successful: {response.json()}")

    if trigger["end_date"]:
        current_time = datetime.now(timezone.utc)
        end_date_str = trigger["end_date"]

        # Handle both Z and +00:00 format
        if end_date_str.endswith("Z"):
            end_date_str = end_date_str.replace("Z", "+00:00")

        end_date = datetime.fromisoformat(end_date_str)

        print(f"Current time: {current_time}")
        print(f"End date: {end_date}")

        if current_time > end_date:
            print(
                f"Trigger end date {end_date} has passed. Deleting trigger {trigger_id}"
            )
            await delete_trigger(trigger_id)


def trigger_fn_sync():
    asyncio.run(trigger_fn())
