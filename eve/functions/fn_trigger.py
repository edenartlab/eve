import os
import requests
import asyncio
from datetime import datetime, timezone

trigger_message = """<SystemMessage>
You have received a request from an admin to run a scheduled task. The instructions for the task are below. In your response, do not ask for clarification, just do the task. Do not acknowledge receipt of this message, as no one else in the chat can see it and the admin is absent. Simply follow whatever instructions are below.
</SystemMessage>
<Task>
{task}
</Task>"""

trigger_message_post = """
<PostInstruction>
When you have completed the task, write out a single summary of the result of the task. Make sure to include the URLs to any relevant media you created. Do not include intermediate results, just the media relevant to the task. Then post it on {platform} using the discord_post tool to channel "{platform_channel_id}". Do not forget to do this at the end.
</PostInstruction>"""


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

    user_message = trigger_message.format(task=trigger["message"])
    update_config = trigger.get("update_config", None)

    if update_config:
        discord_channel_id = update_config.get("discord_channel_id", None)
        telegram_channel_id = update_config.get("telegram_channel_id", None)
        if discord_channel_id:
            update_config = None
            user_message += trigger_message_post.format(
                platform="Discord",
                platform_channel_id=discord_channel_id,
            )
        elif telegram_channel_id:
            update_config = None
            user_message += trigger_message_post.format(
                platform="Telegram",
                platform_channel_id=telegram_channel_id,
            )
        else:
            print("No platform specified")

    chat_request = {
        "user_id": trigger["user"],
        "agent_id": trigger["agent"],
        "thread_id": trigger["thread"],
        "user_message": {"content": user_message},
        "update_config": update_config,
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

    if trigger["schedule"].get("end_date"):
        # Get current time with full precision
        current_time = datetime.now(timezone.utc)
        end_date_str = trigger["schedule"]["end_date"]

        # Parse the date string and ensure it's timezone aware
        try:
            # If using fromisoformat, the timezone info should be preserved
            end_date = datetime.fromisoformat(end_date_str)

            # If somehow end_date is still naive, make it timezone aware
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
        except ValueError:
            # Fallback parsing if there's a format issue
            if end_date_str.endswith("Z"):
                end_date_str = end_date_str.replace("Z", "+00:00")
            end_date = datetime.fromisoformat(end_date_str)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)

        # Only round end_date to minute precision
        end_date = end_date.replace(second=0, microsecond=0)

        print(f"Current time: {current_time}")
        print(f"End date (rounded): {end_date}")

        if current_time > end_date:
            print(
                f"Trigger end date {end_date} has passed. Deleting trigger {trigger_id}"
            )
            response = requests.post(
                f"{api_url}/triggers/delete",
                headers={"Authorization": f"Bearer {os.getenv('EDEN_ADMIN_KEY')}"},
                json={"id": trigger["id"]},
            )

            if not response.ok:
                raise Exception(
                    f"Failed to delete trigger: {response.status_code} - {response.text}"
                )


def trigger_fn_sync():
    asyncio.run(trigger_fn())
