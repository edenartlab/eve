import os
import modal
from pathlib import Path
from eve.llm import UserMessage
from eve.agent import Agent
from eve.auth import User
from eve.thread import Thread
from eve.api.api_requests import UpdateConfig
from eve.deploy import Deployment
from eve.api.handlers import run_chat_request

db = os.getenv("DB", "STAGE").upper()


trigger_message = """<AdminMessage>
You have received a request from an admin to run a scheduled task. The instructions for the task are below. In your response, do not ask for clarification, just do the task. Do not acknowledge receipt of this message, as no one else in the chat can see it and the admin is absent. Simply follow whatever instructions are below.
</AdminMessage>
<Task>
{task}
</Task>"""


async def run_chat_trigger(
    user_id: str,
    agent_id: str,
    thread_id: str,
    message: str,
    # schedule: dict,
    discord_channel_id: str,
):
    # Initialize necessary objects
    user = User.from_mongo(user_id)
    agent = Agent.from_mongo(agent_id)
    thread = Thread.from_mongo(thread_id)
    tools = agent.get_tools(cache=True)
    
    # Get Discord deployment for this agent
    sub_channel_name = f"{agent.username}_discord_{db}"
    deployment = Deployment.load(agent=agent.id, platform="discord")
    if not deployment:
        print("No Discord deployment found for this agent")
        return
    
    update_config = UpdateConfig(
        sub_channel_name=sub_channel_name,
        deployment_id=str(deployment.id),
        discord_channel_id=discord_channel_id,
    )

    user_message = UserMessage(
        content=trigger_message.format(task=message), 
        hidden=True
    )
    
    result = await run_chat_request(
        user=user,
        agent=agent,
        thread=thread,
        tools=tools,
        user_message=user_message,
        update_config=update_config,
        force_reply=True,
        dont_reply=False,
        model="claude-3-5-sonnet-20241022"
    )

    print(result)


root_dir = Path(__file__).parent.parent.parent
workflows_dir = root_dir / ".." / "workflows"

app = modal.App(
    "scheduled-chat",
    secrets=[
        modal.Secret.from_name("eve-secrets"),
        modal.Secret.from_name(f"eve-secrets-{db}"),
    ],
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"DB": db})
    .apt_install("git", "libmagic1", "ffmpeg", "wget")
    .pip_install_from_pyproject(str(root_dir / "pyproject.toml"))
    .copy_local_dir(str(workflows_dir), "/workflows")
)

@app.function(
    image=image,
    schedule=modal.Cron("48 5 * * 3")
)
async def create_chat_trigger():
    await run_chat_trigger(
        user_id="65284b18f8bbb9bff13ebe65",
        agent_id="675fd3af79e00297cdac1324",
        thread_id="67ac076fb3d975b5820fce3c",
        message="make up a new episode of Seinfeld. do not end with a question or call to action or request to create something. just the Seinfeld episode and then end.",
        discord_channel_id="1268682080263606443"
    )
