from typing import Optional

from eve.agent.agent import Agent
from eve.agent.session.models import Deployment
from eve.tool import ToolContext

from eve.agent.deployments.email import EmailClient


async def handler(context: ToolContext):
    if not context.agent:
        raise Exception("Agent is required")

    return await handle_email_send(
        agent_id=str(context.agent),
        args=context.args,
        user=str(context.user) if context.user else None,
    )


async def handle_email_send(
    agent_id: str,
    args: dict,
    user: Optional[str] = None,
):
    agent = Agent.from_mongo(agent_id)
    deployment = Deployment.load(agent=agent.id, platform="email")
    if not deployment or not deployment.valid:
        raise Exception("No valid email deployments found")

    client = EmailClient(agent=agent, deployment=deployment)

    to_address = args.get("to")
    subject = args.get("subject")
    text = args.get("text") or args.get("body")
    html = args.get("html")
    reply_to = args.get("reply_to")

    if not to_address:
        raise ValueError("Recipient email address is required")
    if not subject:
        raise ValueError("Email subject is required")
    if not text and not html:
        raise ValueError("Email body is required")

    response = await client.send_email(
        to_address=to_address,
        subject=subject,
        text_content=text or "",
        html_content=html,
        reply_to=reply_to,
    )

    message_id = response.get("id") if isinstance(response, dict) else None

    return {
        "output": [
            {
                "url": f"mailto:{to_address}",
                "title": subject,
                "status": "sent",
                "message_id": message_id,
            }
        ]
    }
