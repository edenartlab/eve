from eve.agent.deployments.gmail import GmailClient
from eve.tool import ToolContext


async def handler(context: ToolContext):
    return await GmailClient.send_tool_email(
        agent_id=str(context.agent) if context.agent else None,
        args=context.args,
        user=str(context.user) if context.user else None,
    )
