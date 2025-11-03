from eve.agent.deployments.email import EmailClient
from eve.tool import ToolContext


async def handler(context: ToolContext):
    return await EmailClient.send_tool_email(
        agent_id=str(context.agent) if context.agent else None,
        args=context.args,
        user=str(context.user) if context.user else None,
    )
