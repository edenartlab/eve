from ..handler import handle_email_send


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise ValueError("Agent identifier is required")

    return await handle_email_send(agent_id=agent, args=args, user=user)
