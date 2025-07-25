from eve.agent.deployments import Deployment
from eve.agent import Agent
from telegram import Bot


async def handler(args: dict, user: str = None, agent: str = None):
    if not agent:
        raise Exception("Agent is required")
    agent = Agent.from_mongo(agent)
    deployment = Deployment.load(agent=agent.id, platform="telegram")
    if not deployment:
        raise Exception("No valid Telegram deployments found")

    # Get allowed chats from deployment config
    allowed_chats = deployment.config.telegram.topic_allowlist
    if not allowed_chats:
        raise Exception("No chats configured for this deployment")

    # Get chat ID and content from args
    chat_id = args["chat_id"]
    content = args["content"]

    # Verify the chat is in the allowlist
    if not any(str(chat.id) == chat_id for chat in allowed_chats):
        raise Exception(f"Chat {chat_id} is not in the allowlist")

    # Create Telegram bot
    bot = Bot(token=deployment.secrets.telegram.token)

    try:
        # Send the message
        message = await bot.send_message(chat_id=chat_id, text=content)

        return {
            "output": [{
                "message_id": message.message_id,
                "chat_id": message.chat.id,
                "url": f"https://t.me/{message.chat.username}/{message.message_id}",
            }]
        }

    finally:
        await bot.close()
