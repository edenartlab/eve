from eve.agent.agent import Agent
from eve.agent.session.models import Deployment, ChatMessage, LLMConfig, LLMContext
from eve.agent.session.session_llm import ToolMetadataBuilder, async_prompt
from pydantic import BaseModel
import discord


class DiscordPostQuery(BaseModel):
    channel_id: str  # The ID of the channel to post to
    content: str  # The message content to post


async def handler(args: dict, user: str = None, agent: str = None):
    agent = Agent.from_mongo(args["agent"])
    deployment = Deployment.load(agent=agent.id, platform="discord")
    if not deployment:
        raise Exception("No valid Discord deployments found")

    # Get allowed channels from deployment config
    allowed_channels = deployment.config.discord.channel_allowlist
    if not allowed_channels:
        raise Exception("No channels configured for this deployment")

    # Create a mapping of channel notes to their IDs
    channel_map = {
        str(channel.note).lower(): channel.id for channel in allowed_channels
    }

    # Get query and content from args
    query = args.get("query")
    content = args.get("content")

    if not query:
        raise Exception("Query parameter is required")
    if not content:
        raise Exception("Content parameter is required")

    # Use LLM to parse the query and determine target channel
    system_message = """You are a Discord post query parser. Your task is to:
1. Analyze the query to determine which channel to post to
2. Return a structured response with the channel ID and content

Available channel IDs + notes:
{channel_notes}

Example queries:
"Post 'Hello everyone!' to general chat" -> Use general chat channel
"Send announcement to the news channel" -> Use news/announcements channel
"Share this in tech support" -> Use tech support channel

You must return a DiscordPostQuery object containing:
- channel_id: The ID of the channel to post to (must match one of the available channel IDs)
- content: The exact message content to post

The content should be extracted from the query or use the provided content parameter.""".format(
        channel_notes="\n".join(f"{id}: {note}" for id, note in channel_map.items())
    )

    messages = [
        ChatMessage(role="system", content=system_message),
        ChatMessage(
            role="user",
            content=f"Parse this Discord post query: {query}. Content to post: {content}",
        ),
    ]

    response = await async_prompt(
        context=LLMContext(
            messages=messages,
            metadata=ToolMetadataBuilder(
                tool_name="discord_post",
                user_id=args.get("user"),
                agent_id=args.get("agent"),
            )(),
            config=LLMConfig(
                response_format=DiscordPostQuery,
            ),
        ),
    )
    parsed_query = DiscordPostQuery.model_validate_json(response.content)

    # Verify the channel is in the allowlist
    channel_id = parsed_query.channel_id
    if not any(str(channel.id) == channel_id for channel in allowed_channels):
        allowed_channels_info = {
            channel.note: str(channel.id) for channel in allowed_channels
        }
        raise Exception(
            f"Channel {channel_id} is not in the allowlist. Allowed channels (note: id): {allowed_channels_info}"
        )

    # Create Discord client
    client = discord.Client(intents=discord.Intents.default())

    try:
        # Login to Discord
        await client.login(deployment.secrets.discord.token)

        # Get the channel and post the message
        channel = await client.fetch_channel(int(channel_id))
        message = await channel.send(content=parsed_query.content)

        return {
            "output": [
                {
                    "url": f"https://discord.com/channels/{channel.guild.id}/{channel.id}/{message.id}",
                }
            ]
        }

    finally:
        await client.close()
