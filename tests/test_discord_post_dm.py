"""
Tests for discord_post DM functionality
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId
import discord

from eve.tools.discord.discord_post.handler import handler, send_dm
from eve.agent.agent import Agent
from eve.agent.session.models import Deployment


class TestDiscordPostDM:
    """Test class for discord_post DM functionality"""

    @pytest.mark.asyncio
    async def test_handler_validation(self):
        """Test handler parameter validation"""
        mock_agent = MagicMock(spec=Agent)
        mock_agent.id = ObjectId()

        with patch("eve.tools.discord.discord_post.handler.Agent.from_mongo") as mock_agent_from_mongo:
            mock_agent_from_mongo.return_value = mock_agent

            with patch("eve.tools.discord.discord_post.handler.Deployment.load") as mock_deployment_load:
                mock_deployment = MagicMock(spec=Deployment)
                mock_deployment_load.return_value = mock_deployment

                # Test missing both channel_id and discord_user_id
                args = {"content": "Test message"}
                with pytest.raises(Exception, match="Either channel_id or discord_user_id must be provided"):
                    await handler(args, user="test_user", agent="test_agent")

                # Test empty content
                args = {
                    "discord_user_id": "123456789",
                    "content": ""
                }
                with pytest.raises(Exception, match="Content cannot be empty"):
                    await handler(args, user="test_user", agent="test_agent")

                # Test whitespace-only content
                args = {
                    "discord_user_id": "123456789",
                    "content": "   "
                }
                with pytest.raises(Exception, match="Content cannot be empty"):
                    await handler(args, user="test_user", agent="test_agent")

    @pytest.mark.asyncio
    async def test_send_dm_success(self):
        """Test successful DM sending"""
        mock_client = MagicMock(spec=discord.Client)
        mock_user = MagicMock()
        mock_user.name = "testuser"

        mock_dm_channel = MagicMock()
        mock_dm_channel.id = "987654321"
        mock_user.dm_channel = mock_dm_channel

        mock_message = MagicMock()
        mock_message.id = "message123"

        mock_client.fetch_user = AsyncMock(return_value=mock_user)
        mock_user.send = AsyncMock(return_value=mock_message)

        result = await send_dm(mock_client, "123456789", "Test DM content")

        # Verify the result structure
        assert "output" in result
        assert len(result["output"]) == 1
        assert "url" in result["output"][0]
        assert result["output"][0]["url"] == "https://discord.com/channels/@me/987654321/message123"

        # Verify Discord API calls
        mock_client.fetch_user.assert_called_once_with(123456789)
        mock_user.send.assert_called_once_with(content="Test DM content")

    @pytest.mark.asyncio
    async def test_send_dm_forbidden(self):
        """Test DM sending when user has DMs disabled"""
        mock_client = MagicMock(spec=discord.Client)
        mock_user = MagicMock()

        mock_client.fetch_user = AsyncMock(return_value=mock_user)
        mock_user.send = AsyncMock(side_effect=discord.Forbidden(
            response=MagicMock(), message="Cannot send messages to this user"
        ))

        with pytest.raises(Exception, match="Cannot send DM to user 123456789: DMs disabled or bot blocked"):
            await send_dm(mock_client, "123456789", "Test DM content")

    @pytest.mark.asyncio
    async def test_send_dm_user_not_found(self):
        """Test DM sending when user doesn't exist"""
        mock_client = MagicMock(spec=discord.Client)

        mock_client.fetch_user = AsyncMock(side_effect=discord.NotFound(
            response=MagicMock(), message="Unknown User"
        ))

        with pytest.raises(Exception, match="User 123456789 not found"):
            await send_dm(mock_client, "123456789", "Test DM content")

    @pytest.mark.asyncio
    async def test_handler_dm_flow(self):
        """Test complete DM handler flow"""
        mock_agent = MagicMock(spec=Agent)
        mock_agent.id = ObjectId()

        mock_deployment = MagicMock(spec=Deployment)
        mock_deployment.secrets.discord.token = "test_token"

        mock_client = MagicMock(spec=discord.Client)
        mock_user = MagicMock()
        mock_user.name = "testuser"
        mock_user.dm_channel.id = "987654321"

        mock_message = MagicMock()
        mock_message.id = "message123"

        with patch("eve.tools.discord.discord_post.handler.Agent.from_mongo") as mock_agent_from_mongo:
            mock_agent_from_mongo.return_value = mock_agent

            with patch("eve.tools.discord.discord_post.handler.Deployment.load") as mock_deployment_load:
                mock_deployment_load.return_value = mock_deployment

                with patch("eve.tools.discord.discord_post.handler.discord.Client") as mock_client_class:
                    mock_client_class.return_value = mock_client
                    mock_client.login = AsyncMock()
                    mock_client.close = AsyncMock()
                    mock_client.fetch_user = AsyncMock(return_value=mock_user)
                    mock_user.send = AsyncMock(return_value=mock_message)

                    args = {
                        "discord_user_id": "123456789",
                        "content": "Test DM message"
                    }

                    result = await handler(args, user="test_user", agent="test_agent")

                    # Verify the result
                    assert "output" in result
                    assert len(result["output"]) == 1
                    assert "url" in result["output"][0]

                    # Verify Discord client was used correctly
                    mock_client.login.assert_called_once_with("test_token")
                    mock_client.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])