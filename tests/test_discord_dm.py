"""
Basic validation tests for discord_dm tool
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from bson import ObjectId

from eve.tools.discord.discord_dm.handler import (
    handler,
    discover_active_users,
    map_discord_to_eden_users,
    DiscordUser
)
from eve.user import User
from eve.agent.agent import Agent
from eve.agent.session.models import Deployment


class TestDiscordDM:
    """Test class for discord_dm functionality"""

    def test_discord_user_model(self):
        """Test DiscordUser model creation and validation"""
        user = DiscordUser(
            discord_id="123456789",
            discord_username="testuser",
            message_count=5,
            last_seen="2024-01-01T12:00:00Z"
        )

        assert user.discord_id == "123456789"
        assert user.discord_username == "testuser"
        assert user.message_count == 5
        assert user.last_seen == "2024-01-01T12:00:00Z"

    @pytest.mark.asyncio
    async def test_handler_validation_missing_agent(self):
        """Test handler fails without agent"""
        args = {
            "channel_id": "123456789",
            "instruction": "Test instruction"
        }

        with pytest.raises(Exception, match="Agent is required"):
            await handler(args, user="test_user", agent=None)

    @pytest.mark.asyncio
    async def test_handler_validation_missing_parameters(self):
        """Test handler validates required parameters"""
        mock_agent = MagicMock(spec=Agent)
        mock_agent.id = ObjectId()

        with patch("eve.tools.discord.discord_dm.handler.Agent.from_mongo") as mock_agent_from_mongo:
            mock_agent_from_mongo.return_value = mock_agent

            with patch("eve.tools.discord.discord_dm.handler.Deployment.load") as mock_deployment_load:
                mock_deployment = MagicMock(spec=Deployment)
                mock_deployment_load.return_value = mock_deployment

                # Test missing channel_id
                args = {"instruction": "Test instruction"}
                with pytest.raises(Exception, match="channel_id and instruction are required"):
                    await handler(args, user="test_user", agent="test_agent")

                # Test missing instruction
                args = {"channel_id": "123456789"}
                with pytest.raises(Exception, match="channel_id and instruction are required"):
                    await handler(args, user="test_user", agent="test_agent")

                # Test invalid active_days
                args = {
                    "channel_id": "123456789",
                    "instruction": "Test instruction",
                    "active_days": 0
                }
                with pytest.raises(Exception, match="active_days must be between 1 and 30"):
                    await handler(args, user="test_user", agent="test_agent")

                # Test invalid message_limit
                args = {
                    "channel_id": "123456789",
                    "instruction": "Test instruction",
                    "message_limit": 0
                }
                with pytest.raises(Exception, match="message_limit must be between 1 and 1000"):
                    await handler(args, user="test_user", agent="test_agent")

    @pytest.mark.asyncio
    async def test_map_discord_to_eden_users(self):
        """Test Discord user to Eden user mapping"""
        discord_users = [
            DiscordUser(
                discord_id="123456789",
                discord_username="testuser1",
                message_count=5,
                last_seen="2024-01-01T12:00:00Z"
            ),
            DiscordUser(
                discord_id="987654321",
                discord_username="testuser2",
                message_count=3,
                last_seen="2024-01-01T11:00:00Z"
            )
        ]

        mock_eden_user1 = MagicMock(spec=User)
        mock_eden_user1.id = ObjectId()
        mock_eden_user1.username = "discord_testuser1"

        mock_eden_user2 = MagicMock(spec=User)
        mock_eden_user2.id = ObjectId()
        mock_eden_user2.username = "discord_testuser2"

        with patch("eve.tools.discord.discord_dm.handler.User.from_discord") as mock_from_discord:
            mock_from_discord.side_effect = [mock_eden_user1, mock_eden_user2]

            result = await map_discord_to_eden_users(discord_users)

            assert len(result) == 2
            assert result[0][0] == discord_users[0]
            assert result[0][1] == mock_eden_user1
            assert result[1][0] == discord_users[1]
            assert result[1][1] == mock_eden_user2

            # Verify User.from_discord was called correctly
            mock_from_discord.assert_any_call("123456789", "testuser1")
            mock_from_discord.assert_any_call("987654321", "testuser2")


if __name__ == "__main__":
    pytest.main([__file__])