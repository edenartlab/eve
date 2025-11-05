import pytest
import json
from bson import ObjectId
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks

from eve.api.api_requests import (
    PromptSessionRequest,
    ChatMessageRequestInput,
    UpdateConfig,
)
from eve.api.handlers import handle_prompt_session, setup_session
from eve.api.errors import APIError
from eve.agent.session.models import Session
from eve.user import User
from eve.agent import Agent


class TestSessionPrompt:
    """Test class for session prompt functionality"""

    @pytest.fixture
    def mock_user(self):
        """Create a mock user"""
        user = MagicMock(spec=User)
        user.id = ObjectId()
        return user

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent"""
        agent = MagicMock(spec=Agent)
        agent.id = ObjectId()
        return agent

    @pytest.fixture
    def mock_session(self, mock_user, mock_agent):
        """Create a mock session"""
        session = MagicMock(spec=Session)
        session.id = ObjectId()
        session.owner = mock_user.id
        session.agents = [mock_agent.id]
        session.status = "active"
        session.messages = []
        return session

    def test_setup_session_existing_session_id(self, mock_session):
        """Test setup_session with existing session_id"""
        session_id = str(mock_session.id)
        user_id = str(mock_session.owner)

        with patch("eve.api.handlers.Session.from_mongo") as mock_from_mongo:
            mock_from_mongo.return_value = mock_session

            result = setup_session(session_id, user_id)

            assert result == mock_session
            mock_from_mongo.assert_called_once_with(ObjectId(session_id))

    def test_setup_session_nonexistent_session_id(self):
        """Test setup_session with non-existent session_id"""
        session_id = str(ObjectId())
        user_id = str(ObjectId())

        with patch("eve.api.handlers.Session.from_mongo") as mock_from_mongo:
            mock_from_mongo.return_value = None

            with pytest.raises(APIError, match="Session not found"):
                setup_session(session_id, user_id)

    def test_setup_session_create_new_session_success(self, mock_user, mock_agent):
        """Test successful session creation when session_id is None"""
        request = PromptSessionRequest(
            session_id=None,
            owner_id=str(mock_user.id),
            agents=[str(mock_agent.id)],
            title="Test Session",
            budget=100.0,
        )

        mock_session = MagicMock(spec=Session)
        mock_session.id = ObjectId()

        with patch("eve.api.handlers.Session") as mock_session_class:
            mock_session_class.return_value = mock_session

            result = setup_session(None, str(mock_user.id), request)

            assert result == mock_session
            mock_session_class.assert_called_once_with(
                owner=mock_user.id,
                agents=[mock_agent.id],
                title="Test Session",
                budget=100.0,
                status="active",
            )
            mock_session.save.assert_called_once()

    def test_setup_session_create_new_session_missing_owner_id(self):
        """Test session creation failure when owner_id is missing"""
        request = PromptSessionRequest(
            session_id=None, agents=[str(ObjectId())], title="Test Session"
        )

        with pytest.raises(APIError, match="owner_id is required for session creation"):
            setup_session(None, str(ObjectId()), request)

    def test_setup_session_create_new_session_missing_agents(self):
        """Test session creation failure when agents list is missing"""
        request = PromptSessionRequest(
            session_id=None, owner_id=str(ObjectId()), title="Test Session"
        )

        with pytest.raises(
            APIError, match="agents list is required for session creation"
        ):
            setup_session(None, str(ObjectId()), request)

    def test_setup_session_create_new_session_no_request(self):
        """Test session creation failure when no request object is provided"""
        with pytest.raises(
            APIError, match="Session creation requires additional parameters"
        ):
            setup_session(None, str(ObjectId()), None)

    def test_setup_session_warns_about_ignored_fields(self, mock_session, caplog):
        """Test that a warning is logged when session creation fields are provided with existing session_id"""
        session_id = str(mock_session.id)
        user_id = str(mock_session.owner)

        request = PromptSessionRequest(
            session_id=session_id,
            owner_id=str(ObjectId()),  # This should be ignored
            agents=[str(ObjectId())],  # This should be ignored
            title="Ignored Title",  # This should be ignored
        )

        with patch("eve.api.handlers.Session.from_mongo") as mock_from_mongo:
            mock_from_mongo.return_value = mock_session

            result = setup_session(session_id, user_id, request)

            assert result == mock_session
            # Check that warning was logged
            assert any(
                "Session creation fields provided but ignored" in record.message
                for record in caplog.records
            )

    @pytest.mark.asyncio
    async def test_handle_prompt_session_existing_session(self, mock_session):
        """Test handle_prompt_session with existing session"""
        request = PromptSessionRequest(
            session_id=str(mock_session.id),
            message=ChatMessageRequestInput(content="Hello"),
            user_id=str(ObjectId()),
        )
        background_tasks = BackgroundTasks()

        with patch("eve.api.handlers.setup_session") as mock_setup:
            mock_setup.return_value = mock_session
            with patch("eve.api.handlers.run_prompt_session") as mock_run_prompt:
                result = await handle_prompt_session(request, background_tasks)

                assert result == {"session_id": str(mock_session.id)}
                mock_setup.assert_called_once_with(
                    str(mock_session.id), request.user_id, request
                )

    @pytest.mark.asyncio
    async def test_handle_prompt_session_create_new_session(
        self, mock_user, mock_agent
    ):
        """Test handle_prompt_session with new session creation"""
        mock_session = MagicMock(spec=Session)
        mock_session.id = ObjectId()

        request = PromptSessionRequest(
            session_id=None,
            message=ChatMessageRequestInput(content="Hello"),
            user_id=str(mock_user.id),
            owner_id=str(mock_user.id),
            agents=[str(mock_agent.id)],
            title="New Test Session",
        )
        background_tasks = BackgroundTasks()

        with patch("eve.api.handlers.setup_session") as mock_setup:
            mock_setup.return_value = mock_session
            with patch("eve.api.handlers.run_prompt_session") as mock_run_prompt:
                result = await handle_prompt_session(request, background_tasks)

                assert result == {"session_id": str(mock_session.id)}
                mock_setup.assert_called_once_with(None, request.user_id, request)

    @pytest.mark.asyncio
    async def test_handle_prompt_session_stream_response(self, mock_session):
        """Test handle_prompt_session with streaming enabled"""
        request = PromptSessionRequest(
            session_id=str(mock_session.id),
            message=ChatMessageRequestInput(content="Hello"),
            user_id=str(ObjectId()),
            stream=True,
        )
        background_tasks = BackgroundTasks()

        mock_stream_data = [
            {"type": "start_prompt", "agent": None},
            {"type": "assistant_token", "text": "Hello"},
            {"type": "end_prompt"},
        ]

        async def mock_stream_generator():
            for data in mock_stream_data:
                yield data

        with patch("eve.api.handlers.setup_session") as mock_setup:
            mock_setup.return_value = mock_session
            with patch("eve.api.handlers.run_prompt_session_stream") as mock_stream:
                mock_stream.return_value = mock_stream_generator()

                result = await handle_prompt_session(request, background_tasks)

                # For streaming, we expect a StreamingResponse
                assert hasattr(result, "media_type")
                assert result.media_type == "text/event-stream"


if __name__ == "__main__":
    pytest.main([__file__])


def example_usage():
    """Example showing how to use the new session creation functionality"""
    from eve.api.api_requests import PromptSessionRequest, ChatMessageRequestInput

    # Example 1: Create a new session and prompt it
    new_session_request = PromptSessionRequest(
        session_id=None,  # No session_id means create new session
        message=ChatMessageRequestInput(content="Hello, let's start a conversation!"),
        user_id="60d4f1c85f8b8b1234567890",
        owner_id="60d4f1c85f8b8b1234567890",  # Required for new sessions
        agents=[
            "60d4f1c85f8b8b1234567891",
            "60d4f1c85f8b8b1234567892",
        ],  # Required for new sessions
        title="New Chat Session",
        budget=100.0,
    )

    # Example 2: Use existing session
    existing_session_request = PromptSessionRequest(
        session_id="60d4f1c85f8b8b1234567893",  # Use existing session
        message=ChatMessageRequestInput(content="Continue our conversation"),
        user_id="60d4f1c85f8b8b1234567890",
    )

    # print("Example requests created successfully!")
    # print(f"New session request: {new_session_request}")
    # print(f"Existing session request: {existing_session_request}")


if __name__ == "__main__":
    # Run tests by default, but can also call example_usage() for demonstration
    pytest.main([__file__])
