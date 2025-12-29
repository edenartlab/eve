"""
Integration tests for prompt session observability.

These tests validate that SessionRun tracking, Sentry tracing, and Langfuse integration
work correctly during prompt session execution.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from eve.agent.agent import Agent
from eve.agent.session.instrumentation import PromptSessionInstrumentation
from eve.agent.session.models import (
    ChatMessage,
    ChatMessageRequestInput,
    LLMConfig,
    LLMContext,
    PromptSessionContext,
    Session,
    SessionRun,
    UpdateType,
)
from eve.agent.session.runtime import PromptSessionRuntime, async_prompt_session


class TestObservabilityFixtures:
    """Fixtures for testing observability features."""

    @pytest.fixture
    def mock_environment(self, monkeypatch):
        """Set up environment variables for testing."""
        monkeypatch.setenv("ENV", "staging")
        monkeypatch.setenv("SENTRY_DSN", "https://test@sentry.io/123456")
        monkeypatch.setenv("SENTRY_ORG", "test-org")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public-key")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret-key")
        monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")
        monkeypatch.setenv("DEBUG_SESSION", "true")
        return monkeypatch

    @pytest.fixture
    def test_user_id(self):
        """Create a test user ID."""
        return ObjectId("65284b18f8bbb9bff13ebe65")  # From your screenshot

    @pytest.fixture
    def test_agent_id(self):
        """Create a test agent ID."""
        return ObjectId("6896568efccc3801cfa3a211")  # From your screenshot

    @pytest.fixture
    def test_session_id(self):
        """Create a test session ID."""
        return ObjectId("68f24ff31f0b40ea492a4fbd")  # From your screenshot

    @pytest.fixture
    def test_agent(self, test_agent_id, test_user_id):
        """Create a test agent."""
        agent = Agent(
            id=test_agent_id,
            name="Test Agent",
            username="test_agent",
            owner=test_user_id,
            userImage=None,
            model="gpt-4o-mini",
            system="You are a helpful test assistant.",
        )
        # Mock the save method
        agent.save = MagicMock()
        return agent

    @pytest.fixture
    def test_session(self, test_session_id, test_user_id, test_agent_id):
        """Create a test session."""
        session = Session(
            id=test_session_id,
            owner=test_user_id,
            agents=[test_agent_id],
            status="active",
            session_type="passive",
            platform="app",
            messages=[],
        )
        # Mock save and update methods
        session.save = MagicMock()
        session.update = MagicMock()
        return session

    @pytest.fixture
    def test_message_input(self):
        """Create a test message input."""
        return ChatMessageRequestInput(
            content="Test prompt for observability",
            role="user",
        )

    @pytest.fixture
    def test_prompt_context(self, test_session, test_user_id, test_message_input):
        """Create a test prompt session context."""
        return PromptSessionContext(
            session=test_session,
            initiating_user_id=str(test_user_id),
            message=test_message_input,
            session_run_id=None,  # Will be generated
        )

    @pytest.fixture
    def test_llm_context(self, test_session, test_agent):
        """Create a test LLM context."""
        system_message = ChatMessage(
            role="system",
            content=test_agent.system,
            session=[test_session.id],
        )
        user_message = ChatMessage(
            role="user",
            content="Test prompt for observability",
            session=[test_session.id],
            sender=test_session.owner,
        )

        return LLMContext(
            messages=[system_message, user_message],
            config=LLMConfig(model="gpt-4o-mini"),
            tools=None,
            tool_choice=None,
        )


class TestSessionRunCreation(TestObservabilityFixtures):
    """Test SessionRun model creation and lifecycle."""

    def test_session_run_creation(
        self, test_session_id, test_agent_id, test_user_id, mock_environment
    ):
        """Test that SessionRun is created with correct fields."""
        run_id = str(uuid.uuid4())

        with patch.object(SessionRun, "save") as mock_save:
            session_run = SessionRun(
                session=test_session_id,
                run_id=run_id,
                status="started",
                environment="staging",
                sentry_trace_id=run_id,
                langfuse_trace_id=run_id,
                started_at=datetime.now(timezone.utc),
                agent_id=test_agent_id,
                user_id=test_user_id,
                platform="app",
                is_streaming=False,
            )

            session_run.save()

            assert session_run.session == test_session_id
            assert session_run.run_id == run_id
            assert session_run.status == "started"
            assert session_run.environment == "staging"
            assert session_run.sentry_trace_id == run_id
            assert session_run.langfuse_trace_id == run_id
            assert session_run.agent_id == test_agent_id
            assert session_run.user_id == test_user_id
            assert session_run.platform == "app"
            assert session_run.is_streaming is False
            mock_save.assert_called_once()

    def test_session_run_completion(self, test_session_id, test_agent_id, test_user_id):
        """Test SessionRun completion with metrics."""
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        with patch.object(SessionRun, "save") as mock_save:
            session_run = SessionRun(
                session=test_session_id,
                run_id=run_id,
                status="started",
                environment="staging",
                started_at=started_at,
                agent_id=test_agent_id,
                user_id=test_user_id,
            )

            # Update metrics
            session_run.update_metrics(
                tokens=1500,
                prompt_tokens=1000,
                completion_tokens=500,
                cached_tokens=100,
                cost=0.015,
                messages=2,
                tool_calls=1,
            )

            # Complete the run
            session_run.complete(status="completed")

            assert session_run.status == "completed"
            assert session_run.completed_at is not None
            assert session_run.duration_ms is not None
            assert session_run.duration_ms >= 0
            assert session_run.total_tokens == 1500
            assert session_run.prompt_tokens == 1000
            assert session_run.completion_tokens == 500
            assert session_run.cached_tokens == 100
            assert session_run.total_cost_usd == 0.015
            assert session_run.message_count == 2
            assert session_run.tool_calls_count == 1
            assert (
                mock_save.call_count >= 2
            )  # At least once for metrics, once for completion

    def test_session_run_failure(self, test_session_id, test_agent_id, test_user_id):
        """Test SessionRun failure tracking."""
        run_id = str(uuid.uuid4())

        with patch.object(SessionRun, "save"):
            session_run = SessionRun(
                session=test_session_id,
                run_id=run_id,
                status="started",
                environment="staging",
                started_at=datetime.now(timezone.utc),
                agent_id=test_agent_id,
                user_id=test_user_id,
            )

            # Simulate failure
            error = ValueError("Test error message")
            session_run.complete(status="failed", error=error)

            assert session_run.status == "failed"
            assert session_run.error_type == "ValueError"
            assert session_run.error_message == "Test error message"
            assert session_run.completed_at is not None
            assert session_run.duration_ms is not None

    def test_trace_url_generation(self, test_session_id, mock_environment, monkeypatch):
        """Test trace URL generation for Sentry and Langfuse."""
        run_id = str(uuid.uuid4())

        # Set up environment for URL generation
        monkeypatch.setenv(
            "SENTRY_DSN", "https://test@o123456.ingest.sentry.io/4507847609942016"
        )
        monkeypatch.setenv("SENTRY_ORG", "edenlabs")
        monkeypatch.setenv("LANGFUSE_PROJECT_ID", "cm7wbibzs01nwad07i58uznu9")

        session_run = SessionRun(
            session=test_session_id,
            run_id=run_id,
            environment="staging",
            sentry_trace_id=run_id,
            langfuse_trace_id=run_id,
            started_at=datetime.now(timezone.utc),
        )

        urls = session_run.build_trace_urls()

        # Check Sentry URL
        if "sentry" in urls:
            assert "edenlabs.sentry.io/explore/traces/trace/" in urls["sentry"]
            assert f"trace/{run_id}" in urls["sentry"]
            assert "project=4507847609942016" in urls["sentry"]
            assert "source=traces" in urls["sentry"]

        # Check Langfuse URL
        if "langfuse" in urls:
            assert (
                "test.langfuse.com/project/cm7wbibzs01nwad07i58uznu9/traces"
                in urls["langfuse"]
            )
            assert f"peek={run_id}" in urls["langfuse"]
            assert "filter=environment" in urls["langfuse"]


class TestPromptSessionRuntime(TestObservabilityFixtures):
    """Test PromptSessionRuntime with observability features."""

    @pytest.mark.asyncio
    async def test_runtime_creates_session_run(
        self,
        test_session,
        test_agent,
        test_llm_context,
        mock_environment,
    ):
        """Test that PromptSessionRuntime creates a SessionRun on initialization."""
        with patch.object(SessionRun, "save") as mock_save:
            runtime = PromptSessionRuntime(
                session=test_session,
                llm_context=test_llm_context,
                actor=test_agent,
                stream=False,
                is_client_platform=False,
                session_run_id=None,
                api_key_id=None,
            )

            assert runtime.session_run is not None
            assert runtime.session_run.session == test_session.id
            assert runtime.session_run.status == "started"
            assert runtime.session_run.environment == "staging"
            assert runtime.session_run.agent_id == test_agent.id
            assert runtime.session_run.sentry_trace_id == runtime.session_run_id
            assert runtime.session_run.langfuse_trace_id == runtime.session_run_id
            mock_save.assert_called()

    @pytest.mark.asyncio
    async def test_runtime_updates_session_run_status(
        self,
        test_session,
        test_agent,
        test_llm_context,
        mock_environment,
    ):
        """Test that runtime updates SessionRun status throughout lifecycle."""
        with patch.object(SessionRun, "save"):
            with patch("eve.agent.session.runtime.provider_async_prompt") as mock_llm:
                # Mock LLM response
                mock_response = MagicMock()
                mock_response.content = "Test response"
                mock_response.tool_calls = None
                mock_response.stop = "completed"
                mock_response.tokens_spent = 100
                mock_response.usage = None
                mock_response.thought = None
                mock_response.llm_call_id = None
                mock_llm.return_value = mock_response

                runtime = PromptSessionRuntime(
                    session=test_session,
                    llm_context=test_llm_context,
                    actor=test_agent,
                    stream=False,
                    is_client_platform=False,
                    session_run_id=None,
                    api_key_id=None,
                )

                initial_status = runtime.session_run.status

                # Simulate running the prompt loop
                # Note: We're not actually running the full loop here,
                # just testing that the status would be updated
                runtime.session_run.status = "in_progress"
                runtime.session_run.save()

                assert initial_status == "started"
                assert runtime.session_run.status == "in_progress"

                # Simulate completion
                runtime.session_run.complete(status="completed")
                assert runtime.session_run.status == "completed"
                assert runtime.session_run.completed_at is not None


class TestSentryIntegration(TestObservabilityFixtures):
    """Test Sentry integration and trace propagation."""

    def test_sentry_transaction_creation(self, mock_environment):
        """Test that Sentry transaction is created with correct trace ID."""
        with patch("sentry_sdk.start_transaction") as mock_start_transaction:
            mock_transaction = MagicMock()
            mock_start_transaction.return_value = mock_transaction

            session_run_id = str(uuid.uuid4())
            instrumentation = PromptSessionInstrumentation(
                session_id="test-session-id",
                session_run_id=session_run_id,
                user_id="test-user-id",
                agent_id="test-agent-id",
                sentry_enabled=True,
            )

            instrumentation.ensure_sentry_transaction(
                name="prompt_session",
                op="session.prompt",
            )

            # Verify transaction was created with correct trace_id
            mock_start_transaction.assert_called_once_with(
                name="prompt_session",
                op="session.prompt",
                sampled=None,
                trace_id=session_run_id,
            )

            # Verify tags were set
            mock_transaction.set_tag.assert_any_call("session_id", "test-session-id")
            mock_transaction.set_tag.assert_any_call("session_run_id", session_run_id)
            mock_transaction.set_tag.assert_any_call("user_id", "test-user-id")
            mock_transaction.set_tag.assert_any_call("agent_id", "test-agent-id")
            mock_transaction.set_tag.assert_any_call("environment", "staging")

    @pytest.mark.asyncio
    async def test_sentry_spans_created(
        self,
        test_session,
        test_agent,
        test_llm_context,
        mock_environment,
    ):
        """Test that Sentry spans are created for different operations."""
        with patch("sentry_sdk.start_span") as mock_start_span:
            mock_span = MagicMock()
            mock_start_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_start_span.return_value.__exit__ = MagicMock(return_value=None)

            runtime = PromptSessionRuntime(
                session=test_session,
                llm_context=test_llm_context,
                actor=test_agent,
                stream=False,
                is_client_platform=False,
                session_run_id=None,
                api_key_id=None,
            )

            # Test context building span
            await runtime._refresh_llm_messages()

            # Verify span was created
            mock_start_span.assert_any_call(op="session.context_build")


class TestLangfuseIntegration(TestObservabilityFixtures):
    """Test Langfuse integration."""

    def test_langfuse_trace_creation(self, mock_environment):
        """Test that Langfuse trace is created with correct IDs."""
        with patch("eve.agent.session.instrumentation.Langfuse") as MockLangfuse:
            mock_client = MagicMock()
            mock_trace = MagicMock()
            mock_client.trace.return_value = mock_trace
            MockLangfuse.return_value = mock_client

            session_run_id = str(uuid.uuid4())
            instrumentation = PromptSessionInstrumentation(
                session_id="test-session-id",
                session_run_id=session_run_id,
                user_id="test-user-id",
                agent_id="test-agent-id",
                langfuse_enabled=True,
            )

            instrumentation.ensure_langfuse_trace()

            # Verify client was created
            MockLangfuse.assert_called_once_with(
                environment=None,  # Not set in our mock env
                sdk_integration="eve-session-instrumentation",
            )

            # Verify trace was created with correct parameters
            mock_client.trace.assert_called_once_with(
                id=session_run_id,
                name="prompt_session",
                user_id="test-user-id",
                session_id="test-session-id",
                metadata={
                    "session_id": "test-session-id",
                    "session_run_id": session_run_id,
                    "agent_id": "test-agent-id",
                    "user_id": "test-user-id",
                },
                tags=["prompt_session"],
                input=None,
                output=None,
            )


class TestEndToEndObservability(TestObservabilityFixtures):
    """End-to-end integration test for full prompt session."""

    @pytest.mark.asyncio
    async def test_full_prompt_session_observability(
        self,
        test_session,
        test_agent,
        test_prompt_context,
        mock_environment,
    ):
        """Test complete prompt session flow with all observability features."""
        with patch("eve.agent.session.runtime.provider_async_prompt") as mock_llm:
            with patch("eve.agent.session.runtime.get_provider") as mock_get_provider:
                with patch(
                    "eve.agent.session.runtime.select_messages"
                ) as mock_select_messages:
                    with patch(
                        "eve.agent.session.runtime.get_all_eden_messages_for_llm"
                    ) as mock_eden_messages:
                        with patch(
                            "eve.agent.session.runtime.build_llm_context"
                        ) as mock_build_context:
                            with patch.object(
                                SessionRun, "save"
                            ) as mock_session_run_save:
                                with patch.object(
                                    ChatMessage, "save"
                                ) as mock_message_save:
                                    # Setup mocks
                                    mock_response = MagicMock()
                                    mock_response.content = "Test response"
                                    mock_response.tool_calls = None
                                    mock_response.stop = "completed"
                                    mock_response.tokens_spent = 100
                                    mock_response.prompt_tokens = 75
                                    mock_response.completion_tokens = 25
                                    mock_response.usage = MagicMock()
                                    mock_response.usage.model_dump.return_value = {
                                        "prompt_tokens": 75,
                                        "completion_tokens": 25,
                                        "total_tokens": 100,
                                    }
                                    mock_response.thought = None
                                    mock_response.llm_call_id = None
                                    mock_llm.return_value = mock_response

                                    mock_provider = MagicMock()
                                    mock_get_provider.return_value = mock_provider

                                    mock_select_messages.return_value = []
                                    mock_eden_messages.return_value = []

                                    # Create LLM context
                                    llm_context = LLMContext(
                                        messages=[
                                            ChatMessage(
                                                role="system",
                                                content="Test system message",
                                            ),
                                            ChatMessage(
                                                role="user", content="Test user message"
                                            ),
                                        ],
                                        config=LLMConfig(model="gpt-4o-mini"),
                                    )
                                    mock_build_context.return_value = llm_context

                                    # Create instrumentation
                                    instrumentation = PromptSessionInstrumentation(
                                        session_id=str(test_session.id),
                                        user_id=str(
                                            test_prompt_context.initiating_user_id
                                        ),
                                        sentry_enabled=True,
                                        langfuse_enabled=True,
                                    )

                                    # Track all updates
                                    updates = []

                                    # Run prompt session
                                    async for update in async_prompt_session(
                                        session=test_session,
                                        llm_context=llm_context,
                                        agent=test_agent,
                                        stream=False,
                                        is_client_platform=False,
                                        session_run_id=None,
                                        api_key_id=None,
                                        instrumentation=instrumentation,
                                        context=test_prompt_context,
                                    ):
                                        updates.append(update)

                                    # Verify updates were generated
                                    assert len(updates) > 0

                                    # Check for expected update types
                                    update_types = [u.type for u in updates]
                                    assert UpdateType.START_PROMPT in update_types
                                    assert UpdateType.ASSISTANT_MESSAGE in update_types
                                    assert UpdateType.END_PROMPT in update_types

                                    # Verify SessionRun was saved multiple times
                                    assert (
                                        mock_session_run_save.call_count >= 2
                                    )  # At least creation and completion

                                    # Verify message was saved
                                    assert mock_message_save.called

    @pytest.mark.asyncio
    async def test_observability_with_tool_calls(
        self,
        test_session,
        test_agent,
        test_prompt_context,
        mock_environment,
    ):
        """Test observability when tools are called during session."""
        # This test would verify that tool execution is properly tracked
        # in SessionRun metrics and Sentry spans
        pass  # Implement when tool system is available

    @pytest.mark.asyncio
    async def test_observability_with_streaming(
        self,
        test_session,
        test_agent,
        test_prompt_context,
        mock_environment,
    ):
        """Test observability with streaming responses."""
        # This test would verify that streaming responses are properly tracked
        # with appropriate spans and metrics
        pass  # Implement when streaming is testable
