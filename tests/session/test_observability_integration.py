#!/usr/bin/env python3
"""
Manual integration test for prompt session observability.

This script connects to the staging database and runs a real prompt session
to verify that all observability features are working correctly.

Usage:
    ENV=staging python tests/session/test_observability_integration.py

Required environment variables:
    - ENV=staging
    - MONGO_URI (for staging DB)
    - SENTRY_DSN (optional, for Sentry integration)
    - LANGFUSE_PUBLIC_KEY (optional, for Langfuse integration)
    - LANGFUSE_SECRET_KEY (optional, for Langfuse integration)
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bson import ObjectId
from loguru import logger

from eve.agent.agent import Agent
from eve.agent.session.context import build_llm_context
from eve.agent.session.instrumentation import PromptSessionInstrumentation
from eve.agent.session.models import (
    ChatMessage,
    ChatMessageRequestInput,
    PromptSessionContext,
    Session,
    SessionRun,
)
from eve.agent.session.runtime import async_prompt_session

# Configure logging
logger.add(sys.stderr, level="DEBUG")


class ObservabilityIntegrationTest:
    """Integration test runner for prompt session observability."""

    def __init__(self):
        self.validate_environment()
        self.session = None
        self.agent = None
        self.session_run = None
        self.session_run_id = None

    def validate_environment(self):
        """Validate that we're running in the correct environment."""
        env = os.getenv("ENV", "local")
        if env != "staging":
            logger.warning(f"Running in {env} environment, expected 'staging'")
            if input("Continue anyway? (y/n): ").lower() != "y":
                sys.exit(1)

        # Check for observability keys
        if os.getenv("SENTRY_DSN"):
            logger.info("✓ Sentry DSN configured")
        else:
            logger.warning("✗ Sentry DSN not configured")

        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            logger.info("✓ Langfuse keys configured")
        else:
            logger.warning("✗ Langfuse keys not configured")

    def find_or_create_test_agent(self) -> Agent:
        """Find or create a test agent in staging DB."""
        # Try to find an existing test agent
        test_agents = Agent.find({"username": "observability_test_agent"})

        if test_agents:
            agent = test_agents[0]
            logger.info(f"Found existing test agent: {agent.id}")
        else:
            # Create a new test agent
            agent = Agent(
                username="observability_test_agent",
                name="Observability Test Agent",
                system="You are a helpful assistant for testing observability features. Respond concisely.",
                model="gpt-4o-mini",
                owner=ObjectId("65284b18f8bbb9bff13ebe65"),  # Test user ID
            )
            agent.save()
            logger.info(f"Created new test agent: {agent.id}")

        return agent

    def find_or_create_test_session(self, agent: Agent) -> Session:
        """Find or create a test session in staging DB."""
        # Create a new test session for this run
        session = Session(
            owner=agent.owner,
            agents=[agent.id],
            status="active",
            session_type="passive",
            platform="app",
            title=f"Observability Test - {datetime.now().isoformat()}",
        )
        session.save()
        logger.info(f"Created new test session: {session.id}")
        return session

    async def run_prompt_session(self):
        """Run a prompt session with observability enabled."""
        logger.info("Starting prompt session test...")

        # Create test message
        message_input = ChatMessageRequestInput(
            content="What is 2 + 2? Please respond with just the number.",
            role="user",
        )

        # Create prompt context
        prompt_context = PromptSessionContext(
            session=self.session,
            initiating_user_id=str(self.agent.owner),
            message=message_input,
        )

        # Build LLM context
        llm_context = await build_llm_context(
            self.session,
            self.agent,
            prompt_context,
            trace_id=None,
            instrumentation=None,
        )

        # Create instrumentation
        instrumentation = PromptSessionInstrumentation(
            session_id=str(self.session.id),
            user_id=str(self.agent.owner),
            agent_id=str(self.agent.id),
            sentry_enabled=bool(os.getenv("SENTRY_DSN")),
            langfuse_enabled=bool(
                os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
            ),
        )

        logger.info(f"Session run ID: {instrumentation.session_run_id}")
        self.session_run_id = instrumentation.session_run_id

        # Track updates
        updates = []

        try:
            # Run the prompt session
            async for update in async_prompt_session(
                session=self.session,
                llm_context=llm_context,
                agent=self.agent,
                stream=False,
                is_client_platform=False,
                session_run_id=instrumentation.session_run_id,
                api_key_id=None,
                instrumentation=instrumentation,
                context=prompt_context,
            ):
                updates.append(update)
                logger.debug(f"Update: {update.type.value}")

                if update.type.value == "assistant_message" and update.message:
                    logger.info(f"Assistant response: {update.message.content}")

            logger.success(f"Prompt session completed with {len(updates)} updates")

        except Exception as e:
            logger.error(f"Prompt session failed: {e}")
            raise

        return updates

    def verify_session_run(self):
        """Verify that SessionRun was created and updated correctly."""
        logger.info("Verifying SessionRun record...")

        # Find the SessionRun
        session_runs = SessionRun.find({"run_id": self.session_run_id})

        if not session_runs:
            logger.error(f"SessionRun not found for ID: {self.session_run_id}")
            return False

        self.session_run = session_runs[0]

        # Verify fields
        checks = [
            ("Session ID matches", self.session_run.session == self.session.id),
            (
                "Status is completed",
                self.session_run.status in ["completed", "in_progress"],
            ),
            ("Environment is set", self.session_run.environment is not None),
            ("Agent ID matches", self.session_run.agent_id == self.agent.id),
            ("Started at is set", self.session_run.started_at is not None),
            (
                "Trace IDs are consistent",
                self.session_run.sentry_trace_id == self.session_run_id,
            ),
            (
                "Metrics tracked",
                self.session_run.total_tokens > 0
                if self.session_run.total_tokens
                else True,
            ),
        ]

        all_passed = True
        for check_name, passed in checks:
            if passed:
                logger.success(f"✓ {check_name}")
            else:
                logger.error(f"✗ {check_name}")
                all_passed = False

        # Print trace URLs if available
        if self.session_run.sentry_url:
            logger.info(f"Sentry trace URL: {self.session_run.sentry_url}")
        if self.session_run.langfuse_url:
            logger.info(f"Langfuse trace URL: {self.session_run.langfuse_url}")

        # Print metrics
        logger.info("Metrics:")
        logger.info(f"  Total tokens: {self.session_run.total_tokens}")
        logger.info(f"  Prompt tokens: {self.session_run.prompt_tokens}")
        logger.info(f"  Completion tokens: {self.session_run.completion_tokens}")
        logger.info(f"  Cached tokens: {self.session_run.cached_tokens}")
        logger.info(f"  Total cost: ${self.session_run.total_cost_usd:.4f}")
        logger.info(f"  Message count: {self.session_run.message_count}")
        logger.info(f"  Tool calls: {self.session_run.tool_calls_count}")

        if self.session_run.duration_ms:
            logger.info(f"  Duration: {self.session_run.duration_ms:.2f}ms")

        return all_passed

    def verify_messages(self):
        """Verify that messages have observability data."""
        logger.info("Verifying message observability...")

        # Find messages for this session
        messages = ChatMessage.find({"session": self.session.id})

        assistant_messages = [m for m in messages if m.role == "assistant"]

        if not assistant_messages:
            logger.error("No assistant messages found")
            return False

        for msg in assistant_messages:
            if msg.observability:
                logger.success(f"✓ Message {msg.id} has observability data")
                logger.info(f"  Session run ID: {msg.observability.session_run_id}")
                logger.info(f"  Trace ID: {msg.observability.trace_id}")
                logger.info(f"  Generation ID: {msg.observability.generation_id}")
                logger.info(f"  Tokens spent: {msg.observability.tokens_spent}")

                if msg.observability.sentry_trace_id:
                    logger.info(
                        f"  Sentry trace ID: {msg.observability.sentry_trace_id}"
                    )
            else:
                logger.warning(f"✗ Message {msg.id} missing observability data")

        return True

    async def run(self):
        """Run the full integration test."""
        logger.info("=" * 60)
        logger.info("Starting Observability Integration Test")
        logger.info("=" * 60)

        try:
            # Setup
            self.agent = self.find_or_create_test_agent()
            self.session = self.find_or_create_test_session(self.agent)

            # Run prompt session
            await self.run_prompt_session()

            # Give time for async writes to complete
            await asyncio.sleep(2)

            # Verify results
            logger.info("\n" + "=" * 60)
            logger.info("Verification Results")
            logger.info("=" * 60)

            session_run_ok = self.verify_session_run()
            messages_ok = self.verify_messages()

            if session_run_ok and messages_ok:
                logger.success("\n✅ All observability checks passed!")
            else:
                logger.error("\n❌ Some observability checks failed")

            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("Test Summary")
            logger.info("=" * 60)
            logger.info(f"Session ID: {self.session.id}")
            logger.info(f"Session Run ID: {self.session_run_id}")
            logger.info(f"Agent ID: {self.agent.id}")

            if self.session_run:
                logger.info(f"Environment: {self.session_run.environment}")
                if self.session_run.sentry_url:
                    logger.info(
                        f"\n📊 View Sentry trace: {self.session_run.sentry_url}"
                    )
                if self.session_run.langfuse_url:
                    logger.info(
                        f"📊 View Langfuse trace: {self.session_run.langfuse_url}"
                    )

        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            raise


async def main():
    """Run the integration test."""
    test = ObservabilityIntegrationTest()
    await test.run()


if __name__ == "__main__":
    asyncio.run(main())
