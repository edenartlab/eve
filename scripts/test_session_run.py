#!/usr/bin/env python3
"""
Quick test script for SessionRun URL generation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone

from bson import ObjectId

# Set up environment variables as they would be in production
os.environ["SENTRY_DSN"] = (
    "https://77484b6a0ff1393f6e1deb3ead9ff325@o4505840236363776.ingest.us.sentry.io/4508724347142144"
)
os.environ["SENTRY_ORG"] = "edenlabs"
os.environ["SENTRY_ENV"] = "jmill-dev"
os.environ["LANGFUSE_PROJECT_ID"] = "cm7wbibzs01nwad07i58uznu9"
os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"
os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = "jmill-dev"

from eve.agent.session.models import SessionRun

# Create a test SessionRun
session_run = SessionRun(
    session=ObjectId(),
    run_id="12518cde-148b-4866-904e-bb4b508900c9",
    environment="jmill-dev",  # Should match SENTRY_ENV/LANGFUSE_TRACING_ENVIRONMENT
    sentry_trace_id="12518cde-148b-4866-904e-bb4b508900c9",
    langfuse_trace_id="12518cde-148b-4866-904e-bb4b508900c9",
    started_at=datetime.now(timezone.utc),
    agent_id=ObjectId("675f880079e00297cd9b45d9"),
    user_id=ObjectId("6952aea2cc6f530c955ddbea"),
    platform="app",
    is_streaming=False,
)

# Build URLs
urls = session_run.build_trace_urls()

print("=" * 60)
print("SessionRun URL Generation Test")
print("=" * 60)
print()
print(f"Session Run ID: {session_run.run_id}")
print(f"Environment: {session_run.environment}")
print()
print("Generated URLs:")
print("-" * 60)
print()

if "sentry" in urls:
    print("✓ Sentry URL generated:")
    print(f"  {urls['sentry']}")
    print()
    # Verify structure
    assert "edenlabs.sentry.io/explore/traces/trace/" in urls["sentry"]
    assert session_run.run_id in urls["sentry"]
    assert "project=4508724347142144" in urls["sentry"]
    print("  ✓ URL structure validated")
else:
    print("✗ Sentry URL not generated")

print()

if "langfuse" in urls:
    print("✓ Langfuse URL generated:")
    print(f"  {urls['langfuse']}")
    print()
    # Verify structure
    assert (
        "us.cloud.langfuse.com/project/cm7wbibzs01nwad07i58uznu9/traces"
        in urls["langfuse"]
    )
    assert f"peek={session_run.run_id}" in urls["langfuse"]
    assert "filter=environment" in urls["langfuse"]
    assert "jmill-dev" in urls["langfuse"]
    print("  ✓ URL structure validated")
else:
    print("✗ Langfuse URL not generated")

print()
print("=" * 60)
print("All tests passed!")
print("=" * 60)
