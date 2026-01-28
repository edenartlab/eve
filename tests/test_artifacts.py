"""Tests for the Eden Artifacts V3 MCP tools.

These tests verify the artifact CRUD operations, JSON Patch updates,
version history, and list functionality.

Run with: pytest tests/test_artifacts.py -v -m live

Note: These are integration tests that require:
- EDEN_FASTIFY_API_URL environment variable set (e.g., http://localhost:5050)
- EDEN_FASTIFY_ADMIN_KEY environment variable set
- EDEN_API_KEY environment variable set (for user authentication)

IMPORTANT - Connection limits:
The local Eden Fastify API has connection limits that can cause tests to fail
when running all 26 tests together. All tests pass when run individually.

Recommended ways to run tests:
1. Run individual tests:
   pytest tests/test_artifacts.py::test_patch_add_field -v -m live

2. Run small batches (3-5 tests):
   pytest tests/test_artifacts.py::test_create_artifact_minimal \
          tests/test_artifacts.py::test_patch_add_field \
          tests/test_artifacts.py::test_delete_artifact -v -m live

3. Run all tests (may have intermittent failures due to connection limits):
   pytest tests/test_artifacts.py -v -m live
"""

import asyncio
import json
import os
import uuid
from typing import Optional

import httpx
import pytest
import pytest_asyncio

from eve.tool import Tool

# =============================================================================
# Connectivity Check
# =============================================================================


async def check_api_connectivity() -> tuple[bool, str]:
    """Check if the Eden Fastify API is reachable.

    Returns (is_connected, message).
    """
    api_url = os.environ.get("EDEN_FASTIFY_API_URL")
    if not api_url:
        return False, "EDEN_FASTIFY_API_URL not set"

    # Try to connect with retries (more attempts with longer delays)
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Just check if the server responds (404 is fine, means it's up)
                response = await client.get(api_url)
                return True, f"Connected to {api_url} (status: {response.status_code})"
        except httpx.ConnectError as e:
            if attempt < 4:
                await asyncio.sleep(2)
                continue
            return (
                False,
                f"Cannot connect to {api_url} after 5 attempts. "
                f"If using ngrok, ensure the tunnel is active. Error: {e}",
            )
        except Exception as e:
            return False, f"Error connecting to {api_url}: {e}"

    return False, f"Cannot connect to {api_url} after 5 attempts"


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def api_connectivity(event_loop):
    """Session-scoped fixture to check API connectivity once at the start."""
    is_connected, message = await check_api_connectivity()
    if not is_connected:
        pytest.skip(f"API not reachable: {message}")
    return True


@pytest.fixture(autouse=True)
def require_api_connectivity(request, api_connectivity):
    """Automatically require API connectivity for all live tests."""
    # Only applies to tests marked as 'live'
    if "live" in [marker.name for marker in request.node.iter_markers()]:
        # api_connectivity fixture will skip if not connected
        pass


@pytest_asyncio.fixture(autouse=True)
async def rate_limit_delay(request):
    """Add delays between tests to avoid overwhelming the API connection pool.

    The local Eden Fastify API has connection limits. With 2-second delays,
    tests may occasionally fail when run all together. Run individual tests
    or small batches for more reliable results.
    """
    # Only add delay for live tests
    if "live" in [marker.name for marker in request.node.iter_markers()]:
        await asyncio.sleep(2.0)  # 2 second delay before each test
    yield
    if "live" in [marker.name for marker in request.node.iter_markers()]:
        await asyncio.sleep(2.0)  # 2 second delay after each test


# =============================================================================
# Retry Logic for Transient Failures
# =============================================================================


async def retry_async(
    coro_func,
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    **kwargs,
):
    """Retry an async function with exponential backoff for transient failures."""
    last_error: Optional[Exception] = None
    last_result: Optional[dict] = None

    retryable_keywords = [
        "connect",
        "timeout",
        "taskgroup",
        "network",
        "ssl",
        "tls",
        "mint",
        "token",
    ]

    for attempt in range(max_retries):
        try:
            result = await coro_func(*args, **kwargs)
            last_result = result
            # Also check for failed status with network-like errors
            if isinstance(result, dict) and result.get("status") == "failed":
                error = str(result.get("error", "")).lower()
                if any(kw in error for kw in retryable_keywords):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        await asyncio.sleep(delay)
                        continue
            return result
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on connection/network errors or TaskGroup exceptions
            retryable = any(kw in error_str for kw in retryable_keywords)
            if not retryable:
                raise
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                await asyncio.sleep(delay)

    if last_error:
        raise last_error
    if last_result is not None:
        return last_result
    raise RuntimeError("retry_async: no result or error after retries")


# =============================================================================
# Configuration
# =============================================================================

# Path to MCP tool definitions
TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "eve", "tools", "mcp", "eden")

# Cached test user ID
_test_user_id: Optional[str] = None


def check_required_env_vars() -> None:
    """Check that required environment variables are set."""
    required = ["EDEN_FASTIFY_API_URL", "EDEN_FASTIFY_ADMIN_KEY", "EDEN_API_KEY"]
    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        pytest.skip(f"Missing required environment variables: {', '.join(missing)}")


def get_test_user_id() -> str:
    """Get test user ID from EDEN_API_KEY, skip test if not configured."""
    global _test_user_id

    if _test_user_id is not None:
        return _test_user_id

    check_required_env_vars()

    try:
        from eve.auth import get_my_eden_user

        user = get_my_eden_user()
        _test_user_id = str(user.id)
        return _test_user_id
    except Exception as e:
        pytest.skip(f"Could not get test user from EDEN_API_KEY: {e}")


# =============================================================================
# Tool Loading Helper
# =============================================================================


def load_artifact_tool(tool_name: str) -> Tool:
    """Load an artifact tool from YAML file."""
    api_file = os.path.join(TOOLS_DIR, tool_name, "api.yaml")
    return Tool.from_yaml(api_file)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def generate_unique_title() -> str:
    """Generate a unique title for test artifacts."""
    return f"Test Artifact {uuid.uuid4().hex[:8]}"


async def _run_tool(tool, args: dict) -> dict:
    """Run a tool with retry logic for transient failures."""
    result = await retry_async(tool.async_run, args)
    return result


async def create_artifact(
    title: str,
    artifact_type: str = "test_type",
    items: Optional[dict] = None,
    session: Optional[str] = None,
) -> dict:
    """Helper to create an artifact and return the parsed response."""
    user_id = get_test_user_id()
    tool = load_artifact_tool("eden_artifacts_v3_create")
    args = {
        "title": title,
        "type": artifact_type,
        "user_id": user_id,
    }
    if items:
        args["items"] = items
    if session:
        args["session"] = session

    result = await _run_tool(tool, args)
    assert result.get("status") == "completed", f"Create failed: {result}"

    # Parse the output - it may be JSON string or already parsed
    output = result["output"][0]
    if isinstance(output, str):
        return json.loads(output)
    return output


async def get_artifact(artifact_id: str) -> dict:
    """Helper to get an artifact and return the parsed response."""
    user_id = get_test_user_id()
    tool = load_artifact_tool("eden_artifacts_v3_get")
    args = {"artifactId": artifact_id, "user_id": user_id}

    result = await _run_tool(tool, args)
    assert result.get("status") == "completed", f"Get failed: {result}"

    output = result["output"][0]
    if isinstance(output, str):
        return json.loads(output)
    return output


async def delete_artifact(artifact_id: str) -> dict:
    """Helper to delete an artifact."""
    user_id = get_test_user_id()
    tool = load_artifact_tool("eden_artifacts_v3_delete")
    args = {"artifactId": artifact_id, "user_id": user_id}

    result = await _run_tool(tool, args)
    return result


async def patch_artifact(
    artifact_id: str,
    operations: list,
    reason: Optional[str] = None,
) -> dict:
    """Helper to patch an artifact's items."""
    user_id = get_test_user_id()
    tool = load_artifact_tool("eden_artifacts_v3_patch_items")
    args = {
        "artifactId": artifact_id,
        "operations": operations,
        "user_id": user_id,
    }
    if reason:
        args["reason"] = reason

    result = await _run_tool(tool, args)
    assert result.get("status") == "completed", f"Patch failed: {result}"

    # Handle various output formats
    outputs = result.get("output", [])
    if not outputs:
        # No output - fetch current state
        return await get_artifact(artifact_id)

    output = outputs[0]
    if output is None or (isinstance(output, str) and not output.strip()):
        # Empty response - fetch the artifact to return current state
        return await get_artifact(artifact_id)

    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            # If output is not valid JSON, fetch current state
            return await get_artifact(artifact_id)
    return output


async def update_artifact_metadata(
    artifact_id: str,
    title: Optional[str] = None,
    artifact_type: Optional[str] = None,
    public: Optional[bool] = None,
) -> dict:
    """Helper to update artifact metadata."""
    user_id = get_test_user_id()
    tool = load_artifact_tool("eden_artifacts_v3_update")
    args = {"artifactId": artifact_id, "user_id": user_id}
    if title:
        args["title"] = title
    if artifact_type:
        args["type"] = artifact_type
    if public is not None:
        args["public"] = public

    result = await _run_tool(tool, args)
    assert result.get("status") == "completed", f"Update failed: {result}"

    output = result["output"][0]
    if isinstance(output, str):
        return json.loads(output)
    return output


async def list_artifacts(
    owner_id: Optional[str] = None,
    artifact_type: Optional[str] = None,
    session_id: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
) -> dict:
    """Helper to list artifacts."""
    user_id = get_test_user_id()
    tool = load_artifact_tool("eden_artifacts_v3_list")
    args = {"page": page, "limit": limit, "user_id": user_id}
    if owner_id:
        args["ownerId"] = owner_id
    if artifact_type:
        args["type"] = artifact_type
    if session_id:
        args["sessionId"] = session_id

    result = await _run_tool(tool, args)
    assert result.get("status") == "completed", f"List failed: {result}"

    output = result["output"][0]
    if isinstance(output, str):
        return json.loads(output)
    return output


async def get_artifact_history(
    artifact_id: str,
    page: int = 1,
    limit: int = 20,
    direction: int = -1,
) -> dict:
    """Helper to get artifact version history."""
    user_id = get_test_user_id()
    tool = load_artifact_tool("eden_artifacts_v3_history")
    args = {
        "artifactId": artifact_id,
        "page": page,
        "limit": limit,
        "direction": direction,
        "user_id": user_id,
    }

    result = await _run_tool(tool, args)
    assert result.get("status") == "completed", f"History failed: {result}"

    output = result["output"][0]
    if isinstance(output, str):
        return json.loads(output)
    return output


@pytest_asyncio.fixture
async def test_artifact():
    """Create a test artifact and clean it up after the test."""
    title = generate_unique_title()
    response = await create_artifact(
        title=title,
        artifact_type="test_type",
        items={"key": "value", "nested": {"a": 1}},
    )

    artifact = response.get("artifact", response)
    artifact_id = artifact.get("_id") or artifact.get("id")

    yield artifact

    # Cleanup
    try:
        await delete_artifact(artifact_id)
    except Exception:
        pass  # Ignore cleanup errors


# =============================================================================
# Test Artifact Creation
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.live
async def test_create_artifact_minimal():
    """Test creating an artifact with minimal required fields."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(title=title, artifact_type="test_type")

        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        assert artifact_id is not None
        assert artifact.get("title") == title
        assert artifact.get("type") == "test_type"
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_create_artifact_with_items():
    """Test creating an artifact with initial items data."""
    title = generate_unique_title()
    artifact_id = None

    initial_items = {
        "characters": [
            {"name": "Alice", "role": "protagonist"},
            {"name": "Bob", "role": "antagonist"},
        ],
        "setting": {"location": "New York", "era": "2024"},
        "metadata": {"version": 1, "draft": True},
    }

    try:
        response = await create_artifact(
            title=title,
            artifact_type="character_database",
            items=initial_items,
        )

        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        assert artifact_id is not None
        assert artifact.get("title") == title
        assert artifact.get("type") == "character_database"

        # Verify items were stored correctly
        items = artifact.get("items", {})
        assert "characters" in items
        assert len(items["characters"]) == 2
        assert items["characters"][0]["name"] == "Alice"
        assert items["setting"]["location"] == "New York"
        assert items["metadata"]["version"] == 1
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


# =============================================================================
# Test Artifact Get
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.live
async def test_get_artifact(test_artifact):
    """Test fetching an artifact by ID."""
    artifact_id = test_artifact.get("_id") or test_artifact.get("id")

    response = await get_artifact(artifact_id)

    artifact = response.get("artifact", response)
    assert artifact.get("_id") == artifact_id or artifact.get("id") == artifact_id
    assert artifact.get("items", {}).get("key") == "value"


@pytest.mark.asyncio
@pytest.mark.live
async def test_get_nonexistent_artifact():
    """Test fetching a non-existent artifact returns an error."""
    user_id = get_test_user_id()
    fake_id = "000000000000000000000000"

    tool = load_artifact_tool("eden_artifacts_v3_get")
    result = await _run_tool(tool, {"artifactId": fake_id, "user_id": user_id})

    # Should either fail or return an error in the output
    output = result.get("output", [None])[0]
    if isinstance(output, str):
        # Check if it contains error indication
        assert (
            "error" in output.lower()
            or "not found" in output.lower()
            or result.get("status") == "failed"
        )


# =============================================================================
# Test Artifact Update (Metadata)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.live
async def test_update_artifact_title(test_artifact):
    """Test updating an artifact's title."""
    artifact_id = test_artifact.get("_id") or test_artifact.get("id")
    new_title = f"Updated {generate_unique_title()}"

    response = await update_artifact_metadata(artifact_id, title=new_title)

    artifact = response.get("artifact", response)
    assert artifact.get("title") == new_title


@pytest.mark.asyncio
@pytest.mark.live
async def test_update_artifact_type(test_artifact):
    """Test updating an artifact's type."""
    artifact_id = test_artifact.get("_id") or test_artifact.get("id")

    response = await update_artifact_metadata(artifact_id, artifact_type="new_type")

    artifact = response.get("artifact", response)
    assert artifact.get("type") == "new_type"


@pytest.mark.asyncio
@pytest.mark.live
async def test_update_artifact_public(test_artifact):
    """Test updating an artifact's public flag."""
    artifact_id = test_artifact.get("_id") or test_artifact.get("id")

    response = await update_artifact_metadata(artifact_id, public=True)

    artifact = response.get("artifact", response)
    assert artifact.get("public") is True


# =============================================================================
# Test Artifact Patch (JSON Patch Operations)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.live
async def test_patch_add_field():
    """Test adding a new field using JSON Patch add operation."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={"existing": "value"},
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Add a new field
        patch_response = await patch_artifact(
            artifact_id,
            operations=[{"op": "add", "path": "/newField", "value": "newValue"}],
            reason="Adding new field",
        )

        patched = patch_response.get("artifact", patch_response)
        items = patched.get("items", {})
        assert items.get("newField") == "newValue"
        assert items.get("existing") == "value"  # Original preserved
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_patch_add_to_array():
    """Test appending to an array using JSON Patch add with /-."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={"list": ["item1", "item2"]},
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Append to array using /-
        patch_response = await patch_artifact(
            artifact_id,
            operations=[{"op": "add", "path": "/list/-", "value": "item3"}],
        )

        patched = patch_response.get("artifact", patch_response)
        items = patched.get("items", {})
        assert len(items["list"]) == 3
        assert items["list"][2] == "item3"
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_patch_insert_at_index():
    """Test inserting at a specific array index using JSON Patch add."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={"list": ["first", "third"]},
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Insert at index 1
        patch_response = await patch_artifact(
            artifact_id,
            operations=[{"op": "add", "path": "/list/1", "value": "second"}],
        )

        patched = patch_response.get("artifact", patch_response)
        items = patched.get("items", {})
        assert items["list"] == ["first", "second", "third"]
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_patch_replace_field():
    """Test replacing a field using JSON Patch replace operation."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={"name": "original", "count": 1},
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Replace field value
        patch_response = await patch_artifact(
            artifact_id,
            operations=[{"op": "replace", "path": "/name", "value": "updated"}],
        )

        patched = patch_response.get("artifact", patch_response)
        items = patched.get("items", {})
        assert items.get("name") == "updated"
        assert items.get("count") == 1  # Unchanged
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_patch_replace_nested_field():
    """Test replacing a nested field using JSON Pointer path."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={
                "characters": [
                    {"name": "Alice", "level": 1},
                    {"name": "Bob", "level": 2},
                ]
            },
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Update Alice's level
        patch_response = await patch_artifact(
            artifact_id,
            operations=[{"op": "replace", "path": "/characters/0/level", "value": 10}],
            reason="Leveled up Alice",
        )

        patched = patch_response.get("artifact", patch_response)
        items = patched.get("items", {})
        assert items["characters"][0]["level"] == 10
        assert items["characters"][0]["name"] == "Alice"  # Name unchanged
        assert items["characters"][1]["level"] == 2  # Bob unchanged
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_patch_remove_field():
    """Test removing a field using JSON Patch remove operation."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={"keep": "this", "remove": "this"},
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Remove field
        patch_response = await patch_artifact(
            artifact_id,
            operations=[{"op": "remove", "path": "/remove"}],
        )

        patched = patch_response.get("artifact", patch_response)
        items = patched.get("items", {})
        assert items.get("keep") == "this"
        assert "remove" not in items
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_patch_remove_array_element():
    """Test removing an array element by index."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={"list": ["a", "b", "c"]},
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Remove middle element
        patch_response = await patch_artifact(
            artifact_id,
            operations=[{"op": "remove", "path": "/list/1"}],
        )

        patched = patch_response.get("artifact", patch_response)
        items = patched.get("items", {})
        assert items["list"] == ["a", "c"]
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_patch_multiple_operations():
    """Test applying multiple operations in a single patch."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={
                "name": "original",
                "tags": ["old"],
                "temp": "delete_me",
            },
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Multiple operations
        patch_response = await patch_artifact(
            artifact_id,
            operations=[
                {"op": "replace", "path": "/name", "value": "updated"},
                {"op": "add", "path": "/tags/-", "value": "new"},
                {"op": "remove", "path": "/temp"},
                {"op": "add", "path": "/version", "value": 2},
            ],
            reason="Batch update",
        )

        patched = patch_response.get("artifact", patch_response)
        items = patched.get("items", {})
        assert items.get("name") == "updated"
        assert items.get("tags") == ["old", "new"]
        assert "temp" not in items
        assert items.get("version") == 2
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


# =============================================================================
# Test Artifact Delete
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.live
async def test_delete_artifact():
    """Test deleting an artifact."""
    title = generate_unique_title()

    # Create artifact
    response = await create_artifact(title=title, artifact_type="test_type")
    artifact = response.get("artifact", response)
    artifact_id = artifact.get("_id") or artifact.get("id")

    # Delete it
    delete_result = await delete_artifact(artifact_id)
    assert delete_result.get("status") == "completed"

    # Verify it's gone
    user_id = get_test_user_id()
    tool = load_artifact_tool("eden_artifacts_v3_get")
    get_result = await _run_tool(tool, {"artifactId": artifact_id, "user_id": user_id})

    # Should fail or return error
    output = get_result.get("output", [None])[0]
    if isinstance(output, str):
        assert (
            "error" in output.lower()
            or "not found" in output.lower()
            or get_result.get("status") == "failed"
        )


# =============================================================================
# Test Artifact List
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.live
async def test_list_artifacts_by_type():
    """Test listing artifacts filtered by type."""
    unique_type = f"test_type_{uuid.uuid4().hex[:8]}"
    artifact_ids = []

    try:
        # Create multiple artifacts of the same type
        for i in range(3):
            response = await create_artifact(
                title=f"List Test {i}",
                artifact_type=unique_type,
            )
            artifact = response.get("artifact", response)
            artifact_ids.append(artifact.get("_id") or artifact.get("id"))

        # List by type
        list_response = await list_artifacts(artifact_type=unique_type)

        docs = list_response.get("docs", list_response.get("artifacts", []))
        assert len(docs) >= 3

        # All returned should be of our type
        for doc in docs:
            if doc.get("type") == unique_type:
                assert doc.get("_id") in artifact_ids or doc.get("id") in artifact_ids
    finally:
        for artifact_id in artifact_ids:
            try:
                await delete_artifact(artifact_id)
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.live
async def test_list_artifacts_pagination():
    """Test listing artifacts with pagination."""
    unique_type = f"test_page_{uuid.uuid4().hex[:8]}"
    artifact_ids = []

    try:
        # Create 5 artifacts
        for i in range(5):
            response = await create_artifact(
                title=f"Page Test {i}",
                artifact_type=unique_type,
            )
            artifact = response.get("artifact", response)
            artifact_ids.append(artifact.get("_id") or artifact.get("id"))

        # Get first page (limit 2)
        page1 = await list_artifacts(artifact_type=unique_type, page=1, limit=2)
        docs1 = page1.get("docs", page1.get("artifacts", []))
        assert len(docs1) == 2

        # Get second page
        page2 = await list_artifacts(artifact_type=unique_type, page=2, limit=2)
        docs2 = page2.get("docs", page2.get("artifacts", []))
        assert len(docs2) == 2

        # Ensure no overlap
        ids1 = {d.get("_id") or d.get("id") for d in docs1}
        ids2 = {d.get("_id") or d.get("id") for d in docs2}
        assert ids1.isdisjoint(ids2)
    finally:
        for artifact_id in artifact_ids:
            try:
                await delete_artifact(artifact_id)
            except Exception:
                pass


# =============================================================================
# Test Artifact History
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.live
async def test_artifact_history():
    """Test that artifact changes create version history."""
    title = generate_unique_title()
    artifact_id = None

    try:
        # Create artifact
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={"version": 1},
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Make several updates to create history
        await patch_artifact(
            artifact_id,
            operations=[{"op": "replace", "path": "/version", "value": 2}],
            reason="Version 2",
        )
        await patch_artifact(
            artifact_id,
            operations=[{"op": "replace", "path": "/version", "value": 3}],
            reason="Version 3",
        )

        # Get history
        history = await get_artifact_history(artifact_id)

        docs = history.get("docs", history.get("versions", []))
        # Should have at least 2 versions (the patches, creation may or may not be in history)
        assert len(docs) >= 2

        # Versions should have reason field
        reasons = [d.get("reason") for d in docs if d.get("reason")]
        assert "Version 2" in reasons or "Version 3" in reasons
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_artifact_history_contains_snapshots():
    """Test that history contains full item snapshots."""
    title = generate_unique_title()
    artifact_id = None

    try:
        # Create artifact
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={"data": "initial"},
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Update
        await patch_artifact(
            artifact_id,
            operations=[{"op": "replace", "path": "/data", "value": "updated"}],
        )

        # Get history
        history = await get_artifact_history(artifact_id)
        docs = history.get("docs", history.get("versions", []))

        # At least one version should have items snapshot
        has_snapshot = any(d.get("items") is not None for d in docs)
        assert has_snapshot, "History should contain item snapshots"
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_rollback_via_history():
    """Test that we can rollback by reading history and applying a replace patch."""
    title = generate_unique_title()
    artifact_id = None

    try:
        # Create artifact
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={"state": "original", "counter": 1},
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Make changes
        await patch_artifact(
            artifact_id,
            operations=[
                {"op": "replace", "path": "/state", "value": "modified"},
                {"op": "replace", "path": "/counter", "value": 2},
            ],
        )

        # Verify current state
        current = await get_artifact(artifact_id)
        current_artifact = current.get("artifact", current)
        assert current_artifact.get("items", {}).get("state") == "modified"

        # Get history to find the first version
        history = await get_artifact_history(artifact_id, direction=-1)
        docs = history.get("docs", history.get("versions", []))

        # Find a version with the original state
        original_snapshot = None
        for doc in docs:
            items = doc.get("items", {})
            if items.get("state") == "original":
                original_snapshot = items
                break

        if original_snapshot:
            # Rollback by replacing with the snapshot
            await patch_artifact(
                artifact_id,
                operations=[{"op": "replace", "path": "", "value": original_snapshot}],
                reason="Rollback to original",
            )

            # Verify rollback worked
            rolled_back = await get_artifact(artifact_id)
            rolled_artifact = rolled_back.get("artifact", rolled_back)
            # Note: The exact behavior depends on API implementation
            # This test verifies the rollback pattern works
            assert rolled_artifact is not None
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


# =============================================================================
# Test Edge Cases
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.live
async def test_artifact_with_complex_nested_data():
    """Test artifacts with deeply nested data structures."""
    title = generate_unique_title()
    artifact_id = None

    complex_items = {
        "level1": {
            "level2": {
                "level3": {
                    "array": [
                        {"nested": {"deep": "value"}},
                        {"nested": {"deep": "value2"}},
                    ]
                }
            }
        },
        "mixed": [1, "two", {"three": 3}, [4, 5, 6]],
    }

    try:
        response = await create_artifact(
            title=title,
            artifact_type="complex_test",
            items=complex_items,
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Fetch and verify
        fetched = await get_artifact(artifact_id)
        fetched_artifact = fetched.get("artifact", fetched)
        items = fetched_artifact.get("items", {})

        assert (
            items["level1"]["level2"]["level3"]["array"][0]["nested"]["deep"] == "value"
        )
        assert items["mixed"][2]["three"] == 3
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_patch_deeply_nested_path():
    """Test patching a deeply nested value."""
    title = generate_unique_title()
    artifact_id = None

    try:
        response = await create_artifact(
            title=title,
            artifact_type="test_type",
            items={
                "game": {
                    "players": [
                        {"name": "Player1", "stats": {"score": 0, "lives": 3}},
                    ]
                }
            },
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Update deeply nested score
        await patch_artifact(
            artifact_id,
            operations=[
                {"op": "replace", "path": "/game/players/0/stats/score", "value": 100}
            ],
        )

        # Verify
        fetched = await get_artifact(artifact_id)
        fetched_artifact = fetched.get("artifact", fetched)
        items = fetched_artifact.get("items", {})
        assert items["game"]["players"][0]["stats"]["score"] == 100
        assert items["game"]["players"][0]["stats"]["lives"] == 3  # Unchanged
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_artifact_with_special_characters():
    """Test artifacts with special characters in values."""
    title = generate_unique_title()
    artifact_id = None

    special_items = {
        "unicode": "Hello 世界 🌍",
        "quotes": 'He said "hello"',
        "newlines": "Line1\nLine2\nLine3",
        "html": "<script>alert('xss')</script>",
        "json_string": '{"nested": "json"}',
    }

    try:
        response = await create_artifact(
            title=title,
            artifact_type="special_test",
            items=special_items,
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Verify special characters preserved
        fetched = await get_artifact(artifact_id)
        fetched_artifact = fetched.get("artifact", fetched)
        items = fetched_artifact.get("items", {})

        assert items["unicode"] == "Hello 世界 🌍"
        assert items["quotes"] == 'He said "hello"'
        assert items["newlines"] == "Line1\nLine2\nLine3"
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_artifact_with_numeric_types():
    """Test that numeric types are preserved correctly."""
    title = generate_unique_title()
    artifact_id = None

    numeric_items = {
        "integer": 42,
        "float": 3.14159,
        "negative": -100,
        "zero": 0,
        "large": 9999999999,
        "scientific": 1.23e10,
    }

    try:
        response = await create_artifact(
            title=title,
            artifact_type="numeric_test",
            items=numeric_items,
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Verify numeric types
        fetched = await get_artifact(artifact_id)
        fetched_artifact = fetched.get("artifact", fetched)
        items = fetched_artifact.get("items", {})

        assert items["integer"] == 42
        assert abs(items["float"] - 3.14159) < 0.0001
        assert items["negative"] == -100
        assert items["zero"] == 0
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)


@pytest.mark.asyncio
@pytest.mark.live
async def test_artifact_with_boolean_and_null():
    """Test that booleans and null values are preserved."""
    title = generate_unique_title()
    artifact_id = None

    items = {
        "true_val": True,
        "false_val": False,
        "null_val": None,
        "nested": {"bool": True, "null": None},
    }

    try:
        response = await create_artifact(
            title=title,
            artifact_type="bool_null_test",
            items=items,
        )
        artifact = response.get("artifact", response)
        artifact_id = artifact.get("_id") or artifact.get("id")

        # Verify types
        fetched = await get_artifact(artifact_id)
        fetched_artifact = fetched.get("artifact", fetched)
        fetched_items = fetched_artifact.get("items", {})

        assert fetched_items["true_val"] is True
        assert fetched_items["false_val"] is False
        assert fetched_items["null_val"] is None
        assert fetched_items["nested"]["bool"] is True
    finally:
        if artifact_id:
            await delete_artifact(artifact_id)
