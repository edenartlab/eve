"""Tests for the Artifact system."""

import json
from datetime import datetime, timezone

import pytest
from bson import ObjectId

from eve.artifact import (
    ARTIFACT_SCHEMAS,
    MEDIUM_ARTIFACT_THRESHOLD,
    SMALL_ARTIFACT_THRESHOLD,
    Artifact,
    ArtifactOperation,
    ArtifactValidationError,
    ArtifactVersion,
    validate_artifact_data,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def owner_id():
    return ObjectId()


@pytest.fixture
def session_id():
    return ObjectId()


@pytest.fixture
def basic_artifact(owner_id, session_id):
    """Create a basic artifact for testing."""
    artifact = Artifact(
        type="screenplay_storyboard",
        name="Test Screenplay",
        description="A test screenplay",
        owner=owner_id,
        session=session_id,
        data={
            "title": "The Journey",
            "scenes": [
                {"image_prompt": "A forest", "description": "Opening scene", "order": 1}
            ],
        },
    )
    artifact.save()
    return artifact


@pytest.fixture
def character_bible_artifact(owner_id):
    """Create a character bible artifact for testing."""
    artifact = Artifact(
        type="character_bible",
        name="Main Character",
        owner=owner_id,
        data={
            "name": "Hero",
            "traits": ["brave", "kind"],
            "backstory": "A humble beginning",
            "relationships": [{"character": "Mentor", "relationship": "teacher"}],
        },
    )
    artifact.save()
    return artifact


# =============================================================================
# Test Artifact Creation and Basic Operations
# =============================================================================


def test_artifact_creation(owner_id):
    """Test creating a new artifact."""
    artifact = Artifact(
        type="project_plan",
        name="Test Project",
        owner=owner_id,
        data={"title": "My Project", "goals": ["Goal 1"]},
    )
    artifact.save()

    assert artifact.id is not None
    assert artifact.type == "project_plan"
    assert artifact.name == "Test Project"
    assert artifact.version == 1
    assert artifact.data["title"] == "My Project"


def test_artifact_creation_with_string_ids():
    """Test that string IDs are converted to ObjectId."""
    owner_str = str(ObjectId())
    session_str = str(ObjectId())

    artifact = Artifact(
        type="test",
        name="Test",
        owner=owner_str,
        session=session_str,
        data={},
    )

    assert isinstance(artifact.owner, ObjectId)
    assert isinstance(artifact.session, ObjectId)


def test_artifact_from_mongo(basic_artifact):
    """Test loading artifact from database."""
    loaded = Artifact.from_mongo(basic_artifact.id)

    assert loaded.id == basic_artifact.id
    assert loaded.name == basic_artifact.name
    assert loaded.type == basic_artifact.type
    assert loaded.data == basic_artifact.data


# =============================================================================
# Test Structured Operations
# =============================================================================


def test_set_operation(basic_artifact):
    """Test the 'set' operation."""
    basic_artifact.apply_operations(
        [{"op": "set", "path": "title", "value": "New Title"}],
        save=False,
    )

    assert basic_artifact.data["title"] == "New Title"
    assert basic_artifact.version == 2


def test_set_nested_operation(basic_artifact):
    """Test 'set' operation on nested path."""
    basic_artifact.apply_operations(
        [{"op": "set", "path": "metadata.author", "value": "John Doe"}],
        save=False,
    )

    assert basic_artifact.data["metadata"]["author"] == "John Doe"


def test_append_operation(basic_artifact):
    """Test the 'append' operation."""
    new_scene = {"image_prompt": "A mountain", "description": "Scene 2", "order": 2}
    basic_artifact.apply_operations(
        [{"op": "append", "path": "scenes", "value": new_scene}],
        save=False,
    )

    assert len(basic_artifact.data["scenes"]) == 2
    assert basic_artifact.data["scenes"][1]["description"] == "Scene 2"


def test_append_creates_array_if_missing(basic_artifact):
    """Test that append creates the array if it doesn't exist."""
    basic_artifact.apply_operations(
        [{"op": "append", "path": "tags", "value": "action"}],
        save=False,
    )

    assert basic_artifact.data["tags"] == ["action"]


def test_insert_operation(basic_artifact):
    """Test the 'insert' operation."""
    # First add another scene
    basic_artifact.apply_operations(
        [{"op": "append", "path": "scenes", "value": {"image_prompt": "End", "description": "End scene", "order": 3}}],
        save=False,
    )

    # Insert in the middle
    basic_artifact.apply_operations(
        [{"op": "insert", "path": "scenes", "index": 1, "value": {"image_prompt": "Middle", "description": "Middle scene", "order": 2}}],
        save=False,
    )

    assert len(basic_artifact.data["scenes"]) == 3
    assert basic_artifact.data["scenes"][1]["description"] == "Middle scene"


def test_remove_operation(basic_artifact):
    """Test the 'remove' operation."""
    # Add a scene first
    basic_artifact.apply_operations(
        [{"op": "append", "path": "scenes", "value": {"image_prompt": "Remove me", "description": "To be removed", "order": 2}}],
        save=False,
    )
    assert len(basic_artifact.data["scenes"]) == 2

    # Remove it
    basic_artifact.apply_operations(
        [{"op": "remove", "path": "scenes", "index": 1}],
        save=False,
    )

    assert len(basic_artifact.data["scenes"]) == 1


def test_update_operation(basic_artifact):
    """Test the 'update' operation for merging dict values."""
    basic_artifact.apply_operations(
        [{"op": "update", "path": "scenes", "index": 0, "value": {"duration": 5.0}}],
        save=False,
    )

    assert basic_artifact.data["scenes"][0]["duration"] == 5.0
    assert basic_artifact.data["scenes"][0]["description"] == "Opening scene"  # Original preserved


def test_replace_operation(basic_artifact):
    """Test the 'replace' operation."""
    new_data = {"title": "Completely New", "scenes": []}
    basic_artifact.apply_operations(
        [{"op": "replace", "data": new_data}],
        save=False,
    )

    assert basic_artifact.data == new_data


def test_multiple_operations_atomic(basic_artifact):
    """Test applying multiple operations in one call."""
    basic_artifact.apply_operations(
        [
            {"op": "set", "path": "title", "value": "Updated Title"},
            {"op": "append", "path": "scenes", "value": {"image_prompt": "New", "description": "New scene", "order": 2}},
            {"op": "set", "path": "narration", "value": "Once upon a time..."},
        ],
        save=False,
    )

    assert basic_artifact.data["title"] == "Updated Title"
    assert len(basic_artifact.data["scenes"]) == 2
    assert basic_artifact.data["narration"] == "Once upon a time..."
    # All operations create one version bump
    assert basic_artifact.version == 2


# =============================================================================
# Test Version History
# =============================================================================


def test_version_increments(basic_artifact):
    """Test that version increments with each operation set."""
    assert basic_artifact.version == 1

    basic_artifact.apply_operations(
        [{"op": "set", "path": "title", "value": "V2"}],
        save=False,
    )
    assert basic_artifact.version == 2

    basic_artifact.apply_operations(
        [{"op": "set", "path": "title", "value": "V3"}],
        save=False,
    )
    assert basic_artifact.version == 3


def test_version_history_stored(basic_artifact):
    """Test that version history is stored correctly."""
    basic_artifact.apply_operations(
        [{"op": "set", "path": "title", "value": "Version 2 Title"}],
        actor_type="user",
        message="Updated title",
        save=False,
    )

    assert len(basic_artifact.versions) == 1
    version = basic_artifact.versions[0]
    assert version.version == 2
    assert version.message == "Updated title"
    assert version.actor_type == "user"
    assert version.data_snapshot["title"] == "Version 2 Title"


def test_rollback_to_version(basic_artifact):
    """Test rolling back to a previous version."""
    original_title = basic_artifact.data["title"]

    # Make changes
    basic_artifact.apply_operations(
        [{"op": "set", "path": "title", "value": "Changed"}],
        save=False,
    )
    basic_artifact.apply_operations(
        [{"op": "set", "path": "title", "value": "Changed Again"}],
        save=False,
    )

    # Rollback to version 2
    success = basic_artifact.rollback_to_version(2)
    assert success
    assert basic_artifact.data["title"] == "Changed"


def test_max_versions_limit(owner_id):
    """Test that old versions are pruned when max is reached."""
    artifact = Artifact(
        type="test",
        name="Test",
        owner=owner_id,
        data={"value": 0},
        max_versions=5,
    )
    artifact.save()

    # Apply more operations than max_versions
    for i in range(10):
        artifact.apply_operations(
            [{"op": "set", "path": "value", "value": i + 1}],
            save=False,
        )

    assert len(artifact.versions) == 5
    assert artifact.versions[0].version == 7  # Oldest kept version


# =============================================================================
# Test Type Validation
# =============================================================================


def test_validate_screenplay_storyboard_valid():
    """Test validation passes for valid screenplay_storyboard data."""
    data = {
        "title": "My Movie",
        "scenes": [
            {"image_prompt": "Scene 1", "description": "Opening", "order": 1}
        ],
    }
    is_valid, errors = validate_artifact_data("screenplay_storyboard", data)
    assert is_valid
    assert errors is None


def test_validate_screenplay_storyboard_missing_required():
    """Test validation fails for missing required fields."""
    data = {"scenes": []}  # Missing 'title'
    is_valid, errors = validate_artifact_data("screenplay_storyboard", data)
    assert not is_valid
    assert errors is not None
    assert any("title" in str(e) for e in errors)


def test_validate_character_bible_valid():
    """Test validation passes for valid character_bible data."""
    data = {
        "name": "Hero",
        "traits": ["brave"],
        "backstory": "A long story",
    }
    is_valid, errors = validate_artifact_data("character_bible", data)
    assert is_valid


def test_validate_unknown_type_allows_anything():
    """Test that unknown artifact types pass validation."""
    data = {"anything": "goes", "random": [1, 2, 3]}
    is_valid, errors = validate_artifact_data("unknown_type", data)
    assert is_valid
    assert errors is None


def test_validate_raises_on_error():
    """Test that validation can raise an exception."""
    data = {"scenes": []}  # Missing 'title'
    with pytest.raises(ArtifactValidationError) as exc_info:
        validate_artifact_data("screenplay_storyboard", data, raise_on_error=True)

    assert exc_info.value.artifact_type == "screenplay_storyboard"
    assert exc_info.value.errors is not None


def test_artifact_validate_data_method(basic_artifact):
    """Test the artifact's validate_data method."""
    is_valid, errors = basic_artifact.validate_data()
    assert is_valid  # Our fixture has valid data


def test_extra_fields_allowed():
    """Test that extra fields beyond schema are allowed."""
    data = {
        "title": "My Movie",
        "scenes": [],
        "custom_field": "This should be allowed",
    }
    is_valid, errors = validate_artifact_data("screenplay_storyboard", data)
    assert is_valid


# =============================================================================
# Test Size and Context Methods
# =============================================================================


def test_get_data_size(basic_artifact):
    """Test data size calculation."""
    size = basic_artifact.get_data_size()
    assert size > 0
    assert size == len(json.dumps(basic_artifact.data, default=str))


def test_is_small_artifact(owner_id):
    """Test small artifact detection."""
    artifact = Artifact(
        type="test",
        name="Small",
        owner=owner_id,
        data={"simple": "data"},
    )
    assert artifact.is_small()
    assert not artifact.is_medium()


def test_is_medium_artifact(owner_id):
    """Test medium artifact detection."""
    # Create data that's between SMALL and MEDIUM thresholds
    large_text = "x" * (SMALL_ARTIFACT_THRESHOLD + 100)
    artifact = Artifact(
        type="test",
        name="Medium",
        owner=owner_id,
        data={"content": large_text},
    )
    assert not artifact.is_small()
    assert artifact.is_medium()


def test_get_context_data_small(owner_id):
    """Test context data for small artifacts returns full data."""
    artifact = Artifact(
        type="test",
        name="Small",
        owner=owner_id,
        data={"key": "value"},
    )
    context_data = artifact.get_context_data()
    assert context_data == artifact.data


def test_get_context_data_medium_truncates(owner_id):
    """Test context data for medium artifacts truncates arrays."""
    artifact = Artifact(
        type="test",
        name="Medium",
        owner=owner_id,
        data={
            "items": list(range(10)),  # Array with 10 items
            "text": "short",
        },
    )
    # Make it medium-sized
    artifact.data["padding"] = "x" * (SMALL_ARTIFACT_THRESHOLD + 100)

    context_data = artifact.get_context_data()

    # Should truncate the array
    assert len(context_data["items"]) == 3
    assert context_data["_items_total"] == 10


def test_get_context_data_large_returns_none(owner_id):
    """Test context data for large artifacts returns None."""
    # Create data larger than MEDIUM threshold
    artifact = Artifact(
        type="test",
        name="Large",
        owner=owner_id,
        data={"content": "x" * (MEDIUM_ARTIFACT_THRESHOLD + 1000)},
    )
    context_data = artifact.get_context_data()
    assert context_data is None


def test_get_summary(basic_artifact):
    """Test summary generation."""
    summary = basic_artifact.get_summary()
    assert basic_artifact.name in summary
    assert basic_artifact.type in summary
    assert "Version:" in summary


def test_to_context_dict(basic_artifact):
    """Test context dict generation."""
    context = basic_artifact.to_context_dict()
    assert context["artifact_id"] == str(basic_artifact.id)
    assert context["type"] == basic_artifact.type
    assert "data" not in context

    context_with_data = basic_artifact.to_context_dict(include_data=True)
    assert "data" in context_with_data
    assert context_with_data["data"] == basic_artifact.data


# =============================================================================
# Test Session Linking
# =============================================================================


def test_find_for_session(basic_artifact, session_id):
    """Test finding artifacts for a session."""
    artifacts = Artifact.find_for_session(session_id)
    assert len(artifacts) >= 1
    assert any(a.id == basic_artifact.id for a in artifacts)


def test_find_for_user(basic_artifact, owner_id):
    """Test finding artifacts for a user."""
    artifacts = Artifact.find_for_user(owner_id)
    assert len(artifacts) >= 1
    assert any(a.id == basic_artifact.id for a in artifacts)


def test_find_for_user_with_type_filter(basic_artifact, character_bible_artifact, owner_id):
    """Test finding artifacts filtered by type."""
    screenplays = Artifact.find_for_user(owner_id, artifact_type="screenplay_storyboard")
    assert all(a.type == "screenplay_storyboard" for a in screenplays)


def test_archive_and_restore(basic_artifact):
    """Test archiving and restoring artifacts."""
    artifact_id = basic_artifact.id

    # Archive
    basic_artifact.archive()
    basic_artifact.reload()
    assert basic_artifact.archived

    # Should not appear in normal queries
    artifacts = Artifact.find_for_session(basic_artifact.session, include_archived=False)
    assert not any(a.id == artifact_id for a in artifacts)

    # Should appear with include_archived
    artifacts = Artifact.find_for_session(basic_artifact.session, include_archived=True)
    assert any(a.id == artifact_id for a in artifacts)

    # Restore
    basic_artifact.restore()
    basic_artifact.reload()
    assert not basic_artifact.archived


# =============================================================================
# Test ArtifactOperation Model
# =============================================================================


def test_artifact_operation_from_dict():
    """Test creating ArtifactOperation from dict."""
    op = ArtifactOperation(**{
        "op": "set",
        "path": "title",
        "value": "New Title",
    })
    assert op.op == "set"
    assert op.path == "title"
    assert op.value == "New Title"


def test_artifact_operation_replace():
    """Test replace operation model."""
    op = ArtifactOperation(**{
        "op": "replace",
        "data": {"new": "data"},
    })
    assert op.op == "replace"
    assert op.data == {"new": "data"}


# =============================================================================
# Test Edge Cases
# =============================================================================


def test_operations_with_no_changes(basic_artifact):
    """Test that invalid operations don't create versions."""
    initial_version = basic_artifact.version

    # This operation should fail (remove from empty index)
    basic_artifact.apply_operations(
        [{"op": "remove", "path": "scenes", "index": 999}],
        save=False,
    )

    # Version should not change
    assert basic_artifact.version == initial_version


def test_nested_array_access(owner_id):
    """Test accessing nested arrays via dot notation."""
    artifact = Artifact(
        type="test",
        name="Test",
        owner=owner_id,
        data={
            "matrix": [[1, 2], [3, 4]],
        },
    )

    value = artifact._get_nested_value("matrix.0.1")
    assert value == 2


def test_empty_operations_list(basic_artifact):
    """Test applying empty operations list."""
    initial_version = basic_artifact.version

    basic_artifact.apply_operations([], save=False)

    # Version should not change
    assert basic_artifact.version == initial_version
