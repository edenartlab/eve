"""
Test script for Verdelis Storyboard handler.
"""

import asyncio
from unittest.mock import patch

from bson import ObjectId


class MockToolContext:
    """Mock ToolContext for testing."""

    def __init__(self, args: dict, session: str = None, agent: str = None):
        self.args = args
        self.session = ObjectId(session) if session else ObjectId()
        self.agent = agent


class MockVerdelisSeed:
    """Mock VerdelisSeed for testing."""

    def __init__(self, title="Test Seed"):
        self.id = ObjectId()
        self.title = title
        self.logline = "A test logline"
        self.agents = []
        self.images = []


async def test_handler_valid_storyboard():
    """Test creating a valid storyboard."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "title": "Test Storyboard",
            "logline": "A test story about testing.",
            "plot": "This is a detailed plot about a developer who tests their code.",
            "agents": ["507f1f77bcf86cd799439011"],
            "image_frames": ["https://example.com/image1.jpg"],
        },
        session="507f1f77bcf86cd799439012",
        agent="507f1f77bcf86cd799439013",
    )

    # Mock the validation functions, seed lookup, and save method
    with patch("handler.VerdelisSeed.find_one", return_value=mock_seed):
        with patch(
            "handler.validate_image_url",
            return_value=(True, {"format": "JPEG", "size": (1920, 1080)}),
        ):
            with patch("handler.VerdelisStoryboard.save"):
                result = await handler(context)

    print("Valid storyboard test result:")
    print(f"  Title: {result['output']['title']}")
    print(f"  Seed ID: {result['output']['seed_id']}")
    print(f"  Frame count: {result['output']['frame_count']}")
    print(f"  Has music: {result['output']['has_music']}")
    print(f"  Has vocals: {result['output']['has_vocals']}")
    print("  Status: PASSED")
    return result


async def test_handler_with_audio():
    """Test creating a storyboard with music and vocals."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "title": "Test Storyboard with Audio",
            "logline": "A test story with sound.",
            "plot": "This story has background music and narration.",
            "agents": [],
            "image_frames": ["https://example.com/image1.jpg"],
            "music": "https://example.com/music.mp3",
            "vocals": "https://example.com/vocals.mp3",
        },
        session="507f1f77bcf86cd799439012",
    )

    # Mock the validation functions, seed lookup, and save method
    with patch("handler.VerdelisSeed.find_one", return_value=mock_seed):
        with patch(
            "handler.validate_image_url",
            return_value=(True, {"format": "JPEG", "size": (1920, 1080)}),
        ):
            with patch(
                "handler.validate_audio_url",
                return_value=(True, {"format": "mp3", "duration": 120.5}),
            ):
                with patch("handler.VerdelisStoryboard.save"):
                    result = await handler(context)

    print("Storyboard with audio test result:")
    print(f"  Title: {result['output']['title']}")
    print(f"  Has music: {result['output']['has_music']}")
    print(f"  Has vocals: {result['output']['has_vocals']}")
    assert result["output"]["has_music"] is True
    assert result["output"]["has_vocals"] is True
    print("  Status: PASSED")
    return result


async def test_handler_missing_seed():
    """Test that missing seed raises error."""
    from handler import handler

    context = MockToolContext(
        args={
            "title": "Test",
            "logline": "A test story.",
            "plot": "Plot details.",
            "image_frames": ["https://example.com/image.jpg"],
        }
    )

    try:
        await handler(context)
        print("Missing seed test: FAILED (should have raised error)")
        return False
    except ValueError as e:
        if "seed" in str(e).lower():
            print("Missing seed test: PASSED")
            return True
        print(f"Missing seed test: FAILED (wrong error: {e})")
        return False


async def test_handler_invalid_seed():
    """Test that invalid seed ID raises error."""
    from handler import handler

    context = MockToolContext(
        args={
            "seed": "507f1f77bcf86cd799439099",  # Valid format but doesn't exist
            "title": "Test",
            "logline": "A test story.",
            "plot": "Plot details.",
            "image_frames": ["https://example.com/image.jpg"],
        }
    )

    # Mock find_one to return None (seed not found)
    with patch("handler.VerdelisSeed.find_one", return_value=None):
        try:
            await handler(context)
            print("Invalid seed test: FAILED (should have raised error)")
            return False
        except ValueError as e:
            if "seed not found" in str(e).lower():
                print("Invalid seed test: PASSED")
                return True
            print(f"Invalid seed test: FAILED (wrong error: {e})")
            return False


async def test_handler_missing_title():
    """Test that missing title raises error."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "logline": "A test story.",
            "plot": "Plot details.",
            "image_frames": ["https://example.com/image.jpg"],
        }
    )

    try:
        await handler(context)
        print("Missing title test: FAILED (should have raised error)")
        return False
    except ValueError as e:
        if "title" in str(e).lower():
            print("Missing title test: PASSED")
            return True
        print(f"Missing title test: FAILED (wrong error: {e})")
        return False


async def test_handler_missing_logline():
    """Test that missing logline raises error."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "title": "Test",
            "plot": "Plot details.",
            "image_frames": ["https://example.com/image.jpg"],
        }
    )

    try:
        await handler(context)
        print("Missing logline test: FAILED (should have raised error)")
        return False
    except ValueError as e:
        if "logline" in str(e).lower():
            print("Missing logline test: PASSED")
            return True
        print(f"Missing logline test: FAILED (wrong error: {e})")
        return False


async def test_handler_missing_plot():
    """Test that missing plot raises error."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "title": "Test",
            "logline": "A test story.",
            "image_frames": ["https://example.com/image.jpg"],
        }
    )

    try:
        await handler(context)
        print("Missing plot test: FAILED (should have raised error)")
        return False
    except ValueError as e:
        if "plot" in str(e).lower():
            print("Missing plot test: PASSED")
            return True
        print(f"Missing plot test: FAILED (wrong error: {e})")
        return False


async def test_handler_missing_image_frames():
    """Test that missing image_frames raises error."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "title": "Test",
            "logline": "A test story.",
            "plot": "Plot details.",
        }
    )

    try:
        await handler(context)
        print("Missing image_frames test: FAILED (should have raised error)")
        return False
    except ValueError as e:
        if "image_frames" in str(e).lower():
            print("Missing image_frames test: PASSED")
            return True
        print(f"Missing image_frames test: FAILED (wrong error: {e})")
        return False


async def test_handler_invalid_image():
    """Test that invalid image URLs raise error."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "title": "Test",
            "logline": "A test story.",
            "plot": "Plot details.",
            "image_frames": ["https://example.com/not-an-image.txt"],
        }
    )

    with patch("handler.VerdelisSeed.find_one", return_value=mock_seed):
        with patch(
            "handler.validate_image_url",
            return_value=(False, {"reason": "Unrecognized image format"}),
        ):
            try:
                await handler(context)
                print("Invalid image test: FAILED (should have raised error)")
                return False
            except ValueError as e:
                if "invalid" in str(e).lower():
                    print("Invalid image test: PASSED")
                    return True
                print(f"Invalid image test: FAILED (wrong error: {e})")
                return False


async def test_handler_invalid_music():
    """Test that invalid music URL raises error."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "title": "Test",
            "logline": "A test story.",
            "plot": "Plot details.",
            "image_frames": ["https://example.com/image.jpg"],
            "music": "https://example.com/not-audio.txt",
        }
    )

    with patch("handler.VerdelisSeed.find_one", return_value=mock_seed):
        with patch(
            "handler.validate_image_url",
            return_value=(True, {"format": "JPEG", "size": (1920, 1080)}),
        ):
            with patch(
                "handler.validate_audio_url",
                return_value=(False, {"reason": "No audio stream found"}),
            ):
                try:
                    await handler(context)
                    print("Invalid music test: FAILED (should have raised error)")
                    return False
                except ValueError as e:
                    if "music" in str(e).lower():
                        print("Invalid music test: PASSED")
                        return True
                    print(f"Invalid music test: FAILED (wrong error: {e})")
                    return False


async def test_handler_invalid_vocals():
    """Test that invalid vocals URL raises error."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "title": "Test",
            "logline": "A test story.",
            "plot": "Plot details.",
            "image_frames": ["https://example.com/image.jpg"],
            "vocals": "https://example.com/not-audio.txt",
        }
    )

    with patch("handler.VerdelisSeed.find_one", return_value=mock_seed):
        with patch(
            "handler.validate_image_url",
            return_value=(True, {"format": "JPEG", "size": (1920, 1080)}),
        ):
            with patch(
                "handler.validate_audio_url",
                return_value=(False, {"reason": "No audio stream found"}),
            ):
                try:
                    await handler(context)
                    print("Invalid vocals test: FAILED (should have raised error)")
                    return False
                except ValueError as e:
                    if "vocals" in str(e).lower():
                        print("Invalid vocals test: PASSED")
                        return True
                    print(f"Invalid vocals test: FAILED (wrong error: {e})")
                    return False


async def test_handler_missing_session():
    """Test that missing session raises error."""
    from handler import handler

    mock_seed = MockVerdelisSeed()

    context = MockToolContext(
        args={
            "seed": str(mock_seed.id),
            "title": "Test",
            "logline": "A test story.",
            "plot": "Plot details.",
            "image_frames": ["https://example.com/image.jpg"],
        }
    )
    context.session = None

    try:
        await handler(context)
        print("Missing session test: FAILED (should have raised error)")
        return False
    except ValueError as e:
        if "session" in str(e).lower():
            print("Missing session test: PASSED")
            return True
        print(f"Missing session test: FAILED (wrong error: {e})")
        return False


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Verdelis Storyboard Handler Tests")
    print("=" * 60)

    tests = [
        test_handler_valid_storyboard,
        test_handler_with_audio,
        test_handler_missing_seed,
        test_handler_invalid_seed,
        test_handler_missing_title,
        test_handler_missing_logline,
        test_handler_missing_plot,
        test_handler_missing_image_frames,
        test_handler_invalid_image,
        test_handler_invalid_music,
        test_handler_invalid_vocals,
        test_handler_missing_session,
    ]

    results = []
    for test in tests:
        print(f"\nRunning: {test.__name__}")
        try:
            result = await test()
            results.append((test.__name__, True if result else False))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((test.__name__, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    failed = len(results) - passed
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed tests:")
        for name, result in results:
            if not result:
                print(f"  - {name}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
