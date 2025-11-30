"""
Test script for Verdelis Seed handler.
"""

import asyncio
from unittest.mock import patch


class MockToolContext:
    """Mock ToolContext for testing."""

    def __init__(self, args: dict, agent: str = None):
        self.args = args
        self.agent = agent


async def test_handler_valid_seed():
    """Test creating a valid seed."""
    from handler import handler

    context = MockToolContext(
        args={
            "title": "Test Seed",
            "logline": "A test idea about testing.",
            "agents": ["507f1f77bcf86cd799439011"],
            "images": ["https://example.com/image1.jpg"],
        },
        agent="507f1f77bcf86cd799439013",
    )

    # Mock the validation functions and save method
    with patch(
        "handler.validate_image_url",
        return_value=(True, {"format": "JPEG", "size": (1920, 1080)}),
    ):
        with patch("handler.VerdelisSeed.save"):
            result = await handler(context)

    print("Valid seed test result:")
    print(f"  Title: {result['output']['title']}")
    print(f"  Image count: {result['output']['image_count']}")
    print("  Status: PASSED")
    return result


async def test_handler_missing_title():
    """Test that missing title raises error."""
    from handler import handler

    context = MockToolContext(
        args={
            "logline": "A test idea.",
            "images": ["https://example.com/image.jpg"],
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

    context = MockToolContext(
        args={
            "title": "Test",
            "images": ["https://example.com/image.jpg"],
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


async def test_handler_missing_images():
    """Test that missing images raises error."""
    from handler import handler

    context = MockToolContext(
        args={
            "title": "Test",
            "logline": "A test idea.",
        }
    )

    try:
        await handler(context)
        print("Missing images test: FAILED (should have raised error)")
        return False
    except ValueError as e:
        if "images" in str(e).lower():
            print("Missing images test: PASSED")
            return True
        print(f"Missing images test: FAILED (wrong error: {e})")
        return False


async def test_handler_invalid_image():
    """Test that invalid image URLs raise error."""
    from handler import handler

    context = MockToolContext(
        args={
            "title": "Test",
            "logline": "A test idea.",
            "images": ["https://example.com/not-an-image.txt"],
        }
    )

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


async def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Verdelis Seed Handler Tests")
    print("=" * 60)

    tests = [
        test_handler_valid_seed,
        test_handler_missing_title,
        test_handler_missing_logline,
        test_handler_missing_images,
        test_handler_invalid_image,
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
