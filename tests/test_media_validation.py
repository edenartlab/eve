import pytest
import requests
from io import BytesIO
from PIL import Image

from eve.utils.file_utils import (
    validate_image_bytes,
    validate_video_bytes,
    validate_audio_bytes
)


class TestImageValidation:
    """Tests for image validation"""

    def test_valid_image_bytes(self):
        """Test validation of valid image bytes"""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        buf = BytesIO()
        img.save(buf, format='PNG')
        image_bytes = buf.getvalue()

        ok, info = validate_image_bytes(image_bytes)

        assert ok is True
        assert info['format'] == 'PNG'
        assert info['size'] == (100, 100)

    def test_invalid_image_bytes(self):
        """Test validation of invalid image bytes"""
        invalid_bytes = b"not an image file"

        ok, info = validate_image_bytes(invalid_bytes)

        assert ok is False
        assert 'reason' in info

    def test_empty_image_bytes(self):
        """Test validation of empty bytes"""
        ok, info = validate_image_bytes(b"")

        assert ok is False
        assert 'reason' in info

    def test_real_image_url(self):
        """Test validation with real image from URL"""
        url = "https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        ok, info = validate_image_bytes(response.content)

        assert ok is True
        assert info['format'] == 'JPEG'
        assert info['size'] == (512, 512)


class TestVideoValidation:
    """Tests for video validation"""

    def test_invalid_video_bytes(self):
        """Test validation of invalid video bytes"""
        invalid_bytes = b"not a video file"

        ok, info = validate_video_bytes(invalid_bytes)

        assert ok is False
        assert 'reason' in info
        assert 'ffprobe' in info['reason'].lower()

    def test_empty_video_bytes(self):
        """Test validation of empty bytes"""
        ok, info = validate_video_bytes(b"")

        assert ok is False
        assert 'reason' in info

    def test_real_video_url(self):
        """Test validation with real video from URL"""
        url = "https://dtut5r9j4w7j4.cloudfront.net/591517521621312417d5f305871b0d27a2d400bab0eb49fa18639af2b7027370.mp4"

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        ok, info = validate_video_bytes(response.content)

        assert ok is True
        assert 'format' in info
        assert 'duration' in info
        assert 'size' in info
        assert info['size'] == (1280, 720)
        assert info['duration'] > 0


class TestAudioValidation:
    """Tests for audio validation"""

    def test_invalid_audio_bytes(self):
        """Test validation of invalid audio bytes"""
        invalid_bytes = b"not an audio file"

        ok, info = validate_audio_bytes(invalid_bytes)

        assert ok is False
        assert 'reason' in info
        assert 'ffprobe' in info['reason'].lower()

    def test_empty_audio_bytes(self):
        """Test validation of empty bytes"""
        ok, info = validate_audio_bytes(b"")

        assert ok is False
        assert 'reason' in info

    def test_real_audio_url(self):
        """Test validation with real audio from URL"""
        url = "https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/8a23c248f3119713267730ce44a7b5c67eca2a6747a629b9fe2a8bc3ac2f03ec.mp3"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        ok, info = validate_audio_bytes(response.content)

        assert ok is True
        assert 'format' in info
        assert 'duration' in info
        assert 'sample_rate' in info
        assert 'channels' in info
        assert info['sample_rate'] == 44100
        assert info['channels'] == 1
        assert info['duration'] > 0


class TestMediaTypeDetection:
    """Tests for media type detection from URLs"""

    @pytest.mark.parametrize("url,expected_type", [
        ("https://example.com/image.jpg", "image"),
        ("https://example.com/photo.PNG", "image"),
        ("https://example.com/pic.jpeg", "image"),
        ("https://example.com/graphic.gif", "image"),
        ("https://example.com/video.mp4", "video"),
        ("https://example.com/clip.WEBM", "video"),
        ("https://example.com/movie.mov", "video"),
        ("https://example.com/song.mp3", "audio"),
        ("https://example.com/sound.WAV", "audio"),
        ("https://example.com/unknown.txt", "unknown"),
        ("https://example.com/file.pdf", "unknown"),
    ])
    def test_get_media_type(self, url, expected_type):
        """Test media type detection from URL extension"""
        from eve.tools.abraham.guardrails import get_media_type

        assert get_media_type(url) == expected_type


class TestMediaURLExtraction:
    """Tests for media URL extraction from markdown"""

    def test_extract_image_urls(self):
        """Test extraction of image URLs from markdown"""
        from eve.tools.abraham.guardrails import extract_media_urls

        markdown = """
        Check out this image: ![alt](https://example.com/image.jpg)
        And this one: https://example.com/photo.png
        """

        urls = extract_media_urls(markdown)

        assert len(urls) == 2
        assert "https://example.com/image.jpg" in urls
        assert "https://example.com/photo.png" in urls

    def test_extract_video_urls(self):
        """Test extraction of video URLs from markdown"""
        from eve.tools.abraham.guardrails import extract_media_urls

        markdown = """
        Watch this: [video](https://example.com/clip.mp4)
        Direct link: https://example.com/movie.webm
        """

        urls = extract_media_urls(markdown)

        assert len(urls) == 2
        assert "https://example.com/clip.mp4" in urls
        assert "https://example.com/movie.webm" in urls

    def test_extract_audio_urls(self):
        """Test extraction of audio URLs from markdown"""
        from eve.tools.abraham.guardrails import extract_media_urls

        markdown = """
        Listen: https://example.com/song.mp3
        More: [audio](https://example.com/sound.wav)
        """

        urls = extract_media_urls(markdown)

        assert len(urls) == 2
        assert "https://example.com/song.mp3" in urls
        assert "https://example.com/sound.wav" in urls

    def test_extract_mixed_media_urls(self):
        """Test extraction of mixed media URLs from markdown"""
        from eve.tools.abraham.guardrails import extract_media_urls

        markdown = """
        ![image](https://example.com/photo.jpg)
        [video](https://example.com/clip.mp4)
        https://example.com/song.mp3
        <img src="https://example.com/pic.png">
        <video src="https://example.com/movie.webm"></video>
        """

        urls = extract_media_urls(markdown)

        assert len(urls) == 5
        assert "https://example.com/photo.jpg" in urls
        assert "https://example.com/clip.mp4" in urls
        assert "https://example.com/song.mp3" in urls
        assert "https://example.com/pic.png" in urls
        assert "https://example.com/movie.webm" in urls

    def test_extract_no_duplicates(self):
        """Test that duplicate URLs are removed"""
        from eve.tools.abraham.guardrails import extract_media_urls

        markdown = """
        https://example.com/image.jpg
        ![alt](https://example.com/image.jpg)
        https://example.com/image.jpg
        """

        urls = extract_media_urls(markdown)

        assert len(urls) == 1
        assert urls[0] == "https://example.com/image.jpg"


class TestRealWorldURLs:
    """Integration tests with real-world URLs"""

    @pytest.mark.parametrize("url,media_type,should_pass", [
        # Should succeed
        ("https://dtut5r9j4w7j4.cloudfront.net/591517521621312417d5f305871b0d27a2d400bab0eb49fa18639af2b7027370.mp4", "video", True),
        ("https://edenartlab-stage-data.s3.amazonaws.com/61ccedc87dd9689b2714daebbd851a37b6f74cd5dc3a16dc0b8267a8b535db04.jpg", "image", True),
        ("https://edenartlab-stage-data.s3.us-east-1.amazonaws.com/8a23c248f3119713267730ce44a7b5c67eca2a6747a629b9fe2a8bc3ac2f03ec.mp3", "audio", True),
        # Should fail
        ("https://dtut5r9j4w7j4.cloudfront.net/fake.mp4", "video", False),
        ("https://dtut5r9j4w7j4.cloudfront.net/fake.png", "image", False),
        ("https://dtut5r9j4w7j4.cloudfront.net/fake.mp3", "audio", False),
    ])
    def test_real_world_urls(self, url, media_type, should_pass):
        """Test validation with real-world URLs"""
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            content_bytes = response.content

            # Validate based on media type
            if media_type == "image":
                ok, info = validate_image_bytes(content_bytes)
            elif media_type == "video":
                ok, info = validate_video_bytes(content_bytes)
            elif media_type == "audio":
                ok, info = validate_audio_bytes(content_bytes)
            else:
                ok = False

            if should_pass:
                assert ok is True, f"Expected {url} to pass validation but it failed: {info}"
            else:
                assert ok is False, f"Expected {url} to fail validation but it passed"

        except requests.exceptions.HTTPError:
            # HTTP errors should occur for fake URLs
            if should_pass:
                pytest.fail(f"Expected {url} to succeed but got HTTP error")
            # Test passes - we expected it to fail

    def test_non_media_url_fails(self):
        """Test that non-media URLs fail validation"""
        url = "https://replicate.com/edenartlab/sdxl-pipelines/llms.txt"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Should fail for all media types
        ok_image, _ = validate_image_bytes(response.content)
        ok_video, _ = validate_video_bytes(response.content)
        ok_audio, _ = validate_audio_bytes(response.content)

        assert ok_image is False
        assert ok_video is False
        assert ok_audio is False
