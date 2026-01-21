"""
End-to-end tests for readtube.
These tests use real YouTube data (with caching) to verify the full pipeline.

Note: These tests make real network requests. Run with:
    pytest tests/test_e2e.py -v -s
"""

import os
import pytest
import tempfile
import shutil

# Test video: Short, has captions, unlikely to be removed
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
TEST_VIDEO_ID = "dQw4w9WgXcQ"


class TestEndToEndPipeline:
    """End-to-end tests for the complete pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    @pytest.mark.slow
    def test_full_pipeline_single_video(self):
        """Test the complete pipeline: fetch video -> get transcript -> create epub."""
        from get_videos import get_video_info
        from get_transcripts import get_transcript
        from create_epub import create_epub_book

        # Step 1: Fetch video info
        video = get_video_info(TEST_VIDEO_URL)

        assert video is not None
        assert video["video_id"] == TEST_VIDEO_ID
        assert video["title"]  # Has a title
        assert video["channel"]  # Has a channel
        assert video.get("thumbnail")  # Has thumbnail

        # Step 2: Get transcript
        transcript = get_transcript(video["video_id"])

        assert transcript is not None
        assert len(transcript) > 100  # Has substantial content

        # Step 3: Create article (simulated - normally Claude writes this)
        article = {
            "title": video["title"],
            "channel": video["channel"],
            "url": video["url"],
            "thumbnail": video.get("thumbnail"),
            "article": f"""# {video['title']}

This is a test article generated from the video transcript.

## Overview

{transcript[:500]}...

## Conclusion

The video covers interesting content about music and culture.
"""
        }

        # Step 4: Create EPUB
        output_path = os.path.join(self.temp_dir, "test_e2e.epub")
        result = create_epub_book([article], output_path=output_path)

        assert os.path.exists(result)
        assert os.path.getsize(result) > 1000  # Reasonable file size

    @pytest.mark.slow
    def test_video_info_extraction(self):
        """Test that we can extract complete video info."""
        from get_videos import get_video_info

        video = get_video_info(TEST_VIDEO_URL)

        # Verify all expected fields
        assert "video_id" in video
        assert "title" in video
        assert "description" in video
        assert "channel" in video
        assert "url" in video
        assert "thumbnail" in video
        assert "duration" in video

        # Verify content
        assert video["video_id"] == TEST_VIDEO_ID
        assert len(video["title"]) > 0
        assert video["duration"] > 0

    @pytest.mark.slow
    def test_transcript_extraction(self):
        """Test that we can extract video transcript."""
        from get_transcripts import get_transcript

        transcript = get_transcript(TEST_VIDEO_ID)

        assert transcript is not None
        assert isinstance(transcript, str)
        assert len(transcript) > 100

    @pytest.mark.slow
    def test_transcript_language_listing(self):
        """Test listing available transcript languages."""
        from get_transcripts import list_available_languages

        languages = list_available_languages(TEST_VIDEO_ID)

        assert isinstance(languages, list)
        # Most videos have at least auto-generated English
        assert len(languages) > 0

        # Check structure
        if languages:
            lang = languages[0]
            assert "code" in lang
            assert "name" in lang

    @pytest.mark.slow
    def test_transcript_caching(self):
        """Test that transcripts are cached correctly."""
        from get_transcripts import get_transcript, load_from_cache, clear_cache
        import time

        # Clear cache first
        clear_cache()

        # First fetch - should hit network
        start = time.time()
        transcript1 = get_transcript(TEST_VIDEO_ID, use_cache=True)
        first_fetch_time = time.time() - start

        # Second fetch - should use cache
        start = time.time()
        transcript2 = get_transcript(TEST_VIDEO_ID, use_cache=True)
        cached_fetch_time = time.time() - start

        assert transcript1 == transcript2
        # Cached fetch should be significantly faster
        assert cached_fetch_time < first_fetch_time

    def test_epub_creation_with_cover(self):
        """Test EPUB creation with thumbnail cover."""
        from create_epub import create_epub_book

        article = {
            "title": "Test Article",
            "channel": "Test Channel",
            "url": "https://example.com",
            "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
            "article": "# Test\n\nThis is test content."
        }

        output_path = os.path.join(self.temp_dir, "test_cover.epub")
        result = create_epub_book([article], output_path=output_path, include_cover=True)

        assert os.path.exists(result)

    def test_html_output(self):
        """Test HTML output generation."""
        from create_epub import create_html

        article = {
            "title": "Test Article",
            "channel": "Test Channel",
            "url": "https://example.com",
            "article": "# Test Heading\n\nThis is a **test** paragraph.\n\n## Section\n\nMore content here."
        }

        output_path = os.path.join(self.temp_dir, "test.html")
        result = create_html([article], output_path=output_path)

        assert os.path.exists(result)

        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check HTML structure
        assert "<!DOCTYPE html>" in content
        assert "Test Heading" in content
        assert "<strong>test</strong>" in content
        assert "Test Channel" in content

        # Check typography CSS is included
        assert "Charter" in content
        assert "line-height" in content

    def test_multiple_articles_epub(self):
        """Test creating EPUB with multiple articles."""
        from create_epub import create_epub_book

        articles = [
            {
                "title": f"Article {i}",
                "channel": "Test Channel",
                "url": f"https://example.com/{i}",
                "article": f"# Article {i}\n\nContent for article {i}."
            }
            for i in range(3)
        ]

        output_path = os.path.join(self.temp_dir, "multi.epub")
        result = create_epub_book(articles, output_path=output_path)

        assert os.path.exists(result)
        # Multiple articles should result in larger file
        assert os.path.getsize(result) > 2000


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_video_url(self):
        """Test handling of invalid video URL."""
        from get_videos import get_video_info

        result = get_video_info("https://www.youtube.com/watch?v=INVALID_ID_12345")

        # Should return None for invalid videos
        assert result is None

    def test_empty_article_list(self):
        """Test EPUB creation with empty article list."""
        from create_epub import create_epub_book

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "empty.epub")

            # Empty list should still create a valid (but empty) EPUB
            result = create_epub_book([], output_path=output_path)
            assert os.path.exists(result)

    def test_special_characters_in_title(self):
        """Test handling of special characters in article titles."""
        from create_epub import create_epub_book

        article = {
            "title": "Test: \"Special\" Characters & Symbols <script>",
            "channel": "Test & Channel",
            "url": "https://example.com",
            "article": "# Content\n\nWith special chars: <>&\""
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "special.epub")
            result = create_epub_book([article], output_path=output_path)

            assert os.path.exists(result)

    def test_very_long_article(self):
        """Test handling of very long articles."""
        from create_epub import create_epub_book

        # Generate a long article
        long_content = "# Long Article\n\n" + ("This is a paragraph. " * 1000 + "\n\n") * 10

        article = {
            "title": "Very Long Article",
            "channel": "Test Channel",
            "url": "https://example.com",
            "article": long_content
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "long.epub")
            result = create_epub_book([article], output_path=output_path)

            assert os.path.exists(result)
            # Long article should result in reasonable file size
            assert os.path.getsize(result) > 1000


class TestPlaylistSupport:
    """Tests for playlist URL handling."""

    def test_playlist_url_detection(self):
        """Test playlist URL detection."""
        from get_videos import is_playlist_url

        # Playlist URLs
        assert is_playlist_url("https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf")
        assert is_playlist_url("https://www.youtube.com/watch?v=abc&list=PLxxx")

        # Non-playlist URLs
        assert not is_playlist_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert not is_playlist_url("https://youtu.be/dQw4w9WgXcQ")
        assert not is_playlist_url("https://www.youtube.com/@channel")
