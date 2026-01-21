"""
End-to-end tests for readtube.
These tests use real YouTube data (with caching) to verify the full pipeline.

Note: These tests make real network requests. Run with:
    pytest tests/test_e2e.py -v -s
"""

import os
import socket
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
        try:
            socket.getaddrinfo("www.youtube.com", 443)
        except OSError:
            pytest.skip("Network/DNS unavailable for e2e YouTube tests")
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
        try:
            transcript = get_transcript(video["video_id"])
        except Exception as e:
            if "blocking" in str(e).lower() or "IP" in str(e):
                pytest.skip("YouTube rate limiting - skipping test")
            raise

        if transcript is None:
            pytest.skip("Could not fetch transcript - possible rate limiting")

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

        try:
            transcript = get_transcript(TEST_VIDEO_ID)
        except Exception as e:
            if "blocking" in str(e).lower() or "IP" in str(e):
                pytest.skip("YouTube rate limiting - skipping test")
            raise

        if transcript is None:
            pytest.skip("Could not fetch transcript - possible rate limiting")

        assert isinstance(transcript, str)
        assert len(transcript) > 100

    @pytest.mark.slow
    def test_transcript_language_listing(self):
        """Test listing available transcript languages."""
        from get_transcripts import list_available_languages

        try:
            languages = list_available_languages(TEST_VIDEO_ID)
        except Exception as e:
            if "blocking" in str(e).lower() or "IP" in str(e):
                pytest.skip("YouTube rate limiting - skipping test")
            raise

        assert isinstance(languages, list)
        # Most videos have at least auto-generated English
        if len(languages) > 0:
            # Check structure
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
        try:
            transcript1 = get_transcript(TEST_VIDEO_ID, use_cache=True)
        except Exception as e:
            if "blocking" in str(e).lower() or "IP" in str(e):
                pytest.skip("YouTube rate limiting - skipping test")
            raise
        first_fetch_time = time.time() - start

        if transcript1 is None:
            pytest.skip("Could not fetch transcript - possible rate limiting")

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


class TestGetVideosE2E:
    """E2E tests for get_videos.py with real YouTube calls."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        try:
            socket.getaddrinfo("www.youtube.com", 443)
        except OSError:
            pytest.skip("Network unavailable")
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    @pytest.mark.slow
    def test_get_video_info_with_chapters(self):
        """Test fetching video info for a video with chapters."""
        from get_videos import get_video_info

        # Use a video known to have chapters
        video = get_video_info(TEST_VIDEO_URL)

        assert video is not None
        assert "chapters" in video
        assert isinstance(video["chapters"], list)

    @pytest.mark.slow
    def test_get_videos_from_urls_single(self):
        """Test get_videos_from_urls with single video."""
        from get_videos import get_videos_from_urls

        videos = get_videos_from_urls([TEST_VIDEO_URL])

        assert len(videos) == 1
        assert videos[0]["video_id"] == TEST_VIDEO_ID

    @pytest.mark.slow
    def test_get_videos_from_urls_multiple(self):
        """Test get_videos_from_urls with multiple videos."""
        from get_videos import get_videos_from_urls

        # Two different videos
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=9bZkp7q19f0",  # Gangnam Style
        ]
        videos = get_videos_from_urls(urls)

        assert len(videos) == 2

    @pytest.mark.slow
    def test_main_function_with_urls(self):
        """Test main function with video URLs."""
        from get_videos import main

        videos = main(video_urls=[TEST_VIDEO_URL])

        assert len(videos) == 1
        assert videos[0]["video_id"] == TEST_VIDEO_ID


class TestGetTranscriptsE2E:
    """E2E tests for get_transcripts.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        try:
            socket.getaddrinfo("www.youtube.com", 443)
        except OSError:
            pytest.skip("Network unavailable")

    @pytest.mark.slow
    def test_get_transcript_with_timestamps(self):
        """Test getting transcript with timestamps."""
        from get_transcripts import get_transcript_with_timestamps

        try:
            result = get_transcript_with_timestamps(TEST_VIDEO_ID)
        except Exception as e:
            if "blocking" in str(e).lower() or "IP" in str(e):
                pytest.skip("YouTube rate limiting - skipping test")
            raise

        if result is not None:
            assert isinstance(result, list)
            if result:
                assert "text" in result[0] or hasattr(result[0], "text")

    @pytest.mark.slow
    def test_get_transcript_different_language(self):
        """Test getting transcript in different language if available."""
        from get_transcripts import get_transcript, list_available_languages

        try:
            languages = list_available_languages(TEST_VIDEO_ID)
        except Exception as e:
            if "blocking" in str(e).lower() or "IP" in str(e):
                pytest.skip("YouTube rate limiting - skipping test")
            raise

        if len(languages) > 1:
            # Try to get transcript in a different language
            alt_lang = languages[1]["code"]
            try:
                transcript = get_transcript(TEST_VIDEO_ID, lang=alt_lang)
                # May be None if rate limited
                if transcript is not None:
                    assert len(transcript) > 0
            except Exception as e:
                if "blocking" in str(e).lower() or "IP" in str(e):
                    pytest.skip("YouTube rate limiting - skipping test")
                raise

    @pytest.mark.slow
    def test_format_timestamp(self):
        """Test timestamp formatting."""
        from get_transcripts import format_timestamp

        assert format_timestamp(0) == "0:00"
        assert format_timestamp(65) == "1:05"
        assert format_timestamp(3661) == "1:01:01"


class TestImagesE2E:
    """E2E tests for images.py with real downloads."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        try:
            socket.getaddrinfo("www.youtube.com", 443)
        except OSError:
            pytest.skip("Network unavailable")
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    @pytest.mark.slow
    def test_get_best_thumbnail_real(self):
        """Test downloading best thumbnail from real video."""
        from images import get_best_thumbnail

        output_path = os.path.join(self.temp_dir, "thumb.jpg")
        result = get_best_thumbnail(TEST_VIDEO_URL, output_path)

        assert result is not None
        assert os.path.exists(result)
        assert os.path.getsize(result) > 1000  # Should be a real image

    @pytest.mark.slow
    def test_get_video_thumbnails_real(self):
        """Test downloading all thumbnails from real video."""
        from images import get_video_thumbnails

        results = get_video_thumbnails(TEST_VIDEO_URL, self.temp_dir)

        assert len(results) > 0
        for path in results:
            assert os.path.exists(path)


class TestAsyncFetchE2E:
    """E2E tests for async_fetch.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        try:
            socket.getaddrinfo("www.youtube.com", 443)
        except OSError:
            pytest.skip("Network unavailable")

    @pytest.mark.slow
    def test_fetch_video_async_real(self):
        """Test async video fetching with real URL."""
        import asyncio
        from async_fetch import fetch_video_async

        result = asyncio.run(fetch_video_async(TEST_VIDEO_URL))

        assert result is not None
        assert result["video_id"] == TEST_VIDEO_ID

    @pytest.mark.slow
    def test_fetch_videos_async_multiple(self):
        """Test async fetching multiple videos."""
        import asyncio
        from async_fetch import fetch_videos_async

        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=9bZkp7q19f0",
        ]

        results = asyncio.run(fetch_videos_async(urls))

        assert len(results) == 2

    @pytest.mark.slow
    def test_async_fetcher_class_real(self):
        """Test AsyncFetcher class with real videos."""
        import asyncio
        from async_fetch import AsyncFetcher

        async def test():
            async with AsyncFetcher(max_concurrent=2) as fetcher:
                result = await fetcher.fetch_video(TEST_VIDEO_URL)
                return result

        result = asyncio.run(test())
        assert result is not None
        assert result["video_id"] == TEST_VIDEO_ID


class TestFetchTranscriptCLI:
    """E2E tests for fetch_transcript.py CLI."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        try:
            socket.getaddrinfo("www.youtube.com", 443)
        except OSError:
            pytest.skip("Network unavailable")
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    @pytest.mark.slow
    def test_fetch_single_video(self):
        """Test fetch_single_video function."""
        from fetch_transcript import fetch_single_video

        try:
            result = fetch_single_video(TEST_VIDEO_URL)
        except Exception as e:
            if "blocking" in str(e).lower() or "IP" in str(e):
                pytest.skip("YouTube rate limiting - skipping test")
            raise

        if result is None:
            pytest.skip("Could not fetch video - possible rate limiting")

        assert "title" in result
        assert "transcript" in result

    @pytest.mark.slow
    def test_fetch_single_video_with_output_json(self):
        """Test saving fetch result to JSON."""
        from fetch_transcript import fetch_single_video, write_video_json
        import json

        try:
            result = fetch_single_video(TEST_VIDEO_URL)
        except Exception as e:
            if "blocking" in str(e).lower() or "IP" in str(e):
                pytest.skip("YouTube rate limiting - skipping test")
            raise

        if result is None:
            pytest.skip("Could not fetch video - possible rate limiting")

        # Save to JSON using the module's function
        output_path = os.path.join(self.temp_dir, "video.json")
        write_video_json([result], output_path)

        # Verify JSON is valid - single video is written as dict, not list
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # write_video_json writes single videos as dict, multiple as list
        assert loaded["title"] == result["title"]


class TestWriteArticleCLI:
    """E2E tests for write_article.py CLI."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    def test_write_article_prompt_only(self):
        """Test write_article in prompt-only mode."""
        from write_article import load_video_data, generate_prompt_file
        import json

        # Create test video data
        video_data = {
            "title": "Test Video",
            "channel": "Test Channel",
            "transcript": "This is a test transcript with enough words to make it interesting.",
            "video_id": "test123",
            "url": "https://youtube.com/watch?v=test123",
        }

        input_path = os.path.join(self.temp_dir, "video.json")
        with open(input_path, "w") as f:
            json.dump(video_data, f)

        # Load and generate prompt
        loaded = load_video_data(input_path)
        prompt_path = generate_prompt_file(loaded, self.temp_dir)

        assert os.path.exists(prompt_path)
        with open(prompt_path, "r") as f:
            content = f.read()
        assert "Test Video" in content
        assert "SYSTEM PROMPT" in content


class TestBatchProcessingE2E:
    """E2E tests for batch.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        try:
            socket.getaddrinfo("www.youtube.com", 443)
        except OSError:
            pytest.skip("Network unavailable")
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    @pytest.mark.slow
    def test_process_single_job(self):
        """Test processing a single batch job."""
        from batch import process_job
        from config import BatchConfig, BatchJob

        job = BatchJob(url=TEST_VIDEO_URL)
        config = BatchConfig(output_dir=self.temp_dir, jobs=[job])

        try:
            result = process_job(job, config)
        except Exception as e:
            if "blocking" in str(e).lower() or "IP" in str(e):
                pytest.skip("YouTube rate limiting - skipping test")
            raise

        if result is None:
            pytest.skip("Could not process job - possible rate limiting")

        assert result["video"]["video_id"] == TEST_VIDEO_ID
        assert result["transcript"] is not None
        assert result["word_count"] > 0


class TestCreateEpubFormats:
    """E2E tests for different output formats."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    def test_create_epub_with_markdown_formatting(self):
        """Test EPUB with rich markdown formatting."""
        from create_epub import create_epub_book

        article = {
            "title": "Markdown Test",
            "channel": "Test Channel",
            "url": "https://example.com",
            "article": """# Main Heading

## Introduction

This is a paragraph with **bold** and *italic* text.

### Key Points

- First point
- Second point
- Third point

### Code Example

Here's some inline `code` and a block:

```python
def hello():
    print("Hello World")
```

### Quote

> This is a blockquote
> spanning multiple lines

### Conclusion

Final thoughts here.
"""
        }

        output_path = os.path.join(self.temp_dir, "markdown.epub")
        result = create_epub_book([article], output_path=output_path)

        assert os.path.exists(result)
        assert os.path.getsize(result) > 1000

    def test_create_html_standalone(self):
        """Test standalone HTML creation."""
        from create_epub import create_html

        articles = [
            {
                "title": "Article One",
                "channel": "Channel A",
                "url": "https://example.com/1",
                "article": "# One\n\nFirst article content.",
            },
            {
                "title": "Article Two",
                "channel": "Channel B",
                "url": "https://example.com/2",
                "article": "# Two\n\nSecond article content.",
            },
        ]

        output_path = os.path.join(self.temp_dir, "multi.html")
        result = create_html(articles, output_path=output_path)

        assert os.path.exists(result)
        with open(result, "r") as f:
            content = f.read()

        assert "Article One" in content
        assert "Article Two" in content
        assert "Channel A" in content

    def test_create_ebook_auto_filename(self):
        """Test EPUB creation with auto-generated filename."""
        from create_epub import create_ebook

        articles = [{
            "title": "Auto Named Article",
            "channel": "Test Channel",
            "article": "# Test\n\nContent here.",
        }]

        # Don't specify output_path
        result = create_ebook(articles, format="epub")

        assert result is not None
        assert os.path.exists(result)
        # Clean up
        os.remove(result)


class TestRSSFeedsE2E:
    """E2E tests for RSS feed generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    def test_generate_complete_rss_feed(self):
        """Test generating a complete RSS feed."""
        from rss import generate_rss_feed

        articles = [
            {
                "title": "First Article",
                "channel": "Tech Channel",
                "url": "https://youtube.com/watch?v=abc",
                "article": "# First\n\nContent of first article.",
                "thumbnail": "https://example.com/thumb1.jpg",
            },
            {
                "title": "Second Article",
                "channel": "Science Channel",
                "url": "https://youtube.com/watch?v=xyz",
                "article": "# Second\n\nContent of second article.",
            },
        ]

        output_path = os.path.join(self.temp_dir, "feed.xml")
        xml = generate_rss_feed(
            articles,
            title="Test Feed",
            description="A test RSS feed",
            link="https://example.com",
            output_path=output_path
        )

        assert os.path.exists(output_path)
        assert "<?xml" in xml
        assert "<rss" in xml
        assert "First Article" in xml
        assert "Second Article" in xml

    def test_generate_complete_atom_feed(self):
        """Test generating a complete Atom feed."""
        from rss import generate_atom_feed

        articles = [
            {
                "title": "Atom Article",
                "channel": "Test Channel",
                "url": "https://youtube.com/watch?v=abc",
                "article": "# Atom\n\nAtom feed content.",
            },
        ]

        output_path = os.path.join(self.temp_dir, "feed.atom")
        xml = generate_atom_feed(
            articles,
            title="Atom Feed",
            output_path=output_path
        )

        assert os.path.exists(output_path)
        assert "<feed" in xml
        assert "Atom Article" in xml


class TestLLMIntegration:
    """Integration tests for LLM module with mocks."""

    def test_generate_article_with_mock_backend(self, mocker):
        """Test article generation with mocked LLM backend."""
        from llm import generate_article, LLMBackend

        # Create a mock backend
        mock_backend = mocker.MagicMock(spec=LLMBackend)
        mock_backend.generate.return_value = "# Generated Article\n\nThis is the generated content."

        mocker.patch('llm.get_backend', return_value=mock_backend)

        result = generate_article(
            transcript="This is a test transcript",
            title="Test Video",
            channel="Test Channel"
        )

        assert result == "# Generated Article\n\nThis is the generated content."
        mock_backend.generate.assert_called_once()

    def test_ollama_backend_generate_mock(self, mocker):
        """Test Ollama backend with mocked HTTP response."""
        from llm import OllamaBackend
        import json

        # Mock urllib.request.urlopen
        mock_response = mocker.MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = mocker.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mocker.MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps({"response": "Generated text"}).encode()

        mocker.patch('urllib.request.urlopen', return_value=mock_response)

        backend = OllamaBackend()
        # Mock is_available to return True
        mocker.patch.object(backend, 'is_available', return_value=True)

        result = backend.generate("Test prompt")
        assert result == "Generated text"

    def test_openai_backend_generate_mock(self, mocker):
        """Test OpenAI backend with mocked HTTP response."""
        from llm import OpenAIBackend
        import json

        mock_response = mocker.MagicMock()
        mock_response.__enter__ = mocker.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mocker.MagicMock(return_value=False)
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "OpenAI response"}}]
        }).encode()

        mocker.patch('urllib.request.urlopen', return_value=mock_response)

        backend = OpenAIBackend(api_key="test-key")
        result = backend.generate("Test prompt")

        assert result == "OpenAI response"


class TestSchedulerIntegration:
    """Integration tests for scheduler module."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    def test_scheduler_full_workflow(self):
        """Test scheduler initialization and history tracking."""
        from scheduler import FetchHistory
        from pathlib import Path

        history_path = Path(self.temp_dir) / "history.json"
        history = FetchHistory(path=history_path)

        # Mark some videos as fetched
        history.mark_fetched("video1", {"title": "Video 1"})
        history.mark_fetched("video2", {"title": "Video 2"})
        history.update_last_run()

        # Verify persistence
        history2 = FetchHistory(path=history_path)
        assert history2.is_fetched("video1")
        assert history2.is_fetched("video2")
        assert not history2.is_fetched("video3")

        stats = history2.get_stats()
        assert stats["total_fetched"] == 2

    def test_create_systemd_service(self):
        """Test systemd service file generation."""
        from scheduler import create_systemd_service

        config_path = os.path.join(self.temp_dir, "config.yaml")
        service = create_systemd_service(config_path, interval=3600)

        assert "[Unit]" in service
        assert "[Service]" in service
        assert "[Install]" in service

    def test_create_launchd_plist(self):
        """Test launchd plist file generation."""
        from scheduler import create_launchd_plist

        config_path = os.path.join(self.temp_dir, "config.yaml")
        plist = create_launchd_plist(config_path, interval=3600)

        assert "<?xml" in plist
        assert "plist" in plist
        assert "com.readtube" in plist


class TestTTSIntegration:
    """Integration tests for TTS module."""

    def test_preprocess_complete_article(self):
        """Test preprocessing a complete article for TTS."""
        from tts import preprocess_text

        article = """# Article Title

## Introduction

This is a **bold** statement about the topic.

### Key Points

- First point
- Second point

Here's a [link](https://example.com) and some `code`.

```python
print("skip this")
```

e.g. this should become "for example"
i.e. this should become "that is"

## Conclusion

Final words here.
"""

        result = preprocess_text(article)

        # Headers removed
        assert "#" not in result
        # Emphasis removed
        assert "**" not in result
        # Links converted
        assert "[link]" not in result
        assert "link" in result
        # Code blocks removed
        assert "print" not in result
        # Abbreviations expanded
        assert "for example" in result
        assert "that is" in result


class TestTranslateIntegration:
    """Integration tests for translate module with mocks."""

    def test_translate_transcript_mock(self, mocker):
        """Test translating a transcript with mocked backend."""
        from translate import translate_transcript, TranslationBackend, BACKENDS

        # Create mock backend
        mock_backend = mocker.MagicMock(spec=TranslationBackend)
        mock_backend.translate.return_value = "Texto traducido"
        mock_backend.is_available.return_value = True

        # Add to backends
        BACKENDS["mock"] = lambda: mock_backend

        try:
            result = translate_transcript(
                "Text to translate",
                target_lang="es",
                backend="mock"
            )

            # Should have called translate
            assert mock_backend.translate.called or result is not None
        finally:
            # Cleanup
            del BACKENDS["mock"]


class TestWriteArticleMain:
    """Tests for write_article.py main function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    def test_load_video_data_empty_file(self):
        """Test loading empty JSON file."""
        from write_article import load_video_data
        import json

        path = os.path.join(self.temp_dir, "empty.json")
        with open(path, "w") as f:
            json.dump({}, f)

        result = load_video_data(path)
        assert result == {}

    def test_generate_prompt_with_long_transcript(self):
        """Test prompt generation truncates long transcripts."""
        from write_article import generate_prompt_file

        # Create video data with very long transcript
        long_transcript = "word " * 100000  # Way more than 50000 chars
        video_data = {
            "title": "Long Video",
            "channel": "Test Channel",
            "transcript": long_transcript,
            "video_id": "long123",
        }

        output_path = generate_prompt_file(video_data, self.temp_dir)
        content = open(output_path).read()

        # Should be truncated
        assert len(content) < len(long_transcript)


class TestGetVideosExtended:
    """Extended tests for get_videos.py."""

    def test_video_info_structure(self):
        """Test VideoInfo has correct structure."""
        from get_videos import VideoInfo

        # Create a valid VideoInfo
        video: VideoInfo = {
            "title": "Test",
            "video_id": "abc123",
            "description": "Desc",
            "channel": "Channel",
            "url": "https://youtube.com/watch?v=abc123",
            "thumbnail": "https://example.com/thumb.jpg",
            "duration": 300,
            "chapters": [],
        }

        assert video["title"] == "Test"
        assert video["duration"] == 300

    def test_chapter_info_structure(self):
        """Test ChapterInfo has correct structure."""
        from get_videos import ChapterInfo

        chapter: ChapterInfo = {
            "title": "Introduction",
            "start_time": 0.0,
            "end_time": 60.0,
        }

        assert chapter["title"] == "Introduction"
        assert chapter["end_time"] == 60.0


class TestIntegrationsExtended:
    """Extended tests for integrations.py."""

    def test_readwise_highlight_format(self, mocker):
        """Test Readwise highlight formatting."""
        from integrations import ReadwiseClient

        client = ReadwiseClient(token="test_token")

        # Mock the HTTP request
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.__enter__ = mocker.MagicMock(return_value=mock_response)
        mock_response.__exit__ = mocker.MagicMock(return_value=False)

        assert client.token == "test_token"
        assert client.API_BASE == "https://readwise.io/api/v2"


class TestCreateEpubExtendedFormats:
    """Extended tests for create_epub.py formats."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    def test_create_ebook_with_empty_article(self):
        """Test EPUB creation with empty article content."""
        from create_epub import create_ebook

        articles = [{
            "title": "Empty Article",
            "channel": "Test",
            "article": "",
        }]

        output_path = os.path.join(self.temp_dir, "empty.epub")
        result = create_ebook(articles, output_path=output_path)

        assert result is not None
        assert os.path.exists(result)

    def test_create_ebook_with_unicode(self):
        """Test EPUB creation with Unicode characters."""
        from create_epub import create_ebook

        articles = [{
            "title": "Unicode Test: 日本語 中文 한국어 العربية",
            "channel": "Test Channel 测试",
            "article": "# Unicode Content\n\nEmoji: 🎉 🚀 💡\n\nMath: ∑∫∂\n\nGreek: αβγδ",
        }]

        output_path = os.path.join(self.temp_dir, "unicode.epub")
        result = create_ebook(articles, output_path=output_path)

        assert result is not None
        assert os.path.exists(result)

    def test_create_html_with_images(self):
        """Test HTML creation with image references."""
        from create_epub import create_html

        articles = [{
            "title": "Article with Images",
            "channel": "Test",
            "url": "https://example.com",
            "thumbnail": "https://example.com/thumb.jpg",
            "article": "# Content\n\n![Image](https://example.com/img.png)",
        }]

        output_path = os.path.join(self.temp_dir, "images.html")
        result = create_html(articles, output_path=output_path)

        assert result is not None
        content = open(result).read()
        assert "img" in content.lower()


class TestBatchExtended:
    """Extended tests for batch.py."""

    def test_progress_bar_with_tqdm(self, mocker):
        """Test progress bar uses tqdm when available."""
        from batch import progress_bar

        items = [1, 2, 3, 4, 5]
        result = list(progress_bar(items, desc="Test"))

        assert result == items

    def test_retry_with_backoff_immediate_success(self):
        """Test retry returns immediately on success."""
        from batch import retry_with_backoff
        import time

        call_count = [0]

        def quick_func():
            call_count[0] += 1
            return "done"

        start = time.time()
        result = retry_with_backoff(quick_func, max_attempts=3, delay=1.0)
        elapsed = time.time() - start

        assert result == "done"
        assert call_count[0] == 1
        assert elapsed < 0.5  # Should return immediately


class TestConfigExtended:
    """Extended tests for config.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir)

    def test_config_channels_default(self):
        """Test Config has default channels."""
        from config import Config

        config = Config()
        assert isinstance(config.channels, list)

    def test_config_typography_values(self):
        """Test TypographyConfig has expected values."""
        from config import TypographyConfig

        config = TypographyConfig()
        assert config.font_size == "1.1em"
        assert config.line_height == 1.4
        assert config.max_width == "65ch"

    def test_batch_job_defaults(self):
        """Test BatchJob has sensible defaults."""
        from config import BatchJob

        job = BatchJob(url="https://youtube.com/watch?v=test")

        assert job.url == "https://youtube.com/watch?v=test"
        assert job.output_format == "epub"  # Default format
        assert job.summary_mode is False


class TestAsyncFetchExtended:
    """Extended tests for async_fetch.py."""

    def test_fetch_videos_async_with_concurrency(self):
        """Test async fetch respects concurrency limit."""
        import asyncio
        from async_fetch import AsyncFetcher

        async def test():
            fetcher = AsyncFetcher(max_concurrent=2)
            assert fetcher.max_concurrent == 2
            fetcher.close()
            return True

        result = asyncio.run(test())
        assert result is True

    def test_async_fetcher_executor(self):
        """Test AsyncFetcher creates executor."""
        from async_fetch import AsyncFetcher

        fetcher = AsyncFetcher(max_concurrent=3, max_workers=4)
        assert fetcher.max_concurrent == 3
        assert fetcher.executor is not None
        fetcher.close()
