"""
Unit tests for readtube components.
Tests individual functions with mocked dependencies.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


class TestGetVideos:
    """Tests for get_videos.py functions."""

    def test_is_playlist_url_with_playlist(self):
        """Test playlist URL detection."""
        from get_videos import is_playlist_url

        assert is_playlist_url("https://www.youtube.com/playlist?list=PLxxx")
        assert is_playlist_url("https://www.youtube.com/watch?v=abc&list=PLxxx")

    def test_is_playlist_url_with_video(self):
        """Test that video URLs are not detected as playlists."""
        from get_videos import is_playlist_url

        assert not is_playlist_url("https://www.youtube.com/watch?v=abc123")
        assert not is_playlist_url("https://youtu.be/abc123")

    def test_get_video_info_success(self, mock_yt_dlp_video):
        """Test successful video info extraction."""
        from get_videos import get_video_info

        result = get_video_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert result is not None
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["title"] == "Rick Astley - Never Gonna Give You Up"
        assert result["channel"] == "Rick Astley"
        assert "thumbnail" in result
        assert "duration" in result

    def test_get_video_info_failure(self, mocker):
        """Test video info extraction failure handling raises NetworkError."""
        from get_videos import get_video_info
        from errors import NetworkError

        mock_ydl = mocker.MagicMock()
        mock_ydl.__enter__ = mocker.MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = mocker.MagicMock(return_value=False)
        mock_ydl.extract_info = mocker.MagicMock(side_effect=Exception("Network error"))

        mocker.patch('yt_dlp.YoutubeDL', return_value=mock_ydl)

        with pytest.raises(NetworkError):
            get_video_info("https://www.youtube.com/watch?v=invalid")

    def test_get_videos_from_urls_single(self, mock_yt_dlp_video):
        """Test fetching multiple URLs."""
        from get_videos import get_videos_from_urls

        urls = ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
        result = get_videos_from_urls(urls)

        assert len(result) == 1
        assert result[0]["video_id"] == "dQw4w9WgXcQ"


class TestGetTranscripts:
    """Tests for get_transcripts.py functions."""

    def test_get_transcript_success(self, mock_transcript):
        """Test successful transcript extraction."""
        from get_transcripts import get_transcript

        result = get_transcript("dQw4w9WgXcQ", use_cache=False)

        assert result is not None
        assert "Hello everyone" in result
        assert "testing" in result

    def test_get_transcript_with_language(self, mock_transcript):
        """Test transcript extraction with language preference."""
        from get_transcripts import get_transcript

        result = get_transcript("dQw4w9WgXcQ", lang="en", use_cache=False)

        assert result is not None

    def test_cache_operations(self, temp_dir, mocker):
        """Test transcript caching."""
        from get_transcripts import save_to_cache, load_from_cache, CACHE_DIR

        # Patch CACHE_DIR to use temp directory
        mocker.patch('get_transcripts.CACHE_DIR', temp_dir)

        # Save to cache
        save_to_cache("test_video_id", "Test transcript content", lang="en")

        # Load from cache
        result = load_from_cache("test_video_id", lang="en")

        assert result == "Test transcript content"

    def test_cache_miss(self, temp_dir, mocker):
        """Test cache miss returns None."""
        from get_transcripts import load_from_cache

        mocker.patch('get_transcripts.CACHE_DIR', temp_dir)

        result = load_from_cache("nonexistent_video", lang="en")
        assert result is None


class TestCreateEpub:
    """Tests for create_epub.py functions."""

    def test_create_epub_book(self, sample_article, temp_dir):
        """Test EPUB creation."""
        from create_epub import create_epub_book

        output_path = os.path.join(temp_dir, "test.epub")
        result = create_epub_book([sample_article], output_path=output_path)

        assert result == output_path
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_create_epub_multiple_articles(self, sample_article, temp_dir):
        """Test EPUB creation with multiple articles."""
        from create_epub import create_epub_book

        articles = [sample_article, sample_article.copy()]
        articles[1]["title"] = "Second Article"

        output_path = os.path.join(temp_dir, "test_multi.epub")
        result = create_epub_book(articles, output_path=output_path)

        assert os.path.exists(output_path)

    def test_create_html(self, sample_article, temp_dir):
        """Test HTML creation."""
        from create_epub import create_html

        output_path = os.path.join(temp_dir, "test.html")
        result = create_html([sample_article], output_path=output_path)

        assert result == output_path
        assert os.path.exists(output_path)

        with open(output_path, 'r') as f:
            content = f.read()
            assert "The Art of Testing" in content
            assert "Test Channel" in content

    def test_create_ebook_epub_format(self, sample_article, temp_dir):
        """Test create_ebook with epub format."""
        from create_epub import create_ebook

        output_path = os.path.join(temp_dir, "test.epub")
        result = create_ebook([sample_article], output_path=output_path, format="epub")

        assert os.path.exists(output_path)

    def test_create_ebook_html_format(self, sample_article, temp_dir):
        """Test create_ebook with html format."""
        from create_epub import create_ebook

        output_path = os.path.join(temp_dir, "test.html")
        result = create_ebook([sample_article], output_path=output_path, format="html")

        assert os.path.exists(output_path)

    def test_typography_css_present(self, sample_article, temp_dir):
        """Test that typography CSS is included in output."""
        from create_epub import create_html, TYPOGRAPHY_CSS

        output_path = os.path.join(temp_dir, "test.html")
        create_html([sample_article], output_path=output_path)

        with open(output_path, 'r') as f:
            content = f.read()
            # Check for key typography elements
            assert "font-family: Charter" in content
            assert "line-height" in content


class TestFetchTranscript:
    """Tests for fetch_transcript.py functions."""

    def test_fetch_single_video(self, mock_yt_dlp_video, mocker):
        """Test fetching a single video with transcript."""
        from fetch_transcript import fetch_single_video

        # Mock get_transcript separately
        mocker.patch('fetch_transcript.get_transcript', return_value="Test transcript content")

        result = fetch_single_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert result is not None
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert "transcript" in result
        assert result["transcript"] == "Test transcript content"

    def test_print_video_data(self, sample_video_with_transcript, capsys):
        """Test video data printing."""
        from fetch_transcript import print_video_data

        print_video_data([sample_video_with_transcript])

        captured = capsys.readouterr()
        assert "Test Video Title" in captured.out
        assert "Test Channel" in captured.out
        assert "TRANSCRIPT:" in captured.out

    def test_print_video_data_summary_mode(self, sample_video_with_transcript, capsys):
        """Test video data printing in summary mode."""
        from fetch_transcript import print_video_data

        print_video_data([sample_video_with_transcript], summary_mode=True)

        captured = capsys.readouterr()
        assert "MODE: SUMMARY" in captured.out

    def test_print_video_data_with_output_dir(self, sample_video_with_transcript, capsys):
        """Test video data printing with output directory."""
        from fetch_transcript import print_video_data

        print_video_data([sample_video_with_transcript], output_dir="/tmp/output")

        captured = capsys.readouterr()
        assert "OUTPUT DIRECTORY:" in captured.out
        assert "/tmp/output" in captured.out

    def test_print_video_data_with_chapters(self, sample_video_with_transcript, capsys):
        """Test video data printing with chapters."""
        from fetch_transcript import print_video_data

        video = sample_video_with_transcript.copy()
        video['chapters'] = [
            {"title": "Introduction", "start_time": 0, "end_time": 60},
            {"title": "Main Content", "start_time": 60, "end_time": 180},
        ]

        print_video_data([video])

        captured = capsys.readouterr()
        assert "CHAPTERS" in captured.out
        assert "Introduction" in captured.out
        assert "Main Content" in captured.out

    def test_format_timestamp(self):
        """Test timestamp formatting."""
        from fetch_transcript import format_timestamp

        assert format_timestamp(0) == "0:00"
        assert format_timestamp(65) == "1:05"
        assert format_timestamp(3665) == "1:01:05"
        assert format_timestamp(7200) == "2:00:00"


class TestEdgeCases:
    """Edge case tests for various components."""

    def test_empty_article_content(self, temp_dir):
        """Test EPUB creation with empty article content."""
        from create_epub import create_epub_book

        articles = [{
            "title": "Empty Article",
            "channel": "Test Channel",
            "url": "https://youtube.com/watch?v=test",
            "article": ""
        }]

        output_path = os.path.join(temp_dir, "empty.epub")
        result = create_epub_book(articles, output_path=output_path)

        assert os.path.exists(output_path)

    def test_special_characters_in_title(self, temp_dir):
        """Test EPUB creation with special characters in title."""
        from create_epub import create_epub_book

        articles = [{
            "title": "Test <>&\"' Special Characters!@#$%",
            "channel": "Test & Channel",
            "url": "https://youtube.com/watch?v=test",
            "article": "# Test\n\nContent with <special> & characters"
        }]

        output_path = os.path.join(temp_dir, "special.epub")
        result = create_epub_book(articles, output_path=output_path)

        assert os.path.exists(output_path)

    def test_very_long_title(self, temp_dir):
        """Test EPUB creation with very long title."""
        from create_epub import create_epub_book

        long_title = "A" * 200
        articles = [{
            "title": long_title,
            "channel": "Test Channel",
            "url": "https://youtube.com/watch?v=test",
            "article": "# Test\n\nContent"
        }]

        output_path = os.path.join(temp_dir, "long_title.epub")
        result = create_epub_book(articles, output_path=output_path)

        assert os.path.exists(output_path)

    def test_unicode_content(self, temp_dir):
        """Test EPUB creation with unicode content."""
        from create_epub import create_epub_book

        articles = [{
            "title": "Unicode Test: 日本語 中文 한국어 العربية",
            "channel": "Test Channel",
            "url": "https://youtube.com/watch?v=test",
            "article": "# Unicode\n\nEmoji: 🎉 🚀 ❤️\n\nSpecial: α β γ δ ε"
        }]

        output_path = os.path.join(temp_dir, "unicode.epub")
        result = create_epub_book(articles, output_path=output_path)

        assert os.path.exists(output_path)

    def test_markdown_code_blocks(self, temp_dir):
        """Test EPUB creation with code blocks in markdown."""
        from create_epub import create_epub_book

        articles = [{
            "title": "Code Example",
            "channel": "Test Channel",
            "url": "https://youtube.com/watch?v=test",
            "article": """# Code Example

Here's some code:

```python
def hello():
    print("Hello, World!")
```

And inline `code` too.
"""
        }]

        output_path = os.path.join(temp_dir, "code.epub")
        result = create_epub_book(articles, output_path=output_path)

        assert os.path.exists(output_path)

    def test_html_output_with_code(self, temp_dir):
        """Test HTML creation with code blocks."""
        from create_epub import create_html

        articles = [{
            "title": "Code Example",
            "channel": "Test Channel",
            "url": "https://youtube.com/watch?v=test",
            "article": "# Test\n\n```python\nprint('hello')\n```"
        }]

        output_path = os.path.join(temp_dir, "code.html")
        result = create_html(articles, output_path=output_path)

        with open(output_path, 'r') as f:
            content = f.read()
            assert "<code>" in content or "<pre>" in content

    def test_get_video_info_with_chapters(self, mocker):
        """Test video info extraction includes chapters."""
        from get_videos import get_video_info

        mock_result = {
            "id": "test123",
            "title": "Test Video",
            "description": "Test description",
            "channel": "Test Channel",
            "thumbnail": "https://example.com/thumb.jpg",
            "duration": 300,
            "chapters": [
                {"title": "Intro", "start_time": 0, "end_time": 60},
                {"title": "Main", "start_time": 60, "end_time": 240},
                {"title": "Outro", "start_time": 240, "end_time": 300},
            ]
        }

        mock_ydl = mocker.MagicMock()
        mock_ydl.__enter__ = mocker.MagicMock(return_value=mock_ydl)
        mock_ydl.__exit__ = mocker.MagicMock(return_value=False)
        mock_ydl.extract_info = mocker.MagicMock(return_value=mock_result)

        mocker.patch('yt_dlp.YoutubeDL', return_value=mock_ydl)

        result = get_video_info("https://www.youtube.com/watch?v=test123")

        assert result is not None
        assert "chapters" in result
        assert len(result["chapters"]) == 3
        assert result["chapters"][0]["title"] == "Intro"

    def test_is_playlist_url_edge_cases(self):
        """Test playlist URL detection edge cases."""
        from get_videos import is_playlist_url

        # Various playlist URL formats
        assert is_playlist_url("https://youtube.com/playlist?list=PLabc123")
        assert is_playlist_url("https://www.youtube.com/watch?v=xyz&list=PLabc123")
        assert is_playlist_url("http://youtube.com/playlist?list=PLabc123")

        # Non-playlist URLs
        assert not is_playlist_url("https://youtube.com/watch?v=abc123")
        assert not is_playlist_url("https://youtu.be/abc123")
        assert not is_playlist_url("https://youtube.com/@channel")
        assert not is_playlist_url("https://youtube.com/channel/UCxxx")

    def test_cache_expiry(self, temp_dir, mocker):
        """Test that expired cache entries are not returned."""
        from get_transcripts import save_to_cache, load_from_cache
        import json
        import os

        mocker.patch('get_transcripts.CACHE_DIR', temp_dir)

        # Create an expired cache entry manually
        cache_file = os.path.join(temp_dir, "expired_video_en.json")
        expired_data = {
            "transcript": "Old transcript",
            "timestamp": 0  # Very old timestamp
        }
        with open(cache_file, 'w') as f:
            json.dump(expired_data, f)

        # Should return None because cache is expired
        result = load_from_cache("expired_video", lang="en")
        assert result is None
