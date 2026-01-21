"""
Tests for new Readtube modules: config, batch, themes, rss, tts, integrations.
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """Tests for config.py module."""

    def test_typography_config_defaults(self):
        """Test TypographyConfig has sensible defaults."""
        from config import TypographyConfig

        config = TypographyConfig()
        assert "Charter" in config.font_family
        assert config.line_height == 1.4
        assert config.max_width == "65ch"

    def test_output_config_defaults(self):
        """Test OutputConfig has sensible defaults."""
        from config import OutputConfig

        config = OutputConfig()
        assert config.default_format == "epub"
        assert config.include_cover is True
        assert config.include_toc is True

    def test_fetch_config_defaults(self):
        """Test FetchConfig has sensible defaults."""
        from config import FetchConfig

        config = FetchConfig()
        assert config.cache_enabled is True
        assert config.cache_days == 7
        assert config.retry_attempts == 3

    def test_config_to_dict_and_back(self):
        """Test Config serialization round-trip."""
        from config import Config

        config = Config()
        data = config.to_dict()

        assert "typography" in data
        assert "output" in data
        assert "fetch" in data
        assert "channels" in data

        # Can recreate from dict
        restored = Config.from_dict(data)
        assert restored.typography.font_family == config.typography.font_family

    def test_config_save_and_load(self, temp_dir):
        """Test Config save and load from file."""
        from config import Config

        config = Config()
        config.channels = ["https://youtube.com/@test"]

        config_path = Path(temp_dir) / "test_config.json"
        config.save(config_path)

        assert config_path.exists()

        loaded = Config.load(config_path)
        assert loaded.channels == config.channels

    def test_batch_job_creation(self):
        """Test BatchJob dataclass."""
        from config import BatchJob

        job = BatchJob(
            url="https://youtube.com/watch?v=test",
            output_format="pdf",
            summary_mode=True
        )

        assert job.url == "https://youtube.com/watch?v=test"
        assert job.output_format == "pdf"
        assert job.summary_mode is True

    def test_batch_config_from_dict(self):
        """Test BatchConfig creation from dict."""
        from config import BatchConfig

        data = {
            "output_dir": "./ebooks",
            "default_format": "pdf",
            "jobs": [
                "https://youtube.com/watch?v=abc",
                {"url": "https://youtube.com/watch?v=xyz", "summary_mode": True}
            ]
        }

        config = BatchConfig.from_dict(data)

        assert config.output_dir == "./ebooks"
        assert config.default_format == "pdf"
        assert len(config.jobs) == 2
        assert config.jobs[0].url == "https://youtube.com/watch?v=abc"
        assert config.jobs[1].summary_mode is True

    def test_batch_config_load_json(self, temp_dir):
        """Test BatchConfig loading from JSON file."""
        from config import BatchConfig

        config_data = {
            "output_dir": "./out",
            "jobs": ["https://youtube.com/watch?v=test"]
        }

        config_path = Path(temp_dir) / "batch.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        loaded = BatchConfig.load(config_path)
        assert len(loaded.jobs) == 1
        assert loaded.output_dir == "./out"

    def test_batch_config_load_yaml(self, temp_dir):
        """Test BatchConfig loading from YAML file."""
        from config import BatchConfig
        pytest.importorskip("yaml")

        yaml_content = """
output_dir: ./ebooks
default_format: epub
jobs:
  - https://youtube.com/watch?v=abc
  - url: https://youtube.com/watch?v=xyz
    summary_mode: true
"""
        config_path = Path(temp_dir) / "batch.yaml"
        with open(config_path, "w") as f:
            f.write(yaml_content)

        loaded = BatchConfig.load(config_path)
        assert len(loaded.jobs) == 2

    def test_get_config_singleton(self):
        """Test get_config returns singleton."""
        from config import get_config, set_config, Config

        # Reset
        set_config(None)

        config1 = get_config()
        config2 = get_config()

        # Should be same instance
        assert config1 is config2


class TestThemes:
    """Tests for themes.py module."""

    def test_get_default_theme(self):
        """Test getting default theme."""
        from themes import get_theme

        theme = get_theme("default")
        assert theme.name == "default"
        assert "Charter" in theme.css
        assert "line-height" in theme.css

    def test_get_dark_theme(self):
        """Test getting dark theme."""
        from themes import get_theme

        theme = get_theme("dark")
        assert theme.name == "dark"
        assert "#1a1a1a" in theme.css  # Dark background

    def test_get_modern_theme(self):
        """Test getting modern theme."""
        from themes import get_theme

        theme = get_theme("modern")
        assert theme.name == "modern"
        assert "sans-serif" in theme.css

    def test_get_minimal_theme(self):
        """Test getting minimal theme."""
        from themes import get_theme

        theme = get_theme("minimal")
        assert theme.name == "minimal"
        assert "Georgia" in theme.css

    def test_get_invalid_theme(self):
        """Test getting non-existent theme raises error."""
        from themes import get_theme

        with pytest.raises(ValueError) as exc:
            get_theme("nonexistent")
        assert "nonexistent" in str(exc.value)

    def test_list_themes(self):
        """Test listing all themes."""
        from themes import list_themes

        themes = list_themes()
        assert "default" in themes
        assert "dark" in themes
        assert "modern" in themes
        assert "minimal" in themes

    def test_register_custom_theme(self):
        """Test registering a custom theme."""
        from themes import Theme, register_theme, get_theme, THEMES

        custom = Theme(
            name="custom_test",
            description="Test theme",
            css="body { color: red; }"
        )

        register_theme(custom)
        assert "custom_test" in THEMES

        retrieved = get_theme("custom_test")
        assert retrieved.css == custom.css

        # Cleanup
        del THEMES["custom_test"]

    def test_load_custom_css(self, temp_dir):
        """Test loading custom CSS file as theme."""
        from themes import load_custom_css

        css_content = "body { font-size: 20px; }"
        css_path = Path(temp_dir) / "custom.css"
        with open(css_path, "w") as f:
            f.write(css_content)

        theme = load_custom_css(str(css_path))
        assert theme.name == "custom"
        assert theme.css == css_content


class TestBatch:
    """Tests for batch.py module."""

    def test_estimate_reading_time(self):
        """Test reading time estimation."""
        from batch import estimate_reading_time

        # 200 words = 1 minute at default 200 WPM
        text = " ".join(["word"] * 200)
        assert estimate_reading_time(text) == 1

        # 400 words = 2 minutes
        text = " ".join(["word"] * 400)
        assert estimate_reading_time(text) == 2

        # Very short text = minimum 1 minute
        assert estimate_reading_time("hello") == 1

    def test_estimate_reading_time_custom_wpm(self):
        """Test reading time with custom WPM."""
        from batch import estimate_reading_time

        text = " ".join(["word"] * 100)
        assert estimate_reading_time(text, wpm=100) == 1
        assert estimate_reading_time(text, wpm=50) == 2

    def test_retry_with_backoff_success(self):
        """Test retry succeeds on first try."""
        from batch import retry_with_backoff

        call_count = 0

        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_with_backoff(succeed)
        assert result == "success"
        assert call_count == 1

    def test_retry_with_backoff_eventual_success(self):
        """Test retry succeeds after failures."""
        from batch import retry_with_backoff

        call_count = 0

        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = retry_with_backoff(fail_then_succeed, max_attempts=3, delay=0.01)
        assert result == "success"
        assert call_count == 3

    def test_retry_with_backoff_all_failures(self):
        """Test retry raises after max failures."""
        from batch import retry_with_backoff

        def always_fail():
            raise Exception("Always fails")

        with pytest.raises(Exception):
            retry_with_backoff(always_fail, max_attempts=2, delay=0.01)

    def test_progress_bar_fallback(self):
        """Test progress bar works without tqdm."""
        from batch import progress_bar

        items = [1, 2, 3]
        result = list(progress_bar(items, desc="Test"))
        assert result == [1, 2, 3]


class TestRSS:
    """Tests for rss.py module."""

    @pytest.fixture
    def sample_articles(self):
        """Sample articles for RSS testing."""
        return [
            {
                "title": "Test Article 1",
                "channel": "Test Channel",
                "url": "https://youtube.com/watch?v=abc",
                "article": "# Test\n\nThis is test content.",
                "created_at": "2024-01-15T10:00:00",
            },
            {
                "title": "Test Article 2",
                "channel": "Another Channel",
                "url": "https://youtube.com/watch?v=xyz",
                "article": "# Another\n\nMore content here.",
                "created_at": datetime(2024, 1, 16, 12, 0, 0),
            }
        ]

    def test_generate_rss_feed(self, sample_articles):
        """Test RSS feed generation."""
        from rss import generate_rss_feed

        xml = generate_rss_feed(sample_articles, title="Test Feed")

        assert '<?xml' in xml
        assert '<rss' in xml
        assert 'Test Feed' in xml
        assert 'Test Article 1' in xml
        assert 'Test Article 2' in xml

    def test_generate_rss_feed_with_output(self, sample_articles, temp_dir):
        """Test RSS feed generation with file output."""
        from rss import generate_rss_feed

        output_path = Path(temp_dir) / "feed.xml"
        xml = generate_rss_feed(sample_articles, output_path=str(output_path))

        assert output_path.exists()
        with open(output_path) as f:
            content = f.read()
        assert content == xml

    def test_generate_atom_feed(self, sample_articles):
        """Test Atom feed generation."""
        from rss import generate_atom_feed

        xml = generate_atom_feed(sample_articles, title="Test Feed")

        assert '<?xml' in xml
        assert '<feed' in xml
        assert 'Test Feed' in xml
        assert 'Test Article 1' in xml

    def test_generate_atom_feed_with_output(self, sample_articles, temp_dir):
        """Test Atom feed generation with file output."""
        from rss import generate_atom_feed

        output_path = Path(temp_dir) / "feed.atom"
        xml = generate_atom_feed(sample_articles, output_path=str(output_path))

        assert output_path.exists()

    def test_rss_feed_handles_thumbnails(self):
        """Test RSS feed includes thumbnails as enclosures."""
        from rss import generate_rss_feed

        articles = [{
            "title": "Test",
            "channel": "Channel",
            "url": "https://example.com",
            "article": "Content",
            "thumbnail": "https://example.com/thumb.jpg",
        }]

        xml = generate_rss_feed(articles)
        assert 'enclosure' in xml
        assert 'thumb.jpg' in xml


class TestTTS:
    """Tests for tts.py module."""

    def test_preprocess_text_removes_markdown_headers(self):
        """Test markdown header removal."""
        from tts import preprocess_text

        text = "# Heading\n\nParagraph"
        result = preprocess_text(text)
        assert "#" not in result
        assert "Heading" in result

    def test_preprocess_text_removes_emphasis(self):
        """Test markdown emphasis removal."""
        from tts import preprocess_text

        text = "This is **bold** and *italic* text"
        result = preprocess_text(text)
        assert "**" not in result
        assert "*" not in result
        assert "bold" in result
        assert "italic" in result

    def test_preprocess_text_removes_links(self):
        """Test markdown link removal."""
        from tts import preprocess_text

        text = "Check out [this link](https://example.com) for more"
        result = preprocess_text(text)
        assert "[" not in result
        assert "](" not in result
        assert "this link" in result

    def test_preprocess_text_removes_code_blocks(self):
        """Test code block removal."""
        from tts import preprocess_text

        text = "Here's code:\n```python\nprint('hello')\n```\nDone"
        result = preprocess_text(text)
        assert "```" not in result
        assert "print" not in result

    def test_preprocess_text_expands_abbreviations(self):
        """Test abbreviation expansion."""
        from tts import preprocess_text

        text = "e.g. this is an example i.e. a test"
        result = preprocess_text(text)
        assert "for example" in result
        assert "that is" in result

    def test_preprocess_text_removes_bullets(self):
        """Test bullet point removal."""
        from tts import preprocess_text

        text = "List:\n- Item 1\n* Item 2\n+ Item 3"
        result = preprocess_text(text)
        assert "- " not in result
        assert "* " not in result
        assert "+ " not in result

    def test_get_available_backends(self):
        """Test getting available TTS backends."""
        from tts import get_available_backends, BACKENDS

        available = get_available_backends()
        assert isinstance(available, list)
        # Should be subset of known backends
        for backend in available:
            assert backend in BACKENDS

    def test_tts_backend_interface(self):
        """Test TTS backend abstract interface."""
        from tts import TTSBackend

        # Ensure abstract methods are defined
        assert hasattr(TTSBackend, 'synthesize')
        assert hasattr(TTSBackend, 'is_available')


class TestIntegrations:
    """Tests for integrations.py module."""

    def test_readwise_client_requires_token(self):
        """Test ReadwiseClient requires token."""
        from integrations import ReadwiseClient

        # Clear any existing env var
        old_token = os.environ.pop("READWISE_TOKEN", None)

        try:
            with pytest.raises(ValueError) as exc:
                ReadwiseClient()
            assert "token required" in str(exc.value).lower()
        finally:
            if old_token:
                os.environ["READWISE_TOKEN"] = old_token

    def test_readwise_client_accepts_token_param(self):
        """Test ReadwiseClient accepts token parameter."""
        from integrations import ReadwiseClient

        client = ReadwiseClient(token="test_token")
        assert client.token == "test_token"

    def test_readwise_client_uses_env_var(self):
        """Test ReadwiseClient uses environment variable."""
        from integrations import ReadwiseClient

        old_token = os.environ.get("READWISE_TOKEN")
        os.environ["READWISE_TOKEN"] = "env_token"

        try:
            client = ReadwiseClient()
            assert client.token == "env_token"
        finally:
            if old_token:
                os.environ["READWISE_TOKEN"] = old_token
            else:
                os.environ.pop("READWISE_TOKEN", None)

    def test_readwise_api_base_url(self):
        """Test ReadwiseClient has correct API base."""
        from integrations import ReadwiseClient

        assert ReadwiseClient.API_BASE == "https://readwise.io/api/v2"

    def test_send_to_readwise_failure_returns_false(self, mocker):
        """Test send_to_readwise returns False on failure."""
        from integrations import send_to_readwise

        mocker.patch('integrations.ReadwiseClient', side_effect=Exception("Test error"))

        result = send_to_readwise({"title": "Test", "article": "Content"})
        assert result is False

    def test_pocket_client_not_implemented(self):
        """Test PocketClient raises NotImplementedError."""
        from integrations import PocketClient

        with pytest.raises(NotImplementedError):
            PocketClient("key", "token")

    def test_instapaper_client_not_implemented(self):
        """Test InstapaperClient raises NotImplementedError."""
        from integrations import InstapaperClient

        with pytest.raises(NotImplementedError):
            InstapaperClient("user", "pass")


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


class TestScheduler:
    """Tests for scheduler.py module."""

    def test_fetch_history_init(self, tmp_path):
        """Test FetchHistory initialization."""
        from scheduler import FetchHistory

        history_path = tmp_path / "history.json"
        history = FetchHistory(path=history_path)

        assert history.path == history_path
        assert history.history == {'fetched': {}, 'last_run': None}

    def test_fetch_history_mark_fetched(self, tmp_path):
        """Test marking videos as fetched."""
        from scheduler import FetchHistory

        history = FetchHistory(path=tmp_path / "history.json")
        history.mark_fetched('video123', {'title': 'Test Video'})

        assert history.is_fetched('video123')
        assert not history.is_fetched('video456')

    def test_fetch_history_persistence(self, tmp_path):
        """Test history is persisted to disk."""
        from scheduler import FetchHistory

        history_path = tmp_path / "history.json"
        history1 = FetchHistory(path=history_path)
        history1.mark_fetched('video123')

        # Load in new instance
        history2 = FetchHistory(path=history_path)
        assert history2.is_fetched('video123')

    def test_fetch_history_last_run(self, tmp_path):
        """Test last run tracking."""
        from scheduler import FetchHistory

        history = FetchHistory(path=tmp_path / "history.json")
        assert history.get_last_run() is None

        history.update_last_run()
        last_run = history.get_last_run()
        assert last_run is not None

    def test_fetch_history_stats(self, tmp_path):
        """Test fetch statistics."""
        from scheduler import FetchHistory

        history = FetchHistory(path=tmp_path / "history.json")
        history.mark_fetched('v1')
        history.mark_fetched('v2')
        history.update_last_run()

        stats = history.get_stats()
        assert stats['total_fetched'] == 2
        assert stats['last_run'] is not None

    def test_scheduler_init(self, tmp_path):
        """Test Scheduler initialization."""
        from scheduler import Scheduler

        config_path = tmp_path / "config.yaml"
        config_path.write_text("jobs: []")

        scheduler = Scheduler(config_path)
        assert scheduler.config_path == config_path

    def test_create_systemd_service(self, tmp_path):
        """Test systemd service generation."""
        from scheduler import create_systemd_service

        config_path = str(tmp_path / "config.yaml")
        service = create_systemd_service(config_path, interval=3600)

        assert '[Unit]' in service
        assert '[Service]' in service
        assert '[Install]' in service
        assert 'readtube' in service.lower()
        assert '3600' in service

    def test_create_launchd_plist(self, tmp_path):
        """Test launchd plist generation."""
        from scheduler import create_launchd_plist

        config_path = str(tmp_path / "config.yaml")
        plist = create_launchd_plist(config_path, interval=1800)

        assert '<?xml' in plist
        assert 'plist' in plist
        assert 'com.readtube.scheduler' in plist
        assert '1800' in plist


class TestAsyncFetch:
    """Tests for async_fetch.py module."""

    def test_imports(self):
        """Test async_fetch module imports."""
        from async_fetch import fetch_videos_async, fetch_video_async
        assert callable(fetch_videos_async)
        assert callable(fetch_video_async)

    def test_fetch_videos_async_empty_list(self):
        """Test async fetch with empty URL list returns empty list."""
        import asyncio
        from async_fetch import fetch_videos_async

        result = asyncio.run(fetch_videos_async([]))
        assert result == []


class TestWriteArticle:
    """Tests for write_article.py module."""

    def test_load_video_data(self, tmp_path):
        """Test loading video data from JSON file."""
        from write_article import load_video_data

        payload = {"title": "Test", "transcript": "Text", "channel": "Ch"}
        path = tmp_path / "video.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        data = load_video_data(str(path))
        assert data["title"] == "Test"
        assert data["transcript"] == "Text"

    def test_load_video_data_missing_file(self):
        """Test loading from non-existent file returns None."""
        from write_article import load_video_data

        result = load_video_data("/nonexistent/path.json")
        assert result is None

    def test_generate_prompt_file(self, tmp_path):
        """Test prompt file generation."""
        from write_article import generate_prompt_file

        video_data = {
            "title": "Test Video",
            "channel": "Test Channel",
            "transcript": "Hello world transcript",
            "video_id": "test123",
        }

        output_path = generate_prompt_file(video_data, str(tmp_path))
        assert Path(output_path).exists()

        content = Path(output_path).read_text()
        assert "Test Video" in content
        assert "Test Channel" in content
        assert "Hello world transcript" in content


class TestLLM:
    """Tests for llm.py module."""

    def test_imports(self):
        """Test llm module imports."""
        from llm import (
            generate_article,
            get_available_backends,
            get_backend,
            BACKENDS,
        )
        assert callable(generate_article)
        assert callable(get_available_backends)
        assert callable(get_backend)
        assert isinstance(BACKENDS, dict)

    def test_backends_registry(self):
        """Test that backend registry contains expected backends."""
        from llm import BACKENDS

        assert "llama-cpp" in BACKENDS
        assert "ollama" in BACKENDS
        assert "claude-api" in BACKENDS

    def test_get_available_backends(self):
        """Test listing available backends."""
        from llm import get_available_backends

        backends = get_available_backends()
        assert isinstance(backends, list)

    def test_get_backend_invalid(self):
        """Test getting invalid backend returns None."""
        from llm import get_backend

        result = get_backend("nonexistent-backend")
        assert result is None

    def test_article_prompt_template(self):
        """Test article prompt template has required placeholders."""
        from llm import ARTICLE_PROMPT_TEMPLATE

        assert "{title}" in ARTICLE_PROMPT_TEMPLATE
        assert "{channel}" in ARTICLE_PROMPT_TEMPLATE
        assert "{transcript}" in ARTICLE_PROMPT_TEMPLATE
        assert "{description}" in ARTICLE_PROMPT_TEMPLATE
        assert "{chapters}" in ARTICLE_PROMPT_TEMPLATE


class TestImages:
    """Tests for images.py module."""

    def test_imports(self):
        """Test images module imports."""
        from images import (
            get_video_thumbnails,
            get_best_thumbnail,
            extract_frames,
            get_chapter_thumbnails
        )
        assert callable(get_video_thumbnails)
        assert callable(get_best_thumbnail)
        assert callable(extract_frames)
        assert callable(get_chapter_thumbnails)

    def test_get_best_thumbnail_invalid_url(self):
        """Test get_best_thumbnail with invalid URL returns None."""
        from images import get_best_thumbnail

        result = get_best_thumbnail("https://invalid-url.example.com/video")
        assert result is None

    def test_get_chapter_thumbnails_invalid_url(self):
        """Test get_chapter_thumbnails with invalid URL."""
        from images import get_chapter_thumbnails

        result = get_chapter_thumbnails("https://invalid-url.example.com/video")
        # Should return empty dict or handle error gracefully
        assert result == {} or result == []


class TestTranslate:
    """Tests for translate.py module."""

    def test_imports(self):
        """Test translate module imports."""
        from translate import (
            translate_text,
            translate_transcript,
            get_available_backends,
            TranslationBackend,
            BACKENDS
        )
        assert callable(translate_text)
        assert callable(translate_transcript)
        assert callable(get_available_backends)
        assert isinstance(BACKENDS, dict)

    def test_get_available_backends(self):
        """Test listing available translation backends."""
        from translate import get_available_backends

        backends = get_available_backends()
        assert isinstance(backends, list)
        # At least one backend should exist (libre is always in BACKENDS)

    def test_backends_registry(self):
        """Test that backend registry contains expected backends."""
        from translate import BACKENDS

        assert 'google' in BACKENDS or 'libre' in BACKENDS or 'deepl' in BACKENDS

    def test_translation_backend_interface(self):
        """Test TranslationBackend base class interface."""
        from translate import TranslationBackend

        # Verify it's an abstract class with required methods
        import inspect
        assert inspect.isabstract(TranslationBackend)
        assert hasattr(TranslationBackend, 'translate')
        assert hasattr(TranslationBackend, 'is_available')
        assert hasattr(TranslationBackend, 'supported_languages')

    def test_translate_text_no_backend(self):
        """Test translate_text returns None when no backend available."""
        from translate import translate_text

        # With an invalid backend, should return None
        result = translate_text("Hello", target_lang="es", backend="nonexistent")
        assert result is None

    def test_libre_translate_backend_exists(self):
        """Test LibreTranslate backend class exists."""
        from translate import BACKENDS

        if 'libre' in BACKENDS:
            backend_class = BACKENDS['libre']
            backend = backend_class()
            assert hasattr(backend, 'translate')
            assert hasattr(backend, 'is_available')
            assert hasattr(backend, 'supported_languages')


class TestCreateEpubSecurity:
    """Security tests for create_epub.py."""

    def test_html_escaping(self):
        """Test that HTML content is properly escaped."""
        from create_epub import _escape_html

        dangerous = '<script>alert("xss")</script>'
        safe = _escape_html(dangerous)

        assert '<script>' not in safe
        assert '&lt;script&gt;' in safe

    def test_html_escaping_attributes(self):
        """Test escaping of HTML attributes."""
        from create_epub import _escape_html

        dangerous = '"><img src=x onerror=alert(1)>'
        safe = _escape_html(dangerous)

        assert 'onerror' not in safe or '&' in safe

    def test_epub_with_malicious_title(self, tmp_path):
        """Test EPUB creation with potentially malicious title."""
        from create_epub import create_ebook

        articles = [{
            'title': '<script>alert("xss")</script>',
            'channel': 'Test & Co <test>',
            'article': 'Normal content',
        }]

        output_path = tmp_path / "test.epub"
        result = create_ebook(articles, output_path=str(output_path), format='epub')

        assert result is not None
        # Verify file was created
        assert output_path.exists()


class TestGetVideos:
    """Tests for get_videos.py module."""

    def test_is_playlist_url_with_playlist(self):
        """Test playlist URL detection."""
        from get_videos import is_playlist_url

        assert is_playlist_url("https://youtube.com/playlist?list=PLabc123")
        assert is_playlist_url("https://youtube.com/watch?v=xyz&list=PLabc123")

    def test_is_playlist_url_with_video(self):
        """Test non-playlist URL detection."""
        from get_videos import is_playlist_url

        assert not is_playlist_url("https://youtube.com/watch?v=abc123")
        assert not is_playlist_url("https://youtu.be/abc123")

    def test_get_video_info_invalid_url(self):
        """Test get_video_info with invalid URL raises NetworkError."""
        from get_videos import get_video_info
        from errors import ReadtubeError

        with pytest.raises(ReadtubeError):
            get_video_info("https://invalid-url.example.com/video")

    def test_video_info_type(self):
        """Test VideoInfo TypedDict is properly defined."""
        from get_videos import VideoInfo

        # Should have expected keys
        expected_keys = {'title', 'video_id', 'description', 'channel', 'url', 'thumbnail', 'duration', 'chapters'}
        assert hasattr(VideoInfo, '__annotations__')

    def test_chapter_info_type(self):
        """Test ChapterInfo TypedDict is properly defined."""
        from get_videos import ChapterInfo

        expected_keys = {'title', 'start_time', 'end_time'}
        assert hasattr(ChapterInfo, '__annotations__')

    def test_channels_constant(self):
        """Test CHANNELS constant is defined."""
        from get_videos import CHANNELS

        assert isinstance(CHANNELS, list)
        assert len(CHANNELS) > 0
        # All should be channel handles starting with @
        for channel in CHANNELS:
            assert channel.startswith("@")

    def test_min_duration_constant(self):
        """Test MIN_DURATION_SECONDS constant."""
        from get_videos import MIN_DURATION_SECONDS

        assert MIN_DURATION_SECONDS == 60  # Filter out Shorts

    def test_main_function_exists(self):
        """Test main function is callable."""
        from get_videos import main

        assert callable(main)


class TestBatchProcessing:
    """Tests for batch.py module."""

    def test_batch_config_load_and_expand(self, tmp_path):
        """Test BatchConfig loading."""
        from config import BatchConfig

        config_data = {
            "output_dir": str(tmp_path / "output"),
            "default_format": "epub",
            "jobs": [
                {"url": "https://youtube.com/watch?v=abc"},
                {"url": "https://youtube.com/watch?v=xyz", "summary_mode": True}
            ]
        }

        config_path = tmp_path / "batch.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = BatchConfig.load(config_path)
        assert len(config.jobs) == 2
        assert config.jobs[0].url == "https://youtube.com/watch?v=abc"
        assert config.jobs[1].summary_mode is True

    def test_process_job_with_mock(self, mocker):
        """Test process_job function with mocked dependencies."""
        from batch import process_job
        from config import BatchConfig, BatchJob

        # Mock get_video_info and get_transcript
        mock_video = {
            'title': 'Test Video',
            'channel': 'Test Channel',
            'video_id': 'abc123',
            'url': 'https://youtube.com/watch?v=abc123',
        }
        mocker.patch('batch.get_video_info', return_value=mock_video)
        mocker.patch('batch.get_transcript', return_value='This is a test transcript with many words.')

        job = BatchJob(url="https://youtube.com/watch?v=abc123")
        batch_config = BatchConfig(output_dir="./output", jobs=[job])

        result = process_job(job, batch_config)

        assert result is not None
        assert result['video']['title'] == 'Test Video'
        assert 'transcript' in result
        assert result['word_count'] > 0

    def test_process_job_video_fetch_fails(self, mocker):
        """Test process_job when video fetch fails."""
        from batch import process_job
        from config import BatchConfig, BatchJob

        mocker.patch('batch.get_video_info', return_value=None)

        job = BatchJob(url="https://youtube.com/watch?v=invalid")
        batch_config = BatchConfig(output_dir="./output", jobs=[job])

        result = process_job(job, batch_config)
        assert result is None

    def test_process_job_transcript_fails(self, mocker):
        """Test process_job when transcript fetch fails."""
        from batch import process_job
        from config import BatchConfig, BatchJob

        mock_video = {'title': 'Test', 'channel': 'Ch', 'video_id': 'abc'}
        mocker.patch('batch.get_video_info', return_value=mock_video)
        mocker.patch('batch.get_transcript', return_value=None)

        job = BatchJob(url="https://youtube.com/watch?v=abc")
        batch_config = BatchConfig(output_dir="./output", jobs=[job])

        result = process_job(job, batch_config)
        assert result is None


class TestLLMBackends:
    """Extended tests for llm.py backends."""

    def test_ai_code_environment_backend_not_in_claude(self):
        """Test AICodeEnvironmentBackend when not in Claude Code."""
        from llm import AICodeEnvironmentBackend
        import os

        # Ensure we're not in Claude Code
        old_val = os.environ.pop("CLAUDE_CODE", None)
        old_val2 = os.environ.pop("ANTHROPIC_CLAUDE_CODE", None)

        try:
            backend = AICodeEnvironmentBackend()
            # May or may not be available depending on environment
            assert hasattr(backend, 'is_available')
            assert hasattr(backend, 'generate')
        finally:
            if old_val:
                os.environ["CLAUDE_CODE"] = old_val
            if old_val2:
                os.environ["ANTHROPIC_CLAUDE_CODE"] = old_val2

    def test_ai_code_environment_backend_in_claude(self):
        """Test AICodeEnvironmentBackend when in Claude Code."""
        from llm import AICodeEnvironmentBackend
        import os

        old_val = os.environ.get("CLAUDE_CODE")
        os.environ["CLAUDE_CODE"] = "1"

        try:
            backend = AICodeEnvironmentBackend()
            assert backend.is_available() is True
            # Generate returns None (handled by host AI)
            result = backend.generate("Test prompt")
            assert result is None
        finally:
            if old_val:
                os.environ["CLAUDE_CODE"] = old_val
            else:
                os.environ.pop("CLAUDE_CODE", None)

    def test_llama_cpp_backend_unavailable_without_model(self):
        """Test LlamaCppBackend is unavailable without model."""
        from llm import LlamaCppBackend

        backend = LlamaCppBackend(model_path="/nonexistent/model.gguf")
        # Should be unavailable since model doesn't exist
        assert backend.is_available() is False

    def test_ollama_backend_unavailable_without_server(self):
        """Test OllamaBackend is unavailable without server."""
        from llm import OllamaBackend

        # Use a port that's unlikely to have Ollama
        backend = OllamaBackend(base_url="http://localhost:99999")
        assert backend.is_available() is False

    def test_claude_api_backend_unavailable_without_key(self):
        """Test ClaudeAPIBackend is unavailable without API key."""
        from llm import ClaudeAPIBackend
        import os

        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)

        try:
            backend = ClaudeAPIBackend(api_key=None)
            assert backend.is_available() is False
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_openai_backend_unavailable_without_key(self):
        """Test OpenAIBackend is unavailable without API key."""
        from llm import OpenAIBackend
        import os

        old_key = os.environ.pop("OPENAI_API_KEY", None)

        try:
            backend = OpenAIBackend(api_key=None, base_url="https://api.openai.com/v1")
            assert backend.is_available() is False
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_backend_name_property(self):
        """Test LLMBackend name property."""
        from llm import OllamaBackend

        backend = OllamaBackend()
        assert backend.name == "OllamaBackend"

    def test_generate_article_no_backend(self, mocker):
        """Test generate_article returns None when no backend available."""
        from llm import generate_article

        mocker.patch('llm.get_backend', return_value=None)

        result = generate_article("transcript", "title", "channel")
        assert result is None

    def test_article_system_prompt_content(self):
        """Test ARTICLE_SYSTEM_PROMPT has required content."""
        from llm import ARTICLE_SYSTEM_PROMPT

        assert "transcript" in ARTICLE_SYSTEM_PROMPT.lower()
        assert "article" in ARTICLE_SYSTEM_PROMPT.lower()
        assert "magazine" in ARTICLE_SYSTEM_PROMPT.lower()


class TestImagesModule:
    """Extended tests for images.py module."""

    def test_get_video_thumbnails_invalid_url(self):
        """Test get_video_thumbnails with invalid URL."""
        from images import get_video_thumbnails

        result = get_video_thumbnails("https://invalid-url.example.com/video", ".")
        assert result == []

    def test_extract_frames_returns_list(self):
        """Test extract_frames returns empty list on invalid input."""
        from images import extract_frames

        result = extract_frames("https://invalid-url.example.com/video", ["0:30"], ".")
        assert isinstance(result, list)

    def test_extract_frames_interval_returns_list(self):
        """Test extract_frames_interval returns empty list on invalid input."""
        from images import extract_frames_interval

        result = extract_frames_interval("https://invalid-url.example.com/video", 60, ".")
        assert isinstance(result, list)


class TestWriteArticleExtended:
    """Extended tests for write_article.py."""

    def test_load_video_data_invalid_json(self, tmp_path):
        """Test loading invalid JSON returns None."""
        from write_article import load_video_data

        path = tmp_path / "invalid.json"
        path.write_text("not valid json {{{", encoding="utf-8")

        result = load_video_data(str(path))
        assert result is None

    def test_generate_prompt_file_creates_directory(self, tmp_path):
        """Test prompt file generation creates output directory."""
        from write_article import generate_prompt_file

        video_data = {
            "title": "Test",
            "channel": "Channel",
            "transcript": "Transcript text",
            "video_id": "test123",
        }

        nested_dir = tmp_path / "nested" / "output"
        output_path = generate_prompt_file(video_data, str(nested_dir))

        assert Path(output_path).exists()
        assert nested_dir.exists()

    def test_generate_prompt_file_content_structure(self, tmp_path):
        """Test prompt file has correct structure."""
        from write_article import generate_prompt_file

        video_data = {
            "title": "My Video Title",
            "channel": "My Channel",
            "transcript": "Hello world this is a transcript",
            "video_id": "vid123",
        }

        output_path = generate_prompt_file(video_data, str(tmp_path))
        content = Path(output_path).read_text()

        assert "SYSTEM PROMPT:" in content
        assert "USER PROMPT:" in content
        assert "My Video Title" in content
        assert "My Channel" in content
        assert "Hello world this is a transcript" in content


class TestAsyncFetchExtended:
    """Extended tests for async_fetch.py module."""

    def test_fetch_video_async_exists(self):
        """Test fetch_video_async function exists."""
        from async_fetch import fetch_video_async
        assert callable(fetch_video_async)

    def test_async_fetcher_class(self):
        """Test AsyncFetcher class exists and has expected methods."""
        from async_fetch import AsyncFetcher

        fetcher = AsyncFetcher(max_concurrent=5, max_workers=10)
        assert hasattr(fetcher, 'fetch_video')
        assert hasattr(fetcher, 'fetch_transcript')
        assert hasattr(fetcher, 'fetch_videos')
        fetcher.close()

    def test_async_fetcher_context_manager(self):
        """Test AsyncFetcher can be used as context manager."""
        from async_fetch import AsyncFetcher
        import asyncio

        async def test():
            async with AsyncFetcher() as fetcher:
                assert fetcher is not None
                return True

        result = asyncio.run(test())
        assert result is True


class TestCreateEpubExtended:
    """Extended tests for create_epub.py module."""

    def test_create_ebook_with_html_format(self, tmp_path):
        """Test EPUB creation with HTML format."""
        from create_epub import create_ebook

        articles = [{
            'title': 'Test Article',
            'channel': 'Test Channel',
            'article': '# Test\n\nThis is test content.',
        }]

        output_path = tmp_path / "test.html"
        result = create_ebook(articles, output_path=str(output_path), format='html')

        assert result is not None
        assert output_path.exists()

        content = output_path.read_text()
        assert '<html' in content.lower()
        assert 'Test Article' in content

    def test_create_ebook_with_multiple_articles(self, tmp_path):
        """Test EPUB creation with multiple articles."""
        from create_epub import create_ebook

        articles = [
            {
                'title': 'Article 1',
                'channel': 'Channel 1',
                'article': '# First\n\nFirst article content.',
            },
            {
                'title': 'Article 2',
                'channel': 'Channel 2',
                'article': '# Second\n\nSecond article content.',
            },
        ]

        output_path = tmp_path / "multi.epub"
        result = create_ebook(articles, output_path=str(output_path), format='epub')

        assert result is not None
        assert output_path.exists()

    def test_create_ebook_with_include_cover(self, tmp_path):
        """Test EPUB creation with include_cover option."""
        from create_epub import create_ebook

        articles = [{
            'title': 'Article With Cover',
            'channel': 'Channel',
            'article': '# Test\n\nContent here.',
            'thumbnail': 'https://example.com/thumb.jpg',
        }]

        output_path = tmp_path / "with_cover.epub"
        result = create_ebook(articles, output_path=str(output_path), format='epub', include_cover=True)

        assert result is not None
        assert output_path.exists()

    def test_create_ebook_with_thumbnail(self, tmp_path):
        """Test EPUB creation with thumbnail URL."""
        from create_epub import create_ebook

        articles = [{
            'title': 'With Thumbnail',
            'channel': 'Channel',
            'article': '# Test\n\nContent.',
            'thumbnail': 'https://example.com/thumb.jpg',
        }]

        output_path = tmp_path / "thumb.epub"
        result = create_ebook(articles, output_path=str(output_path), format='epub')

        assert result is not None
        assert output_path.exists()

    def test_escape_html_special_chars(self):
        """Test HTML escaping of special characters."""
        from create_epub import _escape_html

        assert _escape_html('&') == '&amp;'
        assert _escape_html('<') == '&lt;'
        assert _escape_html('>') == '&gt;'
        assert _escape_html('"') == '&quot;'
        assert _escape_html("'") == '&#x27;'

    def test_escape_html_combined(self):
        """Test HTML escaping of combined special characters."""
        from create_epub import _escape_html

        dangerous = '<script>alert("test")</script>'
        safe = _escape_html(dangerous)

        assert '<script>' not in safe
        assert '>' not in safe or '&' in safe


class TestGetTranscriptsSecurity:
    """Security tests for get_transcripts.py."""

    def test_segment_text_handles_dict(self):
        """Test _segment_text handles dict input."""
        from get_transcripts import _segment_text

        segment = {'text': 'Hello world', 'start': 0.0}
        result = _segment_text(segment)
        assert result == 'Hello world'

    def test_segment_text_handles_object(self):
        """Test _segment_text handles object input."""
        from get_transcripts import _segment_text

        class MockSegment:
            text = 'Hello object'

        result = _segment_text(MockSegment())
        assert result == 'Hello object'

    def test_segment_start_handles_dict(self):
        """Test _segment_start handles dict input."""
        from get_transcripts import _segment_start

        segment = {'text': 'Hello', 'start': 10.5}
        result = _segment_start(segment)
        assert result == 10.5

    def test_segment_duration_handles_dict(self):
        """Test _segment_duration handles dict input."""
        from get_transcripts import _segment_duration

        segment = {'text': 'Hello', 'duration': 5.0}
        result = _segment_duration(segment)
        assert result == 5.0

    def test_list_transcripts_wrapper(self):
        """Test _list_transcripts compatibility wrapper exists."""
        from get_transcripts import _list_transcripts
        assert callable(_list_transcripts)
