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


class TestWeb:
    """Tests for web.py module."""

    @pytest.fixture
    def flask_app(self):
        """Create Flask test app."""
        pytest.importorskip("flask")
        from web import app
        app.config['TESTING'] = True
        return app

    @pytest.fixture
    def client(self, flask_app):
        """Create Flask test client."""
        return flask_app.test_client()

    def test_index_route(self, client):
        """Test index page returns HTML."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Readtube' in response.data
        assert b'<!DOCTYPE html>' in response.data

    def test_jobs_api_empty(self, client):
        """Test jobs API returns empty list initially."""
        from web import jobs
        jobs.clear()

        response = client.get('/api/jobs')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []

    def test_fetch_api_missing_url(self, client):
        """Test fetch API requires URL."""
        response = client.post(
            '/api/fetch',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_job_detail_not_found(self, client):
        """Test job detail returns 404 for missing job."""
        response = client.get('/api/jobs/nonexistent')
        assert response.status_code == 404


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)
