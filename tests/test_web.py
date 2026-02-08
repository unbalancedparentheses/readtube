"""
End-to-end tests for the Readtube Web UI.

Tests the FastAPI routes, database CRUD, worker pipeline, and HTMX endpoints.
Uses httpx TestClient (no real network requests to YouTube — worker is mocked).

Run with:
    pytest tests/test_web.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
import time
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Generator[None, None, None]:
    """Use a temporary database for every test."""
    db_path = str(tmp_path / "test_web.db")
    monkeypatch.setattr("web_db._DB_PATH", db_path)
    import web_db as db
    # Reset thread-local so the next _get_conn() picks up the new path
    db._local.conn = None
    db.init_db()
    yield
    # Cleanup
    if hasattr(db._local, "conn") and db._local.conn:
        db._local.conn.close()
        db._local.conn = None


@pytest.fixture()
def client() -> Generator[Any, None, None]:
    """Create a test client with the worker disabled."""
    # Prevent the real worker from starting
    with patch("web_worker.start"), patch("web_worker.stop"):
        from httpx import ASGITransport, AsyncClient
        from web import app
        import asyncio

        transport = ASGITransport(app=app)

        async def _lifespan() -> None:
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                pass

        # We use the synchronous TestClient from httpx via starlette
        from starlette.testclient import TestClient
        with TestClient(app) as tc:
            yield tc


# ── Database CRUD Tests ───────────────────────────────────

class TestWebDB:
    """Tests for web_db.py CRUD operations."""

    def test_settings_default_values(self) -> None:
        import web_db as db
        assert db.setting_get("llm_backend") == "ollama"
        assert db.setting_get("theme") == "default"
        assert db.setting_get("ollama_model") == "llama3.2"

    def test_settings_set_and_get(self) -> None:
        import web_db as db
        db.setting_set("llm_backend", "claude-api")
        assert db.setting_get("llm_backend") == "claude-api"

    def test_settings_no_overwrite(self) -> None:
        import web_db as db
        db.setting_set("theme", "dark")
        db.setting_set("theme", "modern", overwrite=False)
        assert db.setting_get("theme") == "dark"

    def test_settings_get_all(self) -> None:
        import web_db as db
        all_settings = db.settings_get_all()
        assert isinstance(all_settings, dict)
        assert "llm_backend" in all_settings
        assert "theme" in all_settings

    def test_settings_get_missing_key(self) -> None:
        import web_db as db
        assert db.setting_get("nonexistent", "fallback") == "fallback"

    # ── Sources ───────────────────────────────────────────

    def test_source_add_and_list(self) -> None:
        import web_db as db
        src_id = db.source_add("https://youtube.com/playlist?list=PL123", source_type="playlist", name="My Playlist")
        assert src_id > 0
        sources = db.source_list()
        assert len(sources) == 1
        assert sources[0]["url"] == "https://youtube.com/playlist?list=PL123"
        assert sources[0]["source_type"] == "playlist"
        assert sources[0]["name"] == "My Playlist"

    def test_source_add_duplicate(self) -> None:
        import web_db as db
        id1 = db.source_add("https://youtube.com/@test")
        id2 = db.source_add("https://youtube.com/@test")
        assert id1 == id2
        assert len(db.source_list()) == 1

    def test_source_delete(self) -> None:
        import web_db as db
        src_id = db.source_add("https://youtube.com/@delete_me")
        db.source_delete(src_id)
        assert len(db.source_list()) == 0

    # ── Articles ──────────────────────────────────────────

    def test_article_upsert_and_get(self) -> None:
        import web_db as db
        art_id = db.article_upsert(
            "abc123",
            title="Test Video",
            channel="Test Channel",
            url="https://youtube.com/watch?v=abc123",
        )
        assert art_id > 0
        art = db.article_get("abc123")
        assert art is not None
        assert art["title"] == "Test Video"
        assert art["channel"] == "Test Channel"
        assert art["status"] == "pending"

    def test_article_upsert_updates_existing(self) -> None:
        import web_db as db
        db.article_upsert("abc123", title="Original")
        db.article_upsert("abc123", title="Updated")
        art = db.article_get("abc123")
        assert art["title"] == "Updated"

    def test_article_update(self) -> None:
        import web_db as db
        db.article_upsert("abc123", title="Test")
        db.article_update("abc123", status="done", article_md="# Hello")
        art = db.article_get("abc123")
        assert art["status"] == "done"
        assert art["article_md"] == "# Hello"

    def test_article_list_all(self) -> None:
        import web_db as db
        db.article_upsert("v1", title="Video 1")
        db.article_upsert("v2", title="Video 2")
        articles = db.article_list()
        assert len(articles) == 2

    def test_article_list_by_status(self) -> None:
        import web_db as db
        db.article_upsert("v1", title="Video 1", status="pending")
        db.article_upsert("v2", title="Video 2", status="pending")
        db.article_update("v1", status="done")
        assert len(db.article_list(status="done")) == 1
        assert len(db.article_list(status="pending")) == 1

    def test_article_delete(self) -> None:
        import web_db as db
        db.article_upsert("v1", title="Delete Me")
        db.article_delete("v1")
        assert db.article_get("v1") is None

    def test_article_get_nonexistent(self) -> None:
        import web_db as db
        assert db.article_get("nonexistent") is None

    # ── Jobs ──────────────────────────────────────────────

    def test_job_create_and_list(self) -> None:
        import web_db as db
        art_id = db.article_upsert("v1", title="Test")
        job_id = db.job_create("fetch_video", article_id=art_id)
        assert job_id > 0
        active = db.job_list_active()
        assert len(active) == 1
        assert active[0]["job_type"] == "fetch_video"

    def test_job_next_pending(self) -> None:
        import web_db as db
        art_id = db.article_upsert("v1", title="Test")
        db.job_create("fetch_video", article_id=art_id)
        job = db.job_next_pending()
        assert job is not None
        assert job["status"] == "pending"  # returned before update committed... but let's check DB
        # After claiming, it should be running in the DB
        active = db.job_list_active()
        assert len(active) == 1
        assert active[0]["status"] == "running"

    def test_job_next_pending_empty(self) -> None:
        import web_db as db
        assert db.job_next_pending() is None

    def test_job_update_status(self) -> None:
        import web_db as db
        art_id = db.article_upsert("v1", title="Test")
        job_id = db.job_create("fetch_video", article_id=art_id)
        db.job_update(job_id, status="done")
        # Should not appear in active list
        assert len(db.job_list_active()) == 0

    def test_job_list_all(self) -> None:
        import web_db as db
        art_id = db.article_upsert("v1", title="Test")
        src_id = db.source_add("https://youtube.com/@test")
        db.job_create("fetch_video", article_id=art_id)
        db.job_create("fetch_source", source_id=src_id)
        all_jobs = db.job_list_all()
        assert len(all_jobs) == 2


# ── Route Tests ───────────────────────────────────────────

class TestDashboardRoutes:
    """Tests for dashboard and article routes."""

    def test_get_dashboard(self, client: Any) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Readtube" in resp.text
        assert "Add a YouTube video" in resp.text

    def test_add_video_url(self, client: Any) -> None:
        resp = client.post("/add", data={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
        assert resp.status_code == 200
        import web_db as db
        articles = db.article_list()
        assert len(articles) == 1
        assert articles[0]["video_id"] == "dQw4w9WgXcQ"

    def test_add_playlist_url(self, client: Any) -> None:
        resp = client.post("/add", data={"url": "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"})
        assert resp.status_code == 200
        import web_db as db
        sources = db.source_list()
        assert len(sources) == 1
        assert sources[0]["source_type"] == "playlist"

    def test_add_channel_url(self, client: Any) -> None:
        resp = client.post("/add", data={"url": "https://www.youtube.com/@testchannel"})
        assert resp.status_code == 200
        import web_db as db
        sources = db.source_list()
        assert len(sources) == 1
        assert sources[0]["source_type"] == "channel"

    def test_add_empty_url(self, client: Any) -> None:
        resp = client.post("/add", data={"url": ""})
        # FastAPI will return 422 for missing required field
        assert resp.status_code in (200, 422)

    def test_article_reader_not_found(self, client: Any) -> None:
        resp = client.get("/article/nonexistent")
        assert resp.status_code == 404

    def test_article_reader_with_content(self, client: Any) -> None:
        import web_db as db
        db.article_upsert(
            "test123",
            title="Test Article",
            channel="Test Channel",
            url="https://youtube.com/watch?v=test123",
        )
        db.article_update(
            "test123",
            status="done",
            article_md="# Hello World",
            article_html="<h1>Hello World</h1>",
        )
        resp = client.get("/article/test123")
        assert resp.status_code == 200
        assert "Hello World" in resp.text
        assert "Test Article" in resp.text

    def test_article_regenerate(self, client: Any) -> None:
        import web_db as db
        art_id = db.article_upsert("regen123", title="Regen Test", url="https://youtube.com/watch?v=regen123")
        db.article_update("regen123", status="done", article_md="old content")
        resp = client.post("/article/regen123/regenerate")
        assert resp.status_code == 200
        art = db.article_get("regen123")
        assert art["status"] == "pending"
        assert art["article_md"] is None

    def test_article_delete(self, client: Any) -> None:
        import web_db as db
        db.article_upsert("del123", title="Delete Me")
        resp = client.post("/article/del123/delete")
        assert resp.status_code == 200
        assert db.article_get("del123") is None

    def test_article_download_not_ready(self, client: Any) -> None:
        import web_db as db
        db.article_upsert("dl123", title="No Article Yet")
        resp = client.get("/article/dl123/download?format=epub")
        assert resp.status_code == 404


class TestSourceRoutes:
    """Tests for source management routes."""

    def test_get_sources_page(self, client: Any) -> None:
        resp = client.get("/sources")
        assert resp.status_code == 200
        assert "Sources" in resp.text

    def test_add_source(self, client: Any) -> None:
        resp = client.post("/sources/add", data={
            "url": "https://www.youtube.com/@testchannel",
            "name": "Test Channel",
        })
        assert resp.status_code == 200
        import web_db as db
        sources = db.source_list()
        assert len(sources) == 1
        assert sources[0]["name"] == "Test Channel"

    def test_delete_source(self, client: Any) -> None:
        import web_db as db
        src_id = db.source_add("https://youtube.com/@delete_me")
        resp = client.delete(f"/sources/{src_id}")
        assert resp.status_code == 200
        assert len(db.source_list()) == 0


class TestSettingsRoutes:
    """Tests for settings routes."""

    def test_get_settings_page(self, client: Any) -> None:
        resp = client.get("/settings")
        assert resp.status_code == 200
        assert "Settings" in resp.text
        assert "ollama" in resp.text.lower()

    def test_save_settings(self, client: Any) -> None:
        resp = client.post("/settings", data={
            "llm_backend": "claude-api",
            "ollama_model": "mistral",
            "theme": "dark",
            "anthropic_api_key": "sk-test",
            "openai_api_key": "",
        })
        assert resp.status_code == 200
        assert "Settings saved" in resp.text
        import web_db as db
        assert db.setting_get("llm_backend") == "claude-api"
        assert db.setting_get("ollama_model") == "mistral"
        assert db.setting_get("theme") == "dark"
        assert db.setting_get("anthropic_api_key") == "sk-test"


class TestHTMXEndpoints:
    """Tests for HTMX polling endpoints."""

    def test_api_articles(self, client: Any) -> None:
        import web_db as db
        db.article_upsert("htmx1", title="HTMX Article", url="https://youtube.com/watch?v=htmx1")
        resp = client.get("/api/articles")
        assert resp.status_code == 200
        assert "HTMX Article" in resp.text

    def test_api_jobs(self, client: Any) -> None:
        import web_db as db
        art_id = db.article_upsert("j1", title="Job Test")
        db.job_create("fetch_video", article_id=art_id)
        resp = client.get("/api/jobs")
        assert resp.status_code == 200
        assert "fetch_video" in resp.text


class TestRSSFeed:
    """Tests for the RSS feed endpoint."""

    def test_rss_feed_empty(self, client: Any) -> None:
        resp = client.get("/feed/rss")
        assert resp.status_code == 200
        assert "application/rss+xml" in resp.headers["content-type"]
        assert "<rss" in resp.text

    def test_rss_feed_with_articles(self, client: Any) -> None:
        import web_db as db
        from datetime import datetime
        db.article_upsert("rss1", title="RSS Article", channel="RSS Channel", url="https://youtube.com/watch?v=rss1")
        db.article_update("rss1", status="done", article_md="# RSS Content")
        resp = client.get("/feed/rss")
        assert resp.status_code == 200
        assert "RSS Article" in resp.text


# ── Worker Pipeline Tests ─────────────────────────────────

class TestWorkerPipeline:
    """Tests for web_worker.py job processing (with mocked YouTube/LLM calls)."""

    def test_fetch_video_pipeline(self) -> None:
        """Test the full video pipeline with mocked externals."""
        import web_db as db
        import web_worker as worker

        art_id = db.article_upsert(
            "pipe123",
            url="https://www.youtube.com/watch?v=pipe123",
        )
        job_id = db.job_create("fetch_video", article_id=art_id)
        job = {"id": job_id, "job_type": "fetch_video", "article_id": art_id, "source_id": None}

        mock_info: Dict[str, Any] = {
            "title": "Pipeline Test",
            "video_id": "pipe123",
            "description": "Test description",
            "channel": "Test Channel",
            "url": "https://www.youtube.com/watch?v=pipe123",
            "thumbnail": "https://example.com/thumb.jpg",
            "duration": 300,
            "chapters": [],
        }

        with patch("get_videos.get_video_info", return_value=mock_info), \
             patch("get_transcripts.get_transcript", return_value="This is a test transcript about interesting topics."), \
             patch("llm.generate_article", return_value="# Generated Article\n\nThis is the article content."):
            worker._do_fetch_video(job)

        art = db.article_get("pipe123")
        assert art is not None
        assert art["status"] == "done"
        assert art["title"] == "Pipeline Test"
        assert art["article_md"] == "# Generated Article\n\nThis is the article content."
        assert "<h1>" in art["article_html"]

    def test_fetch_video_pipeline_no_info(self) -> None:
        """Test pipeline when video info is not found."""
        import web_db as db
        import web_worker as worker

        art_id = db.article_upsert("fail1", url="https://youtube.com/watch?v=fail1")
        job_id = db.job_create("fetch_video", article_id=art_id)
        job = {"id": job_id, "job_type": "fetch_video", "article_id": art_id, "source_id": None}

        with patch("get_videos.get_video_info", return_value=None):
            worker._do_fetch_video(job)

        art = db.article_get("fail1")
        assert art["status"] == "error"
        assert "Could not fetch video info" in (art.get("error") or "")

    def test_fetch_video_pipeline_no_transcript(self) -> None:
        """Test pipeline when transcript is not available."""
        import web_db as db
        import web_worker as worker

        art_id = db.article_upsert("fail2", url="https://youtube.com/watch?v=fail2")
        job_id = db.job_create("fetch_video", article_id=art_id)
        job = {"id": job_id, "job_type": "fetch_video", "article_id": art_id, "source_id": None}

        mock_info: Dict[str, Any] = {
            "title": "No Transcript",
            "video_id": "fail2",
            "description": "",
            "channel": "Test",
            "url": "https://youtube.com/watch?v=fail2",
            "thumbnail": None,
            "duration": 120,
            "chapters": [],
        }

        with patch("get_videos.get_video_info", return_value=mock_info), \
             patch("get_transcripts.get_transcript", return_value=None):
            worker._do_fetch_video(job)

        art = db.article_get("fail2")
        assert art["status"] == "error"

    def test_fetch_video_pipeline_llm_fails(self) -> None:
        """Test pipeline when LLM fails."""
        import web_db as db
        import web_worker as worker

        art_id = db.article_upsert("fail3", url="https://youtube.com/watch?v=fail3")
        job_id = db.job_create("fetch_video", article_id=art_id)
        job = {"id": job_id, "job_type": "fetch_video", "article_id": art_id, "source_id": None}

        mock_info: Dict[str, Any] = {
            "title": "LLM Fail",
            "video_id": "fail3",
            "description": "",
            "channel": "Test",
            "url": "https://youtube.com/watch?v=fail3",
            "thumbnail": None,
            "duration": 120,
            "chapters": [],
        }

        with patch("get_videos.get_video_info", return_value=mock_info), \
             patch("get_transcripts.get_transcript", return_value="Some transcript"), \
             patch("llm.generate_article", return_value=None):
            worker._do_fetch_video(job)

        art = db.article_get("fail3")
        assert art["status"] == "error"
        assert "LLM returned empty" in (art.get("error") or "")

    def test_fetch_source_pipeline_single_video(self) -> None:
        """Test source pipeline for a single video URL."""
        import web_db as db
        import web_worker as worker

        src_id = db.source_add("https://youtube.com/watch?v=src1", source_type="video")
        job_id = db.job_create("fetch_source", source_id=src_id)
        job = {"id": job_id, "job_type": "fetch_source", "article_id": None, "source_id": src_id}

        mock_info: Dict[str, Any] = {
            "title": "Source Video",
            "video_id": "src1",
            "description": "",
            "channel": "Src Channel",
            "url": "https://youtube.com/watch?v=src1",
            "thumbnail": None,
            "duration": 200,
            "chapters": [],
        }

        with patch("get_videos.get_video_info", return_value=mock_info), \
             patch("get_videos.is_playlist_url", return_value=False):
            worker._do_fetch_source(job)

        # Should have created an article and a fetch_video job
        art = db.article_get("src1")
        assert art is not None
        assert art["title"] == "Source Video"
        active_jobs = db.job_list_active()
        fetch_jobs = [j for j in active_jobs if j["job_type"] == "fetch_video"]
        assert len(fetch_jobs) >= 1

    def test_dispatch_unknown_job_type(self) -> None:
        """Test dispatch with unknown job type."""
        import web_db as db
        import web_worker as worker

        job_id = db.job_create("unknown_type")
        job = {"id": job_id, "job_type": "unknown_type", "article_id": None, "source_id": None}
        worker._dispatch(job)
        all_jobs = db.job_list_all()
        error_jobs = [j for j in all_jobs if j["status"] == "error"]
        assert len(error_jobs) == 1


# ── URL Classification Tests ─────────────────────────────

class TestURLClassification:
    """Tests for URL classification in web.py."""

    def test_classify_video_url(self) -> None:
        from web import _classify_url
        assert _classify_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "video"
        assert _classify_url("https://youtu.be/dQw4w9WgXcQ") == "video"

    def test_classify_playlist_url(self) -> None:
        from web import _classify_url
        assert _classify_url("https://www.youtube.com/playlist?list=PLtest") == "playlist"
        assert _classify_url("https://www.youtube.com/watch?v=abc&list=PLtest") == "playlist"

    def test_classify_channel_url(self) -> None:
        from web import _classify_url
        assert _classify_url("https://www.youtube.com/@testchannel") == "channel"

    def test_extract_video_id(self) -> None:
        from web import _extract_video_id
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert _extract_video_id("not a url") is None


# ── Download Tests ────────────────────────────────────────

class TestDownload:
    """Tests for EPUB download route."""

    def test_download_epub(self, client: Any) -> None:
        import web_db as db
        db.article_upsert(
            "epub1",
            title="EPUB Test",
            channel="Test Channel",
            url="https://youtube.com/watch?v=epub1",
        )
        db.article_update(
            "epub1",
            status="done",
            article_md="# EPUB Content\n\nThis is the article body for the EPUB test.",
        )
        resp = client.get("/article/epub1/download?format=epub")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        assert len(resp.content) > 100

    def test_download_html(self, client: Any) -> None:
        import web_db as db
        db.article_upsert(
            "html1",
            title="HTML Test",
            channel="Test Channel",
            url="https://youtube.com/watch?v=html1",
        )
        db.article_update(
            "html1",
            status="done",
            article_md="# HTML Content\n\nBody text here.",
        )
        resp = client.get("/article/html1/download?format=html")
        assert resp.status_code == 200
        assert len(resp.content) > 100
