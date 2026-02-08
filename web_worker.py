"""
Background worker for Readtube Web UI.
Uses a ThreadPoolExecutor with a polling thread to process jobs from the database.

Pipelines:
  fetch_video  — get info → transcript → LLM article → render HTML
  fetch_source — resolve playlist/channel → create article rows → enqueue fetch_video jobs
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import markdown

import web_db as db

logger = logging.getLogger(__name__)

_executor: Optional[ThreadPoolExecutor] = None
_poll_thread: Optional[threading.Thread] = None
_running: bool = False


def start(workers: int = 2) -> None:
    """Start the background worker pool and polling thread."""
    global _executor, _poll_thread, _running
    if _running:
        return
    _running = True
    _executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="readtube-worker")
    _poll_thread = threading.Thread(target=_poll_loop, daemon=True, name="readtube-poller")
    _poll_thread.start()
    logger.info("Background worker started with %d workers", workers)


def stop() -> None:
    """Shut down the worker pool gracefully."""
    global _running, _executor
    _running = False  # noqa: F841 — used by _poll_loop
    if _executor:
        _executor.shutdown(wait=False)
        _executor = None
    logger.info("Background worker stopped")


def _poll_loop() -> None:
    """Poll for pending jobs and submit them to the executor."""
    while _running:
        try:
            job = db.job_next_pending()
            if job and _executor:
                _executor.submit(_dispatch, job)
            else:
                time.sleep(2)
        except Exception:
            logger.exception("Poller error")
            time.sleep(5)


def _dispatch(job: Dict[str, Any]) -> None:
    """Route a job to the appropriate handler."""
    try:
        if job["job_type"] == "fetch_video":
            _do_fetch_video(job)
        elif job["job_type"] == "fetch_source":
            _do_fetch_source(job)
        else:
            db.job_update(job["id"], status="error", error=f"Unknown job type: {job['job_type']}")
    except Exception as exc:
        logger.exception("Job %s failed", job["id"])
        db.job_update(job["id"], status="error", error=str(exc)[:1000])
        if job.get("article_id"):
            # find the video_id for this article
            try:
                from web_db import _get_conn
                conn = _get_conn()
                cur = conn.cursor()
                cur.execute("SELECT video_id FROM articles WHERE id = ?", (job["article_id"],))
                row = cur.fetchone()
                if row:
                    db.article_update(row["video_id"], status="error", error=str(exc)[:1000])
            except Exception:
                pass


# ── Video pipeline ────────────────────────────────────────

def _do_fetch_video(job: Dict[str, Any]) -> None:
    """Full pipeline: info → transcript → article → HTML."""
    from get_videos import get_video_info
    from get_transcripts import get_transcript
    from llm import generate_article
    from errors import format_error_for_user

    article_id = job["article_id"]

    # Look up article row to get video_id / url
    with db._tx() as cur:
        cur.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
        row = cur.fetchone()
    if not row:
        db.job_update(job["id"], status="error", error="Article row not found")
        return

    video_id = row["video_id"]
    url = row["url"] or f"https://www.youtube.com/watch?v={video_id}"

    # ── Step 1: fetch video info ──
    db.article_update(video_id, status="fetching")
    try:
        info = get_video_info(url)
    except Exception as exc:
        msg = format_error_for_user(exc)
        db.article_update(video_id, status="error", error=msg)
        db.job_update(job["id"], status="error", error=msg)
        return

    if not info:
        db.article_update(video_id, status="error", error="Could not fetch video info")
        db.job_update(job["id"], status="error", error="Could not fetch video info")
        return

    db.article_update(
        video_id,
        title=info.get("title", ""),
        channel=info.get("channel", ""),
        description=info.get("description", ""),
        thumbnail=info.get("thumbnail"),
        duration=info.get("duration", 0),
        url=info.get("url", url),
    )

    # ── Step 2: fetch transcript ──
    db.article_update(video_id, status="transcribing")
    try:
        transcript = get_transcript(info["video_id"])
    except Exception as exc:
        msg = format_error_for_user(exc)
        db.article_update(video_id, status="error", error=msg)
        db.job_update(job["id"], status="error", error=msg)
        return

    if not transcript:
        db.article_update(video_id, status="error", error="Transcript not available")
        db.job_update(job["id"], status="error", error="Transcript not available")
        return

    db.article_update(video_id, transcript=transcript)

    # ── Step 3: generate article via LLM ──
    db.article_update(video_id, status="generating")
    settings = db.settings_get_all()
    backend = settings.get("llm_backend", "ollama")
    backend_kwargs = {}
    if backend == "ollama":
        model = settings.get("ollama_model", "llama3.2")
        backend_kwargs["model"] = model
    elif backend == "claude-api":
        api_key = settings.get("anthropic_api_key", "")
        if api_key:
            backend_kwargs["api_key"] = api_key
    elif backend == "openai":
        api_key = settings.get("openai_api_key", "")
        if api_key:
            backend_kwargs["api_key"] = api_key

    chapters_text = ""
    try:
        chapters = info.get("chapters", [])
        if chapters:
            chapters_text = "\n".join(f"- {ch['title']}" for ch in chapters)
    except Exception:
        pass

    try:
        article_md = generate_article(
            transcript=transcript,
            title=info.get("title", ""),
            channel=info.get("channel", ""),
            description=info.get("description"),
            chapters=chapters_text or None,
            backend=backend,
            **backend_kwargs,
        )
    except Exception as exc:
        msg = format_error_for_user(exc)
        db.article_update(video_id, status="error", error=msg)
        db.job_update(job["id"], status="error", error=msg)
        return

    if not article_md:
        db.article_update(video_id, status="error", error="LLM returned empty article")
        db.job_update(job["id"], status="error", error="LLM returned empty article")
        return

    # ── Step 4: render markdown → HTML ──
    article_html = markdown.markdown(article_md, extensions=["smarty"])

    db.article_update(video_id, status="done", article_md=article_md, article_html=article_html, error=None)
    db.job_update(job["id"], status="done")
    logger.info("Article done: %s", info.get("title", video_id))


# ── Source pipeline ───────────────────────────────────────

def _do_fetch_source(job: Dict[str, Any]) -> None:
    """Resolve a source (playlist/channel) into individual video jobs."""
    from get_videos import get_video_info, get_videos_from_playlist, get_latest_video_from_channel, is_playlist_url
    from errors import format_error_for_user

    source_id = job["source_id"]

    with db._tx() as cur:
        cur.execute("SELECT * FROM sources WHERE id = ?", (source_id,))
        row = cur.fetchone()
    if not row:
        db.job_update(job["id"], status="error", error="Source not found")
        return

    source = dict(row)
    url = source["url"]
    source_type = source["source_type"]

    try:
        videos = []
        if source_type == "playlist" or is_playlist_url(url):
            videos = get_videos_from_playlist(url)
        elif source_type == "channel":
            handle = url.rstrip("/").split("/")[-1]
            if not handle.startswith("@"):
                handle = f"@{handle}"
            video = get_latest_video_from_channel(handle)
            if video:
                videos = [video]
        else:
            info = get_video_info(url)
            if info:
                videos = [info]

        # Update source name from first result if empty
        if videos and not source["name"]:
            name = videos[0].get("channel", "") if source_type != "video" else videos[0].get("title", "")
            with db._tx() as cur:
                cur.execute("UPDATE sources SET name = ? WHERE id = ?", (name, source_id))

        for v in videos:
            vid = v["video_id"]
            art_id = db.article_upsert(
                vid,
                title=v.get("title", ""),
                channel=v.get("channel", ""),
                description=v.get("description", ""),
                thumbnail=v.get("thumbnail"),
                duration=v.get("duration", 0),
                url=v.get("url", ""),
                source_id=source_id,
            )
            # Only enqueue if article is still pending
            art = db.article_get(vid)
            if art and art["status"] == "pending":
                db.job_create("fetch_video", article_id=art_id)

        db.job_update(job["id"], status="done")
        logger.info("Source resolved: %s → %d videos", url, len(videos))

    except Exception as exc:
        msg = format_error_for_user(exc)
        db.job_update(job["id"], status="error", error=msg)
        logger.exception("Source fetch failed: %s", url)
