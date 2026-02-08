#!/usr/bin/env python3
"""
Readtube Web UI — FastAPI + Jinja2 + HTMX.
A local web interface to add YouTube URLs, browse generated articles, and download EPUBs.

Usage:
    python web.py                     # starts on http://127.0.0.1:8000
    uvicorn web:app --port 8000       # same, via uvicorn directly
"""

from __future__ import annotations

import os
import re
import tempfile
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, Form, Query, Request, UploadFile, File
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import web_db as db
import web_worker as worker

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))


# ── Lifespan ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db.init_db()
    worker.start()
    yield
    worker.stop()


app = FastAPI(title="Readtube", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ── Helpers ───────────────────────────────────────────────

_YT_VIDEO_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/)([A-Za-z0-9_-]{11})"
)
_YT_PLAYLIST_RE = re.compile(r"[?&]list=([A-Za-z0-9_-]+)")
_YT_CHANNEL_RE = re.compile(r"youtube\.com/(@[\w.-]+)")

_VALID_YT_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com)/"
)


def _is_valid_youtube_url(url: str) -> bool:
    """Check that the URL looks like a YouTube URL."""
    return bool(_VALID_YT_RE.match(url))


def _classify_url(url: str) -> str:
    """Return 'video', 'playlist', or 'channel' based on URL patterns."""
    if _YT_PLAYLIST_RE.search(url):
        return "playlist"
    if _YT_CHANNEL_RE.search(url):
        return "channel"
    return "video"


def _extract_video_id(url: str) -> Optional[str]:
    m = _YT_VIDEO_RE.search(url)
    return m.group(1) if m else None


def _settings_context() -> Dict[str, Any]:
    """Return current settings dict for templates."""
    return db.settings_get_all()


# ── Page routes ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    q: str = Query("", alias="q"),
    status: str = Query("", alias="status"),
) -> Response:
    search = q.strip() or None
    status_filter = status.strip() or None
    articles = db.article_list(status=status_filter, search=search)
    jobs = db.job_list_active()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "articles": articles,
        "jobs": jobs,
        "settings": _settings_context(),
        "search_query": q,
        "status_filter": status,
    })


@app.post("/add", response_class=HTMLResponse)
async def add_url(request: Request, url: str = Form(...)) -> Response:
    url = url.strip()
    if not url:
        return templates.TemplateResponse("partials/article_list.html", {
            "request": request,
            "articles": db.article_list(),
        })

    if not _is_valid_youtube_url(url):
        articles = db.article_list()
        jobs = db.job_list_active()
        return templates.TemplateResponse("partials/article_list.html", {
            "request": request,
            "articles": articles,
            "jobs": jobs,
            "error": "Please enter a valid YouTube URL.",
        })

    source_type = _classify_url(url)

    if source_type == "video":
        video_id = _extract_video_id(url)
        if not video_id:
            video_id = url  # fallback — let yt-dlp figure it out
        art_id = db.article_upsert(video_id, url=url, status="pending")
        db.job_create("fetch_video", article_id=art_id)
    else:
        src_id = db.source_add(url, source_type=source_type, name="")
        db.job_create("fetch_source", source_id=src_id)

    articles = db.article_list()
    jobs = db.job_list_active()
    return templates.TemplateResponse("partials/article_list.html", {
        "request": request,
        "articles": articles,
        "jobs": jobs,
    })


@app.get("/article/{video_id}", response_class=HTMLResponse)
async def article_reader(request: Request, video_id: str) -> Response:
    article = db.article_get(video_id)
    if not article:
        return HTMLResponse("<h1>Article not found</h1>", status_code=404)

    from themes import get_theme
    settings = _settings_context()
    theme_name = settings.get("theme", "default")
    try:
        theme = get_theme(theme_name)
    except ValueError:
        from themes import THEME_DEFAULT
        theme = THEME_DEFAULT

    return templates.TemplateResponse("reader.html", {
        "request": request,
        "article": article,
        "theme_css": theme.css,
        "settings": settings,
    })


@app.get("/article/{video_id}/download")
async def article_download(
    video_id: str,
    format: str = Query("epub", alias="format"),
) -> Response:
    article = db.article_get(video_id)
    if not article or not article.get("article_md"):
        return HTMLResponse("<h1>Article not ready</h1>", status_code=404)

    safe_title = re.sub(r"[^\w\s-]", "", article["title"])[:50].strip() or "article"

    # Fast path for markdown download
    if format == "md":
        filename = f"{safe_title}.md"
        return Response(
            content=article["article_md"].encode("utf-8"),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    from create_epub import create_ebook

    articles_data = [{
        "title": article["title"],
        "channel": article["channel"],
        "url": article["url"],
        "article": article["article_md"],
        "thumbnail": article.get("thumbnail"),
    }]

    ext = format if format in ("epub", "pdf", "html") else "epub"
    filename = f"{safe_title}.{ext}"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, filename)
        result = create_ebook(articles_data, output_path=out_path, format=format)
        if result and os.path.exists(result):
            with open(result, "rb") as f:
                content = f.read()
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
    return HTMLResponse("<h1>Failed to generate ebook</h1>", status_code=500)


@app.post("/article/{video_id}/regenerate", response_class=HTMLResponse)
async def article_regenerate(request: Request, video_id: str) -> Response:
    article = db.article_get(video_id)
    if not article:
        return HTMLResponse("<h1>Article not found</h1>", status_code=404)
    db.article_update(video_id, status="pending", article_md=None, article_html=None, error=None)
    db.job_create("fetch_video", article_id=article["id"])
    return templates.TemplateResponse("partials/article_card.html", {
        "request": request,
        "article": db.article_get(video_id),
    })


@app.post("/article/{video_id}/send-kindle", response_class=HTMLResponse)
async def article_send_kindle(request: Request, video_id: str) -> Response:
    article = db.article_get(video_id)
    if not article or not article.get("article_md"):
        return HTMLResponse('<span class="text-error">Article not ready</span>')

    settings = _settings_context()
    kindle_email = settings.get("kindle_email", "").strip()
    smtp_host = settings.get("smtp_host", "").strip()
    smtp_from = settings.get("smtp_from", "").strip()

    if not kindle_email or not smtp_host or not smtp_from:
        return HTMLResponse('<span class="text-error">Configure Kindle &amp; SMTP in Settings first</span>')

    from create_epub import create_ebook
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders

    articles_data = [{
        "title": article["title"],
        "channel": article["channel"],
        "url": article["url"],
        "article": article["article_md"],
        "thumbnail": article.get("thumbnail"),
    }]

    safe_title = re.sub(r"[^\w\s-]", "", article["title"])[:50].strip() or "article"
    filename = f"{safe_title}.epub"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, filename)
            result = create_ebook(articles_data, output_path=out_path, format="epub")
            if not result or not os.path.exists(result):
                return HTMLResponse('<span class="text-error">Failed to generate EPUB</span>')
            with open(result, "rb") as f:
                epub_bytes = f.read()

        msg = MIMEMultipart()
        msg["From"] = smtp_from
        msg["To"] = kindle_email
        msg["Subject"] = safe_title

        part = MIMEBase("application", "epub+zip")
        part.set_payload(epub_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
        msg.attach(part)

        smtp_port = int(settings.get("smtp_port", "587"))
        smtp_user = settings.get("smtp_user", "").strip()
        smtp_pass = settings.get("smtp_pass", "").strip()

        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        return HTMLResponse('<span class="text-success">Sent to Kindle!</span>')
    except Exception as exc:
        return HTMLResponse(f'<span class="text-error">Send failed: {str(exc)[:100]}</span>')


@app.post("/article/{video_id}/delete", response_class=HTMLResponse)
async def article_delete(request: Request, video_id: str) -> Response:
    db.article_delete(video_id)
    return HTMLResponse("")


# ── Sources ───────────────────────────────────────────────

@app.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request) -> Response:
    sources = db.source_list()
    return templates.TemplateResponse("sources.html", {
        "request": request,
        "sources": sources,
        "settings": _settings_context(),
    })


@app.post("/sources/add", response_class=HTMLResponse)
async def sources_add(request: Request, url: str = Form(...), name: str = Form("")) -> Response:
    url = url.strip()
    error: Optional[str] = None
    if url and not _is_valid_youtube_url(url):
        error = "Please enter a valid YouTube URL."
    elif url:
        source_type = _classify_url(url)
        src_id = db.source_add(url, source_type=source_type, name=name.strip())
        db.job_create("fetch_source", source_id=src_id)
    sources = db.source_list()
    return templates.TemplateResponse("sources.html", {
        "request": request,
        "sources": sources,
        "settings": _settings_context(),
        "error": error,
    })


@app.post("/sources/import-opml", response_class=HTMLResponse)
async def sources_import_opml(request: Request, file: UploadFile = File(...)) -> Response:
    import xml.etree.ElementTree as ET

    try:
        content = await file.read()
        root = ET.fromstring(content)
    except Exception:
        return HTMLResponse('<div class="alert alert-error">Invalid OPML file</div>')

    count = 0
    for outline in root.iter("outline"):
        html_url = outline.get("htmlUrl", "").strip()
        xml_url = outline.get("xmlUrl", "").strip()
        title = outline.get("title", "").strip() or outline.get("text", "").strip()

        # Extract a usable YouTube URL
        url = ""
        if html_url and "youtube.com" in html_url:
            url = html_url
        elif xml_url and "youtube.com" in xml_url:
            # Convert feed URL to channel URL
            # e.g. https://www.youtube.com/feeds/videos.xml?channel_id=UC...
            import re as _re
            m = _re.search(r"channel_id=([A-Za-z0-9_-]+)", xml_url)
            if m:
                url = f"https://www.youtube.com/channel/{m.group(1)}"
            else:
                url = xml_url

        if not url:
            continue

        db.source_add(url, source_type="channel", name=title)
        count += 1

    return HTMLResponse(f'<div class="alert alert-success">Imported {count} sources from OPML</div>')


@app.post("/sources/{source_id}/toggle-auto-fetch", response_class=HTMLResponse)
async def sources_toggle_auto_fetch(request: Request, source_id: int) -> Response:
    source = db.source_get(source_id)
    if not source:
        return HTMLResponse("Source not found", status_code=404)
    new_val = 0 if source["auto_fetch"] else 1
    db.source_update(source_id, auto_fetch=new_val)
    source = db.source_get(source_id)
    return templates.TemplateResponse("partials/source_row.html", {
        "request": request,
        "source": source,
    })


@app.delete("/sources/{source_id}", response_class=HTMLResponse)
async def sources_delete(request: Request, source_id: int) -> Response:
    db.source_delete(source_id)
    return HTMLResponse("")


# ── Settings ──────────────────────────────────────────────

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> Response:
    from themes import list_themes
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings": _settings_context(),
        "themes": list_themes(),
    })


@app.post("/settings", response_class=HTMLResponse)
async def settings_save(request: Request) -> Response:
    form = await request.form()
    for key in ("llm_backend", "ollama_model", "theme", "anthropic_api_key", "openai_api_key",
                 "kindle_email", "smtp_host", "smtp_port", "smtp_user", "smtp_pass", "smtp_from"):
        val = form.get(key)
        if val is not None:
            db.setting_set(key, str(val))
    # auto_fetch_interval: form sends minutes, store as seconds
    afi = form.get("auto_fetch_interval")
    if afi is not None:
        try:
            db.setting_set("auto_fetch_interval", str(int(afi) * 60))
        except (ValueError, TypeError):
            pass

    from themes import list_themes
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings": _settings_context(),
        "themes": list_themes(),
        "saved": True,
    })


# ── HTMX API endpoints ───────────────────────────────────

@app.get("/api/articles", response_class=HTMLResponse)
async def api_articles(
    request: Request,
    q: str = Query("", alias="q"),
    status: str = Query("", alias="status"),
) -> Response:
    search = q.strip() or None
    status_filter = status.strip() or None
    articles = db.article_list(status=status_filter, search=search)
    jobs = db.job_list_active()
    return templates.TemplateResponse("partials/article_list.html", {
        "request": request,
        "articles": articles,
        "jobs": jobs,
    })


@app.get("/api/jobs", response_class=HTMLResponse)
async def api_jobs(request: Request) -> Response:
    jobs = db.job_list_active()
    return templates.TemplateResponse("partials/job_status.html", {
        "request": request,
        "jobs": jobs,
    })


# ── RSS ───────────────────────────────────────────────────

@app.get("/feed/rss")
async def rss_feed() -> Response:
    from datetime import datetime
    from rss import generate_rss_feed
    articles = db.article_list(status="done")
    feed_articles = []
    for a in articles:
        created = a.get("created_at")
        if isinstance(created, (int, float)):
            created = datetime.fromtimestamp(created)
        feed_articles.append({
            "title": a["title"],
            "channel": a["channel"],
            "url": a["url"],
            "article": a.get("article_md", ""),
            "thumbnail": a.get("thumbnail"),
            "created_at": created,
        })
    xml = generate_rss_feed(feed_articles)
    return Response(content=xml, media_type="application/rss+xml")


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("web:app", host="127.0.0.1", port=port, reload=True)
