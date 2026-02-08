"""
SQLite database for Readtube Web UI.
Separate from the cache DB — stores sources, articles, jobs, and settings.

Follows the same patterns as cache.py: thread-local connections, WAL mode, sqlite3.Row factory.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional


_DB_PATH = os.path.join(os.environ.get("CACHE_DIR", ".cache"), "web.db")
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        db_path = Path(_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


@contextmanager
def _tx() -> Generator[sqlite3.Cursor, None, None]:
    """Context manager for database transactions."""
    conn = _get_conn()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    """Create all tables if they don't exist."""
    with _tx() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL DEFAULT 'video',   -- video | playlist | channel
                name        TEXT NOT NULL DEFAULT '',
                auto_fetch  INTEGER NOT NULL DEFAULT 0,
                created_at  REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id     TEXT NOT NULL UNIQUE,
                title        TEXT NOT NULL DEFAULT '',
                channel      TEXT NOT NULL DEFAULT '',
                description  TEXT NOT NULL DEFAULT '',
                thumbnail    TEXT,
                duration     INTEGER NOT NULL DEFAULT 0,
                url          TEXT NOT NULL DEFAULT '',
                transcript   TEXT,
                article_md   TEXT,
                article_html TEXT,
                status       TEXT NOT NULL DEFAULT 'pending',
                source_id    INTEGER REFERENCES sources(id) ON DELETE SET NULL,
                error        TEXT,
                created_at   REAL NOT NULL,
                updated_at   REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type   TEXT NOT NULL,          -- fetch_video | fetch_source
                article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
                source_id  INTEGER REFERENCES sources(id) ON DELETE CASCADE,
                status     TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | error
                error      TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        # Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")

    # Seed default settings
    defaults = {
        "llm_backend": "ollama",
        "ollama_model": "llama3.2",
        "theme": "default",
        "anthropic_api_key": "",
        "openai_api_key": "",
    }
    for k, v in defaults.items():
        setting_set(k, v, overwrite=False)


# ── Settings ──────────────────────────────────────────────

def setting_get(key: str, default: str = "") -> str:
    with _tx() as cur:
        cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default


def setting_set(key: str, value: str, overwrite: bool = True) -> None:
    with _tx() as cur:
        if overwrite:
            cur.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )
        else:
            cur.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (key, value),
            )


def settings_get_all() -> Dict[str, str]:
    with _tx() as cur:
        cur.execute("SELECT key, value FROM settings")
        return {row["key"]: row["value"] for row in cur.fetchall()}


# ── Sources ───────────────────────────────────────────────

def source_add(url: str, source_type: str = "video", name: str = "", auto_fetch: bool = False) -> int:
    now = time.time()
    with _tx() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO sources (url, source_type, name, auto_fetch, created_at) VALUES (?, ?, ?, ?, ?)",
            (url, source_type, name, int(auto_fetch), now),
        )
        if cur.rowcount == 0:
            cur.execute("SELECT id FROM sources WHERE url = ?", (url,))
            return cur.fetchone()["id"]
        return cur.lastrowid


def source_list() -> List[Dict[str, Any]]:
    with _tx() as cur:
        cur.execute("SELECT * FROM sources ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]


def source_delete(source_id: int) -> None:
    with _tx() as cur:
        cur.execute("DELETE FROM sources WHERE id = ?", (source_id,))


# ── Articles ──────────────────────────────────────────────

def article_upsert(
    video_id: str,
    *,
    title: str = "",
    channel: str = "",
    description: str = "",
    thumbnail: Optional[str] = None,
    duration: int = 0,
    url: str = "",
    status: str = "pending",
    source_id: Optional[int] = None,
) -> int:
    now = time.time()
    with _tx() as cur:
        cur.execute(
            """INSERT INTO articles
                (video_id, title, channel, description, thumbnail, duration, url, status, source_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(video_id) DO UPDATE SET
                 title=excluded.title, channel=excluded.channel, description=excluded.description,
                 thumbnail=excluded.thumbnail, duration=excluded.duration, url=excluded.url,
                 source_id=COALESCE(excluded.source_id, articles.source_id),
                 updated_at=excluded.updated_at
            """,
            (video_id, title, channel, description, thumbnail, duration, url, status, source_id, now, now),
        )
        if cur.lastrowid:
            return cur.lastrowid
        cur.execute("SELECT id FROM articles WHERE video_id = ?", (video_id,))
        return cur.fetchone()["id"]


_ARTICLE_COLUMNS = frozenset({
    "title", "channel", "description", "thumbnail", "duration", "url",
    "transcript", "article_md", "article_html", "status", "source_id", "error",
})


def article_update(video_id: str, **fields: Any) -> None:
    if not fields:
        return
    bad = set(fields) - _ARTICLE_COLUMNS
    if bad:
        raise ValueError(f"Invalid article fields: {bad}")
    fields["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [video_id]
    with _tx() as cur:
        cur.execute(f"UPDATE articles SET {set_clause} WHERE video_id = ?", vals)


def article_get(video_id: str) -> Optional[Dict[str, Any]]:
    with _tx() as cur:
        cur.execute("SELECT * FROM articles WHERE video_id = ?", (video_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def article_list(
    status: Optional[str] = None,
    search: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conditions: List[str] = []
    params: List[Any] = []
    if status:
        conditions.append("status = ?")
        params.append(status)
    if search:
        conditions.append("(title LIKE ? OR channel LIKE ?)")
        like = f"%{search}%"
        params.extend([like, like])
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    with _tx() as cur:
        cur.execute(f"SELECT * FROM articles{where} ORDER BY created_at DESC", params)
        return [dict(row) for row in cur.fetchall()]


def article_delete(video_id: str) -> None:
    with _tx() as cur:
        cur.execute("DELETE FROM articles WHERE video_id = ?", (video_id,))


# ── Jobs ──────────────────────────────────────────────────

def job_create(job_type: str, article_id: Optional[int] = None, source_id: Optional[int] = None) -> int:
    now = time.time()
    with _tx() as cur:
        cur.execute(
            "INSERT INTO jobs (job_type, article_id, source_id, status, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?)",
            (job_type, article_id, source_id, now, now),
        )
        return cur.lastrowid


_JOB_COLUMNS = frozenset({"status", "error"})


def job_update(job_id: int, **fields: Any) -> None:
    if not fields:
        return
    bad = set(fields) - _JOB_COLUMNS
    if bad:
        raise ValueError(f"Invalid job fields: {bad}")
    fields["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [job_id]
    with _tx() as cur:
        cur.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", vals)


def job_next_pending() -> Optional[Dict[str, Any]]:
    """Claim the oldest pending job (atomically set it to running)."""
    with _tx() as cur:
        cur.execute("SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return None
        job = dict(row)
        cur.execute("UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?", (time.time(), job["id"]))
        return job


def job_list_active() -> List[Dict[str, Any]]:
    with _tx() as cur:
        cur.execute(
            "SELECT j.*, a.title AS article_title, a.video_id "
            "FROM jobs j LEFT JOIN articles a ON j.article_id = a.id "
            "WHERE j.status IN ('pending', 'running') ORDER BY j.created_at ASC"
        )
        return [dict(row) for row in cur.fetchall()]


def job_recover_stuck(timeout_seconds: float = 600) -> int:
    """Reset jobs stuck in 'running' for longer than timeout back to 'pending'."""
    cutoff = time.time() - timeout_seconds
    with _tx() as cur:
        cur.execute(
            "UPDATE jobs SET status = 'pending', updated_at = ? "
            "WHERE status = 'running' AND updated_at < ?",
            (time.time(), cutoff),
        )
        return cur.rowcount


def job_cleanup(max_age_seconds: float = 604800) -> int:
    """Delete completed/errored jobs older than max_age (default 7 days)."""
    cutoff = time.time() - max_age_seconds
    with _tx() as cur:
        cur.execute(
            "DELETE FROM jobs WHERE status IN ('done', 'error') AND updated_at < ?",
            (cutoff,),
        )
        return cur.rowcount


def job_list_all() -> List[Dict[str, Any]]:
    with _tx() as cur:
        cur.execute(
            "SELECT j.*, a.title AS article_title, a.video_id "
            "FROM jobs j LEFT JOIN articles a ON j.article_id = a.id "
            "ORDER BY j.created_at DESC LIMIT 100"
        )
        return [dict(row) for row in cur.fetchall()]
