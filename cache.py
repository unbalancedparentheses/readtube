"""
SQLite-based caching for Readtube.
Provides efficient caching for transcripts, video metadata, and LLM responses.

Features:
- SQLite storage (no extra dependencies)
- Automatic cache expiration
- Cache statistics and management
- Thread-safe operations
- Optional compression for large content
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict, List, Union


@dataclass
class CacheConfig:
    """Cache configuration."""
    db_path: str = ".cache/readtube.db"
    default_ttl: int = 7 * 24 * 60 * 60  # 7 days in seconds
    compress_threshold: int = 10000  # Compress content larger than 10KB
    max_size_mb: int = 500  # Max cache size in MB
    vacuum_threshold: int = 100  # Vacuum after this many deletes


@dataclass
class CacheEntry:
    """A cached entry with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: float
    size: int
    compressed: bool
    cache_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SQLiteCache:
    """SQLite-based cache with TTL support."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._local = threading.local()
        self._delete_count = 0
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            db_path = Path(self.config.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._local.conn = sqlite3.connect(
                str(db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._transaction() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    size INTEGER NOT NULL,
                    compressed INTEGER NOT NULL DEFAULT 0,
                    cache_type TEXT NOT NULL DEFAULT 'generic',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires
                ON cache(expires_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_type
                ON cache(cache_type)
            """)

    def _make_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Create a normalized cache key."""
        if namespace:
            return f"{namespace}:{key}"
        return key

    def _serialize(self, value: Any) -> tuple[bytes, bool]:
        """Serialize and optionally compress a value."""
        data = json.dumps(value).encode('utf-8')
        compressed = False

        if len(data) > self.config.compress_threshold:
            compressed_data = zlib.compress(data, level=6)
            if len(compressed_data) < len(data) * 0.9:  # Only use if >10% smaller
                data = compressed_data
                compressed = True

        return data, compressed

    def _deserialize(self, data: bytes, compressed: bool) -> Any:
        """Deserialize and optionally decompress a value."""
        if compressed:
            data = zlib.decompress(data)
        return json.loads(data.decode('utf-8'))

    def get(
        self,
        key: str,
        namespace: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Cache key
            namespace: Optional namespace prefix
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        full_key = self._make_key(key, namespace)
        now = time.time()

        with self._transaction() as cursor:
            cursor.execute(
                "SELECT value, compressed, expires_at FROM cache WHERE key = ?",
                (full_key,)
            )
            row = cursor.fetchone()

            if row is None:
                return default

            if row['expires_at'] < now:
                # Expired, delete and return default
                cursor.execute("DELETE FROM cache WHERE key = ?", (full_key,))
                self._delete_count += 1
                return default

            return self._deserialize(row['value'], bool(row['compressed']))

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None,
        cache_type: str = "generic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time to live in seconds (default from config)
            namespace: Optional namespace prefix
            cache_type: Type of cached content (transcript, video, article, etc.)
            metadata: Optional metadata to store with the entry
        """
        full_key = self._make_key(key, namespace)
        ttl = ttl or self.config.default_ttl
        now = time.time()
        expires_at = now + ttl

        data, compressed = self._serialize(value)
        metadata_json = json.dumps(metadata or {})

        with self._transaction() as cursor:
            cursor.execute("""
                INSERT OR REPLACE INTO cache
                (key, value, created_at, expires_at, size, compressed, cache_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (full_key, data, now, expires_at, len(data), int(compressed), cache_type, metadata_json))

    def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a value from the cache.

        Returns:
            True if the key existed and was deleted
        """
        full_key = self._make_key(key, namespace)

        with self._transaction() as cursor:
            cursor.execute("DELETE FROM cache WHERE key = ?", (full_key,))
            deleted = cursor.rowcount > 0

        if deleted:
            self._delete_count += 1
            if self._delete_count >= self.config.vacuum_threshold:
                self.vacuum()

        return deleted

    def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """Check if a key exists and is not expired."""
        return self.get(key, namespace) is not None

    def clear(self, namespace: Optional[str] = None, cache_type: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            namespace: If provided, only clear entries with this prefix
            cache_type: If provided, only clear entries of this type

        Returns:
            Number of entries deleted
        """
        with self._transaction() as cursor:
            if namespace and cache_type:
                cursor.execute(
                    "DELETE FROM cache WHERE key LIKE ? AND cache_type = ?",
                    (f"{namespace}:%", cache_type)
                )
            elif namespace:
                cursor.execute(
                    "DELETE FROM cache WHERE key LIKE ?",
                    (f"{namespace}:%",)
                )
            elif cache_type:
                cursor.execute(
                    "DELETE FROM cache WHERE cache_type = ?",
                    (cache_type,)
                )
            else:
                cursor.execute("DELETE FROM cache")

            count = cursor.rowcount

        self.vacuum()
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.

        Returns:
            Number of entries removed
        """
        now = time.time()

        with self._transaction() as cursor:
            cursor.execute("DELETE FROM cache WHERE expires_at < ?", (now,))
            count = cursor.rowcount

        if count > 0:
            self.vacuum()

        return count

    def vacuum(self) -> None:
        """Optimize the database file size."""
        self._conn.execute("VACUUM")
        self._delete_count = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        now = time.time()

        with self._transaction() as cursor:
            # Total entries
            cursor.execute("SELECT COUNT(*) as count FROM cache")
            total = cursor.fetchone()['count']

            # Expired entries
            cursor.execute("SELECT COUNT(*) as count FROM cache WHERE expires_at < ?", (now,))
            expired = cursor.fetchone()['count']

            # Total size
            cursor.execute("SELECT COALESCE(SUM(size), 0) as total FROM cache")
            total_size = cursor.fetchone()['total']

            # By type
            cursor.execute("""
                SELECT cache_type, COUNT(*) as count, SUM(size) as size
                FROM cache WHERE expires_at >= ?
                GROUP BY cache_type
            """, (now,))
            by_type = {
                row['cache_type']: {'count': row['count'], 'size': row['size']}
                for row in cursor.fetchall()
            }

            # Database file size
            db_size = Path(self.config.db_path).stat().st_size if Path(self.config.db_path).exists() else 0

        return {
            'total_entries': total,
            'valid_entries': total - expired,
            'expired_entries': expired,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'db_size_bytes': db_size,
            'db_size_mb': round(db_size / (1024 * 1024), 2),
            'by_type': by_type,
        }

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Convenience functions for transcript caching
_cache_instance: Optional[SQLiteCache] = None


def get_cache() -> SQLiteCache:
    """Get the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        cache_dir = os.environ.get('CACHE_DIR', '.cache')
        _cache_instance = SQLiteCache(CacheConfig(db_path=f"{cache_dir}/readtube.db"))
    return _cache_instance


def cache_transcript(
    video_id: str,
    transcript: str,
    lang: Optional[str] = None,
    ttl: int = 7 * 24 * 60 * 60,
) -> None:
    """Cache a transcript."""
    cache = get_cache()
    key = f"{video_id}_{lang or 'auto'}"
    cache.set(
        key,
        transcript,
        ttl=ttl,
        namespace="transcript",
        cache_type="transcript",
        metadata={"video_id": video_id, "lang": lang},
    )


def get_cached_transcript(
    video_id: str,
    lang: Optional[str] = None,
) -> Optional[str]:
    """Get a cached transcript."""
    cache = get_cache()
    key = f"{video_id}_{lang or 'auto'}"
    return cache.get(key, namespace="transcript")


def cache_video_info(
    video_id: str,
    info: Dict[str, Any],
    ttl: int = 24 * 60 * 60,  # 1 day
) -> None:
    """Cache video metadata."""
    cache = get_cache()
    cache.set(
        video_id,
        info,
        ttl=ttl,
        namespace="video",
        cache_type="video",
    )


def get_cached_video_info(video_id: str) -> Optional[Dict[str, Any]]:
    """Get cached video metadata."""
    cache = get_cache()
    return cache.get(video_id, namespace="video")


def cache_article(
    video_id: str,
    article: str,
    backend: str = "unknown",
    ttl: int = 30 * 24 * 60 * 60,  # 30 days
) -> None:
    """Cache a generated article."""
    cache = get_cache()
    cache.set(
        video_id,
        article,
        ttl=ttl,
        namespace="article",
        cache_type="article",
        metadata={"backend": backend},
    )


def get_cached_article(video_id: str) -> Optional[str]:
    """Get a cached article."""
    cache = get_cache()
    return cache.get(video_id, namespace="article")


if __name__ == '__main__':
    # Demo and test
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CacheConfig(db_path=f"{tmpdir}/test.db")
        cache = SQLiteCache(config)

        # Test basic operations
        print("Testing SQLite cache...")

        # Set and get
        cache.set("test_key", {"hello": "world"}, namespace="test")
        result = cache.get("test_key", namespace="test")
        assert result == {"hello": "world"}, "Get failed"
        print("  Set/Get: OK")

        # TTL
        cache.set("expires_soon", "data", ttl=1, namespace="test")
        assert cache.get("expires_soon", namespace="test") == "data"
        time.sleep(1.1)
        assert cache.get("expires_soon", namespace="test") is None
        print("  TTL expiration: OK")

        # Large content (compression)
        large_content = "x" * 20000
        cache.set("large", large_content, namespace="test", cache_type="large")
        assert cache.get("large", namespace="test") == large_content
        print("  Compression: OK")

        # Stats
        stats = cache.stats()
        print(f"  Stats: {stats['total_entries']} entries, {stats['total_size_mb']} MB")

        # Cleanup
        cache.cleanup_expired()
        print("  Cleanup: OK")

        # Clear
        cache.clear(namespace="test")
        assert cache.get("test_key", namespace="test") is None
        print("  Clear: OK")

        cache.close()
        print("\nAll tests passed!")
