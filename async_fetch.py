#!/usr/bin/env python3
"""
Async operations for Readtube.
Provides async versions of video and transcript fetching for better performance.

Usage:
    import asyncio
    from async_fetch import fetch_videos_async

    videos = asyncio.run(fetch_videos_async(urls))
"""

import asyncio
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from config import logger


async def fetch_video_async(url: str, executor: Optional[ThreadPoolExecutor] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch video info asynchronously.

    Args:
        url: YouTube video URL
        executor: Optional ThreadPoolExecutor for running sync code

    Returns:
        Video info dict or None
    """
    from get_videos import get_video_info

    loop = asyncio.get_event_loop()
    if executor:
        return await loop.run_in_executor(executor, get_video_info, url)
    return await loop.run_in_executor(None, get_video_info, url)


async def fetch_transcript_async(
    video_id: str,
    lang: Optional[str] = None,
    executor: Optional[ThreadPoolExecutor] = None
) -> Optional[str]:
    """
    Fetch transcript asynchronously.

    Args:
        video_id: YouTube video ID
        lang: Preferred language code
        executor: Optional ThreadPoolExecutor

    Returns:
        Transcript text or None
    """
    from get_transcripts import get_transcript

    loop = asyncio.get_event_loop()

    def _fetch():
        return get_transcript(video_id, lang=lang)

    if executor:
        return await loop.run_in_executor(executor, _fetch)
    return await loop.run_in_executor(None, _fetch)


async def fetch_video_with_transcript_async(
    url: str,
    lang: Optional[str] = None,
    executor: Optional[ThreadPoolExecutor] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch video info and transcript together asynchronously.

    Args:
        url: YouTube video URL
        lang: Preferred language code
        executor: Optional ThreadPoolExecutor

    Returns:
        Video info dict with transcript, or None
    """
    video = await fetch_video_async(url, executor)
    if not video:
        return None

    transcript = await fetch_transcript_async(video['video_id'], lang, executor)
    if transcript:
        video['transcript'] = transcript
        video['word_count'] = len(transcript.split())

    return video


async def fetch_videos_async(
    urls: List[str],
    lang: Optional[str] = None,
    max_concurrent: int = 5,
    include_transcripts: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch multiple videos concurrently.

    Args:
        urls: List of YouTube video URLs
        lang: Preferred transcript language
        max_concurrent: Maximum concurrent fetches
        include_transcripts: Whether to fetch transcripts

    Returns:
        List of video info dicts
    """
    results: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_semaphore(url: str) -> Optional[Dict[str, Any]]:
        async with semaphore:
            logger.info(f"Fetching: {url}")
            if include_transcripts:
                return await fetch_video_with_transcript_async(url, lang)
            return await fetch_video_async(url)

    # Create tasks for all URLs
    tasks = [fetch_with_semaphore(url) for url in urls]

    # Execute concurrently
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for result in completed:
        if isinstance(result, Exception):
            logger.error(f"Error fetching video: {result}")
        elif result is not None:
            results.append(result)

    return results


async def process_batch_async(
    urls: List[str],
    lang: Optional[str] = None,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    Process a batch of URLs asynchronously with progress reporting.

    Args:
        urls: List of YouTube video URLs
        lang: Preferred transcript language
        max_concurrent: Maximum concurrent fetches

    Returns:
        List of video info dicts with transcripts
    """
    results: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    total = len(urls)

    async def fetch_with_progress(url: str) -> Optional[Dict[str, Any]]:
        nonlocal completed_count

        async with semaphore:
            result = await fetch_video_with_transcript_async(url, lang)
            completed_count += 1
            logger.info(f"Progress: {completed_count}/{total} ({completed_count * 100 // total}%)")
            return result

    tasks = [fetch_with_progress(url) for url in urls]
    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for result in completed:
        if isinstance(result, Exception):
            logger.error(f"Error: {result}")
        elif result is not None:
            results.append(result)

    return results


class AsyncFetcher:
    """Async fetcher with connection pooling."""

    def __init__(self, max_concurrent: int = 5, max_workers: int = 10):
        """
        Initialize async fetcher.

        Args:
            max_concurrent: Maximum concurrent fetches
            max_workers: Maximum thread pool workers
        """
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def fetch_video(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch single video info."""
        async with self.semaphore:
            return await fetch_video_async(url, self.executor)

    async def fetch_transcript(self, video_id: str, lang: Optional[str] = None) -> Optional[str]:
        """Fetch single transcript."""
        async with self.semaphore:
            return await fetch_transcript_async(video_id, lang, self.executor)

    async def fetch_videos(
        self,
        urls: List[str],
        lang: Optional[str] = None,
        include_transcripts: bool = True
    ) -> List[Dict[str, Any]]:
        """Fetch multiple videos."""
        if include_transcripts:
            tasks = [fetch_video_with_transcript_async(url, lang, self.executor) for url in urls]
        else:
            tasks = [fetch_video_async(url, self.executor) for url in urls]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    def close(self) -> None:
        """Close the thread pool."""
        self.executor.shutdown(wait=True)

    async def __aenter__(self) -> 'AsyncFetcher':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


async def main_async(urls: List[str], lang: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Main async entry point.

    Args:
        urls: List of YouTube video URLs
        lang: Preferred transcript language

    Returns:
        List of video info dicts with transcripts
    """
    async with AsyncFetcher(max_concurrent=5) as fetcher:
        return await fetcher.fetch_videos(urls, lang, include_transcripts=True)


def main():
    """CLI entry point."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Async video fetching")
    parser.add_argument("urls", nargs="+", help="YouTube video URLs")
    parser.add_argument("--lang", help="Preferred transcript language")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent fetches")

    args = parser.parse_args()

    videos = asyncio.run(fetch_videos_async(args.urls, args.lang, args.max_concurrent))

    print(f"\nFetched {len(videos)} videos:")
    for v in videos:
        word_count = v.get('word_count', len(v.get('transcript', '').split()))
        print(f"  - {v['title'][:50]} ({word_count} words)")


if __name__ == "__main__":
    main()
