#!/usr/bin/env python3
"""
Batch processing for Readtube.
Process multiple videos from a YAML/JSON config file.

Usage:
    python batch.py config.yaml
    python batch.py config.json --output-dir ./ebooks
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from config import BatchConfig, BatchJob, get_config, logger
from get_videos import get_video_info, get_videos_from_playlist, is_playlist_url
from get_transcripts import get_transcript
from create_epub import create_ebook

# Progress indicator
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def progress_bar(iterable, desc: str = "", total: Optional[int] = None):
    """Wrap iterable with progress bar if tqdm available."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total)
    else:
        # Simple fallback
        items = list(iterable)
        total = len(items)
        for i, item in enumerate(items):
            print(f"\r{desc}: {i+1}/{total}", end="", flush=True)
            yield item
        print()


def estimate_reading_time(text: str, wpm: int = 200) -> int:
    """Estimate reading time in minutes."""
    words = len(text.split())
    return max(1, round(words / wpm))


def retry_with_backoff(func, max_attempts: int = 3, delay: float = 1.0):
    """Retry a function with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = delay * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)


def process_job(job: BatchJob, batch_config: BatchConfig) -> Optional[Dict[str, Any]]:
    """Process a single batch job."""
    url = job.url
    lang = job.language or batch_config.default_language

    logger.info(f"Processing: {url}")

    try:
        # Fetch video info with retry
        video = retry_with_backoff(lambda: get_video_info(url))
        if not video:
            logger.error(f"Failed to fetch video info: {url}")
            return None

        logger.info(f"  Title: {video['title']}")
        logger.info(f"  Channel: {video['channel']}")

        # Fetch transcript with retry
        transcript = retry_with_backoff(
            lambda: get_transcript(video['video_id'], lang=lang)
        )
        if not transcript:
            logger.error(f"  No transcript available")
            return None

        word_count = len(transcript.split())
        reading_time = estimate_reading_time(transcript)
        logger.info(f"  Transcript: {word_count} words (~{reading_time} min read)")

        return {
            'video': video,
            'transcript': transcript,
            'job': job,
            'word_count': word_count,
            'reading_time': reading_time,
        }

    except Exception as e:
        logger.error(f"  Error processing {url}: {e}")
        return None


def process_batch(config_path: Path, output_dir: Optional[str] = None) -> List[str]:
    """
    Process a batch config file.

    Returns:
        List of created file paths
    """
    batch_config = BatchConfig.load(config_path)

    if output_dir:
        batch_config.output_dir = output_dir

    # Create output directory
    out_path = Path(batch_config.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Expand playlist URLs into individual jobs
    expanded_jobs = []
    for job in batch_config.jobs:
        if is_playlist_url(job.url):
            logger.info(f"Expanding playlist: {job.url}")
            videos = get_videos_from_playlist(job.url)
            for video in videos:
                expanded_jobs.append(BatchJob(
                    url=video['url'],
                    output_format=job.output_format,
                    language=job.language,
                    summary_mode=job.summary_mode,
                ))
        else:
            expanded_jobs.append(job)

    logger.info(f"Processing {len(expanded_jobs)} videos...")

    # Process each job
    results = []
    for job in progress_bar(expanded_jobs, desc="Fetching"):
        result = process_job(job, batch_config)
        if result:
            results.append(result)
        time.sleep(0.5)  # Rate limiting

    if not results:
        logger.error("No videos processed successfully")
        return []

    logger.info(f"\nSuccessfully fetched {len(results)} videos")
    logger.info("Note: Article writing should be done by Claude Code")

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH RESULTS")
    print("=" * 60)

    total_words = 0
    total_time = 0

    for i, result in enumerate(results):
        video = result['video']
        print(f"\n{i+1}. {video['title'][:60]}")
        print(f"   Channel: {video['channel']}")
        print(f"   Words: {result['word_count']} (~{result['reading_time']} min read)")
        if video.get('chapters'):
            print(f"   Chapters: {len(video['chapters'])}")
        total_words += result['word_count']
        total_time += result['reading_time']

    print("\n" + "-" * 60)
    print(f"Total: {len(results)} videos, {total_words} words, ~{total_time} min reading time")
    print("=" * 60)

    # Return video data for Claude Code to process
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch process YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example config.yaml:
  output_dir: ./ebooks
  default_format: epub
  default_language: en

  jobs:
    - url: https://www.youtube.com/watch?v=VIDEO_ID
    - url: https://www.youtube.com/playlist?list=PLAYLIST_ID
      output_format: pdf
    - url: https://www.youtube.com/watch?v=OTHER_ID
      summary_mode: true
"""
    )

    parser.add_argument("config", type=Path, help="Path to batch config (YAML or JSON)")
    parser.add_argument("--output-dir", "-o", help="Override output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    results = process_batch(args.config, args.output_dir)

    if results:
        print(f"\nFetched {len(results)} videos. Use Claude Code to write articles.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
