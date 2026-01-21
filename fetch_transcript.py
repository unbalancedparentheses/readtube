#!/usr/bin/env python3
"""
Fetch video info and transcript from YouTube.
This script is designed to be used with Claude Code as a skill.

Usage:
    python fetch_transcript.py URL [URL2 ...]
    python fetch_transcript.py --playlist PLAYLIST_URL
    python fetch_transcript.py --channels
    python fetch_transcript.py URL --lang es
"""

from __future__ import annotations

import sys
import argparse
import json
import platform
from importlib import metadata
from typing import Optional, List, Dict, Any
from get_videos import get_video_info, get_videos_from_channels, get_videos_from_playlist, is_playlist_url
from get_transcripts import get_transcript, list_available_languages
from errors import (
    ReadtubeError,
    RateLimitError,
    format_error_for_user,
    retry_with_backoff,
    YOUTUBE_RETRY_CONFIG,
)


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"
    except Exception:
        return "unknown"


def _print_health(show_only_version: bool = False) -> None:
    """Print basic environment and dependency info."""
    print(f"Python: {platform.python_version()}")
    print("Readtube: local checkout")
    deps = [
        "yt-dlp",
        "youtube-transcript-api",
        "ebooklib",
        "markdown",
        "flask",
        "weasyprint",
    ]
    if show_only_version:
        return
    print("Dependencies:")
    for dep in deps:
        print(f"  {dep}: {_pkg_version(dep)}")


def fetch_single_video(url: str, lang: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch info and transcript for a single video with retry logic."""
    print(f"Fetching video: {url}")

    def _fetch():
        video = get_video_info(url)
        if not video:
            raise ValueError("Could not fetch video info")
        return video

    try:
        video = retry_with_backoff(
            _fetch,
            config=YOUTUBE_RETRY_CONFIG,
            on_retry=lambda e, attempt, delay: print(f"  Retry {attempt}: {e}. Waiting {delay:.1f}s...")
        )
    except ReadtubeError as e:
        print(format_error_for_user(e))
        return None
    except Exception as e:
        print(f"  Error: Could not fetch video info: {e}")
        return None

    print(f"  Title: {video['title']}")
    print(f"  Channel: {video['channel']}")

    def _get_transcript():
        return get_transcript(video['video_id'], lang=lang)

    try:
        transcript = retry_with_backoff(
            _get_transcript,
            config=YOUTUBE_RETRY_CONFIG,
            on_retry=lambda e, attempt, delay: print(f"  Retry {attempt}: {e}. Waiting {delay:.1f}s...")
        )
    except ReadtubeError as e:
        print(format_error_for_user(e))
        return None
    except Exception as e:
        transcript = None

    if not transcript:
        print(f"  Error: Could not fetch transcript")
        return None

    print(f"  Transcript: {len(transcript.split())} words")

    video['transcript'] = transcript
    return video


def fetch_from_playlist(playlist_url: str, lang: Optional[str] = None, max_videos: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch all videos from a playlist with retry logic."""
    try:
        videos = retry_with_backoff(
            lambda: get_videos_from_playlist(playlist_url, max_videos=max_videos),
            config=YOUTUBE_RETRY_CONFIG,
            on_retry=lambda e, attempt, delay: print(f"  Retry {attempt}: {e}. Waiting {delay:.1f}s...")
        )
    except ReadtubeError as e:
        print(format_error_for_user(e))
        return []
    except Exception as e:
        print(f"Error fetching playlist: {e}")
        return []

    results = []
    for video in videos:
        print(f"\nFetching transcript for: {video['title'][:50]}...")

        def _get_transcript(vid=video):
            return get_transcript(vid['video_id'], lang=lang)

        try:
            transcript = retry_with_backoff(
                _get_transcript,
                config=YOUTUBE_RETRY_CONFIG,
                on_retry=lambda e, attempt, delay: print(f"  Retry {attempt}: {e}. Waiting {delay:.1f}s...")
            )
        except ReadtubeError:
            transcript = None
        except Exception:
            transcript = None

        if transcript:
            video['transcript'] = transcript
            results.append(video)
            print(f"  Got {len(transcript.split())} words")
        else:
            print(f"  No transcript available")

    return results


def fetch_from_channels(lang: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch latest videos from configured channels with retry logic."""
    try:
        videos = retry_with_backoff(
            get_videos_from_channels,
            config=YOUTUBE_RETRY_CONFIG,
            on_retry=lambda e, attempt, delay: print(f"  Retry {attempt}: {e}. Waiting {delay:.1f}s...")
        )
    except ReadtubeError as e:
        print(format_error_for_user(e))
        return []
    except Exception as e:
        print(f"Error fetching channels: {e}")
        return []

    results = []
    for video in videos:
        print(f"\nFetching transcript for: {video['title'][:50]}...")

        def _get_transcript(vid=video):
            return get_transcript(vid['video_id'], lang=lang)

        try:
            transcript = retry_with_backoff(
                _get_transcript,
                config=YOUTUBE_RETRY_CONFIG,
                on_retry=lambda e, attempt, delay: print(f"  Retry {attempt}: {e}. Waiting {delay:.1f}s...")
            )
        except ReadtubeError:
            transcript = None
        except Exception:
            transcript = None

        if transcript:
            video['transcript'] = transcript
            results.append(video)
            print(f"  Got {len(transcript.split())} words")
        else:
            print(f"  No transcript available")

    return results


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def print_video_data(videos: List[Dict[str, Any]], summary_mode: bool = False, output_dir: Optional[str] = None) -> None:
    """Print video data for Claude Code to use."""
    print("\n" + "=" * 60)
    print("VIDEO DATA (for article writing):")
    print("=" * 60)

    if summary_mode:
        print("\n[MODE: SUMMARY - Write a short 2-3 paragraph summary for each video]")
    else:
        print("\n[MODE: FULL ARTICLE - Write a complete magazine-style article]")

    if output_dir:
        print(f"[OUTPUT DIRECTORY: {output_dir}]")

    for i, video in enumerate(videos):
        print(f"\n--- VIDEO {i+1} ---")
        print(f"TITLE: {video['title']}")
        print(f"CHANNEL: {video['channel']}")
        print(f"URL: {video['url']}")
        if video.get('thumbnail'):
            print(f"THUMBNAIL: {video['thumbnail']}")
        print(f"DESCRIPTION:\n{video.get('description', 'No description')[:500]}")

        # Print chapters if available
        chapters = video.get('chapters', [])
        if chapters:
            print(f"\nCHAPTERS ({len(chapters)} chapters):")
            for ch in chapters:
                timestamp = format_timestamp(ch['start_time'])
                print(f"  [{timestamp}] {ch['title']}")

        print(f"\nTRANSCRIPT:\n{video['transcript']}")
        print("\n" + "-" * 40)


def write_video_json(videos: List[Dict[str, Any]], output_path: str) -> None:
    """Write video data to a JSON file for downstream processing."""
    payload = videos if len(videos) != 1 else videos[0]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON: {output_path}")


def main() -> Optional[List[Dict[str, Any]]]:
    parser = argparse.ArgumentParser(
        description="Fetch YouTube video info and transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_transcript.py "https://youtube.com/watch?v=abc123"
  python fetch_transcript.py "https://youtube.com/playlist?list=PLxxx" --max 5
  python fetch_transcript.py --channels
  python fetch_transcript.py "https://youtube.com/watch?v=abc123" --lang es
  python fetch_transcript.py VIDEO_ID --list-languages
"""
    )

    parser.add_argument("urls", nargs="*", help="Video or playlist URLs")
    parser.add_argument("--channels", action="store_true", help="Fetch from configured channels")
    parser.add_argument("--lang", help="Preferred transcript language (e.g., 'en', 'es', 'de')")
    parser.add_argument("--max", type=int, help="Max videos to fetch from playlist")
    parser.add_argument("--list-languages", action="store_true", help="List available languages for a video")
    parser.add_argument("--output-dir", "-o", help="Output directory for generated ebooks")
    parser.add_argument("--summary", action="store_true", help="Request short summary instead of full article")
    parser.add_argument("--output-json", help="Write fetched video data to a JSON file")
    parser.add_argument("--version", action="store_true", help="Show version info and exit")
    parser.add_argument("--health", action="store_true", help="Check optional dependencies and exit")

    args = parser.parse_args()

    if args.version or args.health:
        _print_health(show_only_version=args.version and not args.health)
        return

    # List languages mode
    if args.list_languages and args.urls:
        video_id = args.urls[0]
        # Extract video ID if full URL
        if "youtube.com" in video_id or "youtu.be" in video_id:
            video = get_video_info(video_id)
            if video:
                video_id = video['video_id']

        print(f"Available languages for {video_id}:")
        languages = list_available_languages(video_id)
        for lang in languages:
            gen = " (auto-generated)" if lang['is_generated'] else ""
            print(f"  {lang['code']}: {lang['name']}{gen}")
        return

    # Determine mode
    if args.channels:
        videos = fetch_from_channels(lang=args.lang)
    elif args.urls:
        videos = []
        for url in args.urls:
            if is_playlist_url(url):
                playlist_videos = fetch_from_playlist(url, lang=args.lang, max_videos=args.max)
                videos.extend(playlist_videos)
            else:
                video = fetch_single_video(url, lang=args.lang)
                if video:
                    videos.append(video)
    else:
        parser.print_help()
        sys.exit(1)

    if not videos:
        print("\nNo videos with transcripts found.")
        sys.exit(1)

    print_video_data(videos, summary_mode=args.summary, output_dir=args.output_dir)
    if args.output_json:
        write_video_json(videos, args.output_json)
    return videos


if __name__ == "__main__":
    main()
