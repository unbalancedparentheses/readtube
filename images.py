#!/usr/bin/env python3
"""
Image extraction from YouTube videos.
Extract thumbnails and frames at specific timestamps.

Usage:
    python images.py "https://youtube.com/watch?v=VIDEO_ID" --output ./images
    python images.py "https://youtube.com/watch?v=VIDEO_ID" --timestamps 0:30,1:00,2:30
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.request import urlretrieve

from config import logger


def get_video_thumbnails(video_url: str, output_dir: str = ".") -> List[str]:
    """
    Download all available thumbnail sizes for a video.

    Args:
        video_url: YouTube video URL
        output_dir: Directory to save thumbnails

    Returns:
        List of downloaded file paths
    """
    import yt_dlp

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded = []

    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)

        video_id = info.get('id', 'video')
        thumbnails = info.get('thumbnails', [])

        # Download each thumbnail
        for i, thumb in enumerate(thumbnails):
            url = thumb.get('url')
            if not url:
                continue

            # Determine extension
            ext = 'jpg'
            if '.webp' in url:
                ext = 'webp'
            elif '.png' in url:
                ext = 'png'

            # Name by resolution if available
            width = thumb.get('width', 0)
            height = thumb.get('height', 0)
            if width and height:
                filename = f"{video_id}_thumb_{width}x{height}.{ext}"
            else:
                filename = f"{video_id}_thumb_{i}.{ext}"

            filepath = output_path / filename

            try:
                urlretrieve(url, filepath)
                downloaded.append(str(filepath))
                logger.info(f"Downloaded thumbnail: {filename}")
            except Exception as e:
                logger.warning(f"Failed to download thumbnail: {e}")

    except Exception as e:
        logger.error(f"Error getting thumbnails: {e}")

    return downloaded


def get_best_thumbnail(video_url: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Download the highest quality thumbnail for a video.

    Args:
        video_url: YouTube video URL
        output_path: Output file path (optional)

    Returns:
        Path to downloaded thumbnail or None
    """
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)

        video_id = info.get('id', 'video')
        thumbnails = info.get('thumbnails', [])

        if not thumbnails:
            return None

        # Sort by resolution (width * height)
        sorted_thumbs = sorted(
            thumbnails,
            key=lambda t: (t.get('width', 0) or 0) * (t.get('height', 0) or 0),
            reverse=True
        )

        best = sorted_thumbs[0]
        url = best.get('url')

        if not url:
            return None

        # Determine output path
        if output_path is None:
            ext = 'jpg'
            if '.webp' in url:
                ext = 'webp'
            elif '.png' in url:
                ext = 'png'
            output_path = f"{video_id}_thumbnail.{ext}"

        urlretrieve(url, output_path)
        logger.info(f"Downloaded best thumbnail: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error getting thumbnail: {e}")
        return None


def extract_frames(
    video_url: str,
    timestamps: List[str],
    output_dir: str = ".",
    format: str = "jpg"
) -> List[str]:
    """
    Extract frames from a video at specific timestamps.

    Requires ffmpeg to be installed.

    Args:
        video_url: YouTube video URL
        timestamps: List of timestamps (e.g., ["0:30", "1:00", "2:30"])
        output_dir: Directory to save frames
        format: Output format (jpg, png, webp)

    Returns:
        List of extracted frame paths
    """
    import yt_dlp
    import shutil

    # Check ffmpeg is available
    if not shutil.which('ffmpeg'):
        logger.error("ffmpeg is required for frame extraction. Install it first.")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extracted = []

    try:
        # Get video info
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)

        video_id = info.get('id', 'video')

        # Get best video URL for streaming
        formats = info.get('formats', [])
        video_formats = [f for f in formats if f.get('vcodec') != 'none']

        if not video_formats:
            logger.error("No video formats available")
            return []

        # Prefer mp4 format
        mp4_formats = [f for f in video_formats if f.get('ext') == 'mp4']
        best_format = mp4_formats[0] if mp4_formats else video_formats[0]
        stream_url = best_format.get('url')

        if not stream_url:
            logger.error("Could not get video stream URL")
            return []

        # Extract frames at each timestamp
        for ts in timestamps:
            output_file = output_path / f"{video_id}_frame_{ts.replace(':', '-')}.{format}"

            cmd = [
                'ffmpeg',
                '-ss', ts,
                '-i', stream_url,
                '-frames:v', '1',
                '-q:v', '2',  # Quality (lower is better)
                '-y',  # Overwrite
                str(output_file)
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)

            if result.returncode == 0 and output_file.exists():
                extracted.append(str(output_file))
                logger.info(f"Extracted frame at {ts}: {output_file.name}")
            else:
                logger.warning(f"Failed to extract frame at {ts}")

    except subprocess.TimeoutExpired:
        logger.error("Frame extraction timed out")
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")

    return extracted


def extract_frames_interval(
    video_url: str,
    interval: int = 60,
    output_dir: str = ".",
    format: str = "jpg",
    max_frames: int = 10
) -> List[str]:
    """
    Extract frames at regular intervals throughout a video.

    Args:
        video_url: YouTube video URL
        interval: Seconds between frames
        output_dir: Directory to save frames
        format: Output format
        max_frames: Maximum number of frames to extract

    Returns:
        List of extracted frame paths
    """
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)

        duration = info.get('duration', 0)

        if duration <= 0:
            logger.error("Could not determine video duration")
            return []

        # Generate timestamps
        timestamps = []
        current = 0
        while current < duration and len(timestamps) < max_frames:
            minutes = int(current // 60)
            seconds = int(current % 60)
            timestamps.append(f"{minutes}:{seconds:02d}")
            current += interval

        return extract_frames(video_url, timestamps, output_dir, format)

    except Exception as e:
        logger.error(f"Error: {e}")
        return []


def get_chapter_thumbnails(
    video_url: str,
    output_dir: str = ".",
    format: str = "jpg"
) -> List[Dict[str, Any]]:
    """
    Extract a frame at the start of each chapter.

    Args:
        video_url: YouTube video URL
        output_dir: Directory to save frames
        format: Output format

    Returns:
        List of dicts with chapter info and frame paths
    """
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)

        chapters = info.get('chapters', [])

        if not chapters:
            logger.info("No chapters found in video")
            return []

        # Get timestamps from chapters
        timestamps = []
        for chapter in chapters:
            start = chapter.get('start_time', 0)
            minutes = int(start // 60)
            seconds = int(start % 60)
            timestamps.append(f"{minutes}:{seconds:02d}")

        # Extract frames
        frames = extract_frames(video_url, timestamps, output_dir, format)

        # Combine with chapter info
        results = []
        for i, (chapter, frame) in enumerate(zip(chapters, frames)):
            results.append({
                'chapter': chapter.get('title', f'Chapter {i+1}'),
                'start_time': chapter.get('start_time', 0),
                'frame_path': frame
            })

        return results

    except Exception as e:
        logger.error(f"Error extracting chapter thumbnails: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Extract images from YouTube videos")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    parser.add_argument("--thumbnails", action="store_true", help="Download all thumbnails")
    parser.add_argument("--best-thumb", action="store_true", help="Download best quality thumbnail")
    parser.add_argument("--timestamps", help="Extract frames at timestamps (comma-separated, e.g., 0:30,1:00)")
    parser.add_argument("--interval", type=int, help="Extract frames at interval (seconds)")
    parser.add_argument("--chapters", action="store_true", help="Extract frame at each chapter")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png", "webp"], help="Output format")
    parser.add_argument("--max-frames", type=int, default=10, help="Max frames for interval mode")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.thumbnails:
        files = get_video_thumbnails(args.url, str(output_dir))
        print(f"Downloaded {len(files)} thumbnails")

    elif args.best_thumb:
        filepath = get_best_thumbnail(args.url, str(output_dir / "thumbnail.jpg"))
        if filepath:
            print(f"Downloaded: {filepath}")

    elif args.timestamps:
        timestamps = [t.strip() for t in args.timestamps.split(",")]
        files = extract_frames(args.url, timestamps, str(output_dir), args.format)
        print(f"Extracted {len(files)} frames")

    elif args.interval:
        files = extract_frames_interval(
            args.url, args.interval, str(output_dir),
            args.format, args.max_frames
        )
        print(f"Extracted {len(files)} frames")

    elif args.chapters:
        results = get_chapter_thumbnails(args.url, str(output_dir), args.format)
        for r in results:
            print(f"  {r['chapter']}: {r['frame_path']}")

    else:
        # Default: download best thumbnail
        filepath = get_best_thumbnail(args.url, str(output_dir / "thumbnail.jpg"))
        if filepath:
            print(f"Downloaded: {filepath}")


if __name__ == "__main__":
    main()
