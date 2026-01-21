"""
Part 1: Fetch Videos from YouTube
Supports:
1. Fetch latest videos from configured channels
2. Get info for specific video URLs
3. Get all videos from a playlist

Filters out YouTube Shorts by checking video duration.
Uses yt-dlp instead of the YouTube API (no API key required).
"""

import sys
import re
import yt_dlp

# ========================================
# YOUR FAVORITE CHANNELS GO HERE
# Use the @ handle from the channel's YouTube page
# Example: youtube.com/@MrBeast → use "@MrBeast"
# ========================================
CHANNELS = [
    "@LatentSpacePod",
    "@ycombinator",
    "@a16z",
    "@RedpointAI",
    "@EveryInc",
    "@DataDrivenNYC",
    "@NoPriorsPodcast",
    "@DwarkeshPatel",
]

# Minimum duration in seconds to be considered a long-form video (not a Short)
# YouTube Shorts are typically under 60 seconds
MIN_DURATION_SECONDS = 60


def is_playlist_url(url):
    """Check if URL is a YouTube playlist."""
    return "playlist?list=" in url or "&list=" in url


def get_video_info(video_url):
    """
    Get info for a specific video URL.
    Returns video metadata including thumbnail URL and chapters, or None if it fails.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(video_url, download=False)

            if not result:
                return None

            video_id = result.get("id")

            # Get best thumbnail URL
            thumbnail = result.get("thumbnail")
            if not thumbnail:
                thumbnails = result.get("thumbnails", [])
                if thumbnails:
                    # Get highest quality thumbnail
                    thumbnail = thumbnails[-1].get("url")

            # Extract chapters if available
            chapters = []
            raw_chapters = result.get("chapters") or []
            for ch in raw_chapters:
                chapters.append({
                    "title": ch.get("title", ""),
                    "start_time": ch.get("start_time", 0),
                    "end_time": ch.get("end_time", 0),
                })

            return {
                "title": result.get("title", "Unknown Title"),
                "video_id": video_id,
                "description": result.get("description", ""),
                "channel": result.get("channel", result.get("uploader", "Unknown")),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": thumbnail,
                "duration": result.get("duration", 0),
                "chapters": chapters,
            }

    except Exception as e:
        print(f"  Error fetching video: {e}")
        return None


def get_videos_from_playlist(playlist_url, max_videos=None):
    """
    Get all videos from a YouTube playlist.

    Args:
        playlist_url: URL of the playlist
        max_videos: Optional limit on number of videos to fetch
    """
    print(f"Fetching playlist: {playlist_url}\n")
    print("=" * 60)

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,  # Get full info for each video
    }

    if max_videos:
        ydl_opts["playlistend"] = max_videos

    videos = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(playlist_url, download=False)

            if not result:
                print("Could not fetch playlist")
                return []

            playlist_title = result.get("title", "Unknown Playlist")
            print(f"Playlist: {playlist_title}")

            entries = result.get("entries", [])
            print(f"Found {len(entries)} videos\n")

            for entry in entries:
                if entry is None:
                    continue

                # Skip shorts
                duration = entry.get("duration", 0)
                if duration and duration < MIN_DURATION_SECONDS:
                    print(f"  Skipping Short: {entry.get('title', 'Unknown')[:40]}...")
                    continue

                video_id = entry.get("id")

                # Get thumbnail
                thumbnail = entry.get("thumbnail")
                if not thumbnail:
                    thumbnails = entry.get("thumbnails", [])
                    if thumbnails:
                        thumbnail = thumbnails[-1].get("url")

                # Extract chapters if available
                chapters = []
                raw_chapters = entry.get("chapters") or []
                for ch in raw_chapters:
                    chapters.append({
                        "title": ch.get("title", ""),
                        "start_time": ch.get("start_time", 0),
                        "end_time": ch.get("end_time", 0),
                    })

                video = {
                    "title": entry.get("title", "Unknown Title"),
                    "video_id": video_id,
                    "description": entry.get("description", ""),
                    "channel": entry.get("channel", entry.get("uploader", "Unknown")),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "thumbnail": thumbnail,
                    "duration": duration,
                    "chapters": chapters,
                }
                videos.append(video)
                print(f"  Added: {video['title'][:50]}...")

    except Exception as e:
        print(f"Error fetching playlist: {e}")

    print("\n" + "=" * 60)
    print(f"Got {len(videos)} videos from playlist")

    return videos


def get_videos_from_urls(urls):
    """
    Get video info for a list of URLs.
    Automatically detects and handles playlist URLs.
    """
    videos = []

    for url in urls:
        if is_playlist_url(url):
            # Handle playlist
            playlist_videos = get_videos_from_playlist(url)
            videos.extend(playlist_videos)
        else:
            # Handle single video
            print(f"Fetching: {url}")
            video = get_video_info(url)

            if video:
                videos.append(video)
                print(f"  Channel: {video['channel']}")
                print(f"  Title: {video['title']}\n")
            else:
                print(f"  Failed to fetch video\n")

    print("=" * 60)
    print(f"Found {len(videos)} videos total!")

    return videos


def get_latest_video_from_channel(channel_handle):
    """
    Get the most recent LONG-FORM video from a channel.
    Skips YouTube Shorts by checking video duration.
    """
    # Build the channel URL
    channel_url = f"https://www.youtube.com/{channel_handle}/videos"

    # Configure yt-dlp to only extract info (no download)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,  # We need full info to get duration
        "playlistend": 15,  # Only check the 15 most recent videos
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract playlist/channel info
            result = ydl.extract_info(channel_url, download=False)

            if not result:
                return None

            channel_name = result.get("channel", result.get("uploader", channel_handle))
            entries = result.get("entries", [])

            for entry in entries:
                if entry is None:
                    continue

                # Check duration to filter out Shorts
                duration = entry.get("duration", 0)
                if duration and duration < MIN_DURATION_SECONDS:
                    continue  # Skip Shorts

                video_id = entry.get("id")

                # Get thumbnail
                thumbnail = entry.get("thumbnail")
                if not thumbnail:
                    thumbnails = entry.get("thumbnails", [])
                    if thumbnails:
                        thumbnail = thumbnails[-1].get("url")

                # Extract chapters if available
                chapters = []
                raw_chapters = entry.get("chapters") or []
                for ch in raw_chapters:
                    chapters.append({
                        "title": ch.get("title", ""),
                        "start_time": ch.get("start_time", 0),
                        "end_time": ch.get("end_time", 0),
                    })

                return {
                    "title": entry.get("title", "Unknown Title"),
                    "video_id": video_id,
                    "description": entry.get("description", ""),
                    "channel": channel_name,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "thumbnail": thumbnail,
                    "duration": duration,
                    "chapters": chapters,
                }

            return None

    except Exception as e:
        print(f"  Error fetching channel: {e}")
        return None


def get_videos_from_channels(channels=None):
    """
    Fetch the latest long-form video from each channel.
    """
    if channels is None:
        channels = CHANNELS

    print("Fetching latest LONG-FORM videos (skipping Shorts)...\n")
    print("=" * 60)

    videos = []

    for channel_handle in channels:
        print(f"Looking up: {channel_handle}")

        video = get_latest_video_from_channel(channel_handle)

        if video:
            videos.append(video)
            print(f"  Channel: {video['channel']}")
            print(f"  Found: {video['title']}")
            print(f"  URL: {video['url']}\n")
        else:
            print(f"  No long-form videos found\n")

    print("=" * 60)
    print(f"Found {len(videos)} videos total!")

    return videos


def main(video_urls=None):
    """
    Main function - supports multiple modes:
    1. If video_urls contains playlist URLs, fetch all videos from playlists
    2. If video_urls contains video URLs, fetch those specific videos
    3. Otherwise, fetch latest videos from configured channels
    """
    if video_urls:
        return get_videos_from_urls(video_urls)
    else:
        return get_videos_from_channels()


# This runs the main function when you execute the script
if __name__ == "__main__":
    # Check if URLs were passed as command-line arguments
    if len(sys.argv) > 1:
        urls = sys.argv[1:]
        main(video_urls=urls)
    else:
        main()
