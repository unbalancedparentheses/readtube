"""
Part 2: Extract Transcripts from YouTube Videos
Supports:
- Language selection (auto, manual, or preferred language)
- Speaker labels preservation when available
- Caching of fetched transcripts
- Timestamps in output
"""

import os
import time
import json
import hashlib
from typing import Optional, List, Dict, Any, TypedDict, Tuple
from youtube_transcript_api import YouTubeTranscriptApi


class LanguageInfo(TypedDict):
    """Transcript language information."""
    code: str
    name: str
    is_generated: bool
    is_translatable: bool


class TranscriptSegment(TypedDict):
    """A single transcript segment with timing."""
    text: str
    start: float
    duration: float
    timestamp: str

# Cache directory for transcripts
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".transcript_cache")


def _get_cache_settings() -> Tuple[bool, int]:
    """Return cache_enabled and cache_days from config, with safe defaults."""
    try:
        from config import get_config

        cfg = get_config()
        return cfg.fetch.cache_enabled, cfg.fetch.cache_days
    except Exception:
        return True, 7


def _list_transcripts(video_id: str):
    """Get transcript list with compatibility across youtube-transcript-api versions."""
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "list"):
            return api.list(video_id)
    except Exception:
        pass
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        try:
            return YouTubeTranscriptApi.list_transcripts(video_id)
        except Exception:
            pass
    # Final fallback to classmethod if present
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        return YouTubeTranscriptApi.list_transcripts(video_id)
    raise RuntimeError("YouTubeTranscriptApi does not support transcript listing on this version")


def _segment_text(segment) -> str:
    """Extract text from a transcript segment (dict or object)."""
    if isinstance(segment, dict):
        return str(segment.get("text", ""))
    return str(getattr(segment, "text", ""))


def _segment_start(segment) -> float:
    """Extract start time from a transcript segment (dict or object)."""
    if isinstance(segment, dict):
        return float(segment.get("start", 0.0))
    return float(getattr(segment, "start", 0.0))


def _segment_duration(segment) -> float:
    """Extract duration from a transcript segment (dict or object)."""
    if isinstance(segment, dict):
        return float(segment.get("duration", 0.0))
    return float(getattr(segment, "duration", 0.0))


def get_cache_path(video_id: str, lang: Optional[str] = None) -> str:
    """Get cache file path for a video transcript."""
    cache_key = f"{video_id}_{lang or 'auto'}"
    filename = hashlib.md5(cache_key.encode()).hexdigest() + ".json"
    return os.path.join(CACHE_DIR, filename)


def load_from_cache(video_id: str, lang: Optional[str] = None) -> Optional[str]:
    """Load transcript from cache if available."""
    cache_enabled, cache_days = _get_cache_settings()
    if not cache_enabled or cache_days <= 0:
        return None
    cache_path = get_cache_path(video_id, lang)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check cache age
                max_age = cache_days * 24 * 60 * 60
                if time.time() - data.get('timestamp', 0) < max_age:
                    return data.get('transcript')
        except:
            pass
    return None


def save_to_cache(video_id: str, transcript: str, lang: Optional[str] = None) -> None:
    """Save transcript to cache."""
    cache_enabled, _cache_days = _get_cache_settings()
    if not cache_enabled:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = get_cache_path(video_id, lang)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({
                'transcript': transcript,
                'timestamp': time.time(),
                'video_id': video_id,
                'lang': lang
            }, f)
    except:
        pass  # Cache failures are not critical


def list_available_languages(video_id: str) -> List[LanguageInfo]:
    """
    List all available transcript languages for a video.

    Returns:
        List of dicts with 'code' and 'name' for each available language
    """
    try:
        transcript_list = _list_transcripts(video_id)

        languages = []
        for transcript in transcript_list:
            languages.append({
                'code': transcript.language_code,
                'name': transcript.language,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable,
            })

        return languages

    except Exception as e:
        print(f"  Error listing languages: {e}")
        return []


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def get_transcript(
    video_id: str,
    lang: Optional[str] = None,
    preserve_speakers: bool = False,
    use_cache: bool = True,
    include_timestamps: bool = False
) -> Optional[str]:
    """
    Get the transcript for a YouTube video.

    Args:
        video_id: YouTube video ID
        lang: Preferred language code (e.g., 'en', 'es', 'de').
              If None, uses auto-generated or first available.
        preserve_speakers: If True, preserves speaker labels when available
        use_cache: If True, uses cached transcripts when available
        include_timestamps: If True, includes timestamps in the output

    Returns:
        The full text of everything said in the video, or None if unavailable
    """
    # Try cache first
    if use_cache:
        cached = load_from_cache(video_id, lang)
        if cached:
            print(f"  (from cache)")
            return cached

    try:
        # Get available transcripts
        transcript_list = _list_transcripts(video_id)

        transcript = None

        if lang:
            # Try to get specific language
            try:
                transcript = transcript_list.find_transcript([lang])
            except:
                # Try to get translated version
                try:
                    for t in transcript_list:
                        if t.is_translatable:
                            transcript = t.translate(lang)
                            break
                except:
                    pass

        if transcript is None:
            # Fall back to any available transcript
            # Prefer manual over auto-generated
            for t in transcript_list:
                if not t.is_generated:
                    transcript = t
                    break

            if transcript is None:
                # Use first available (likely auto-generated)
                transcript = next(iter(transcript_list))

        # Fetch the actual transcript data
        transcript_data = transcript.fetch()

        # Build the text
        if include_timestamps:
            # Include timestamps at regular intervals or per segment
            full_text = ""
            for segment in transcript_data:
                timestamp = format_timestamp(_segment_start(segment))
                text = _segment_text(segment).strip()
                full_text += f"[{timestamp}] {text}\n"
        elif preserve_speakers:
            # Try to preserve speaker information if available
            full_text = ""
            current_speaker = None

            for segment in transcript_data:
                text = _segment_text(segment)

                # Some transcripts have speaker info in the text
                if text.startswith('[') and ']' in text:
                    bracket_end = text.index(']')
                    speaker = text[1:bracket_end]
                    text = text[bracket_end + 1:].strip()

                    if speaker != current_speaker:
                        if current_speaker is not None:
                            full_text += "\n\n"
                        full_text += f"**{speaker}:** "
                        current_speaker = speaker

                full_text += text + " "
        else:
            # Simple concatenation
            full_text = " ".join(_segment_text(segment) for segment in transcript_data)

        result = full_text.strip()

        # Cache the result
        if use_cache:
            save_to_cache(video_id, result, lang)

        return result

    except Exception as e:
        print(f"  Error getting transcript: {e}")
        return None


def get_transcripts_for_videos(
    videos: List[Dict[str, Any]],
    lang: Optional[str] = None,
    preserve_speakers: bool = False
) -> List[Dict[str, Any]]:
    """
    Get transcripts for a list of videos.

    Args:
        videos: List of video dicts (from get_videos.py)
        lang: Preferred language code
        preserve_speakers: Whether to preserve speaker labels

    Returns:
        List of videos with transcripts added
    """
    print("\nExtracting transcripts...\n")
    print("=" * 60)

    for i, video in enumerate(videos):
        print(f"Getting transcript: {video['title'][:50]}...")

        transcript = get_transcript(
            video["video_id"],
            lang=lang,
            preserve_speakers=preserve_speakers
        )

        if transcript:
            video["transcript"] = transcript
            word_count = len(transcript.split())
            print(f"  Got {word_count} words\n")
        else:
            video["transcript"] = None
            print(f"  No transcript available\n")

        # Small delay between requests to avoid rate limiting
        if i < len(videos) - 1:
            time.sleep(2)

    # Filter out videos without transcripts
    videos_with_transcripts = [v for v in videos if v.get("transcript")]

    print("=" * 60)
    print(f"Got transcripts for {len(videos_with_transcripts)} of {len(videos)} videos")

    return videos_with_transcripts


def get_transcript_with_timestamps(video_id: str, lang: Optional[str] = None) -> Optional[List[TranscriptSegment]]:
    """
    Get transcript as a list of segments with timestamps.

    Args:
        video_id: YouTube video ID
        lang: Preferred language code

    Returns:
        List of dicts with 'text', 'start', 'duration' keys, or None if unavailable
    """
    try:
        transcript_list = _list_transcripts(video_id)

        transcript = None
        if lang:
            try:
                transcript = transcript_list.find_transcript([lang])
            except:
                for t in transcript_list:
                    if t.is_translatable:
                        transcript = t.translate(lang)
                        break

        if transcript is None:
            for t in transcript_list:
                if not t.is_generated:
                    transcript = t
                    break
            if transcript is None:
                transcript = next(iter(transcript_list))

        transcript_data = transcript.fetch()

        return [
            {
                'text': _segment_text(segment),
                'start': _segment_start(segment),
                'duration': _segment_duration(segment),
                'timestamp': format_timestamp(_segment_start(segment)),
            }
            for segment in transcript_data
        ]

    except Exception as e:
        print(f"  Error getting transcript: {e}")
        return None


def clear_cache() -> None:
    """Clear the transcript cache."""
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print("Cache cleared")


# Test it standalone
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_id = sys.argv[1]

        # List available languages
        print(f"Available languages for {video_id}:")
        languages = list_available_languages(video_id)
        for lang in languages:
            gen = " (auto-generated)" if lang['is_generated'] else ""
            print(f"  {lang['code']}: {lang['name']}{gen}")

        # Get transcript
        print(f"\nFetching transcript...")
        transcript = get_transcript(video_id)
        if transcript:
            print(f"Got transcript! First 500 chars:\n{transcript[:500]}...")
    else:
        # Test with sample video
        test_video_id = "dQw4w9WgXcQ"
        print("Testing transcript extraction...")
        transcript = get_transcript(test_video_id)
        if transcript:
            print(f"Got transcript! First 200 chars:\n{transcript[:200]}...")
