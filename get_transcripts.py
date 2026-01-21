"""
Part 2: Extract Transcripts from YouTube Videos
Supports:
- Language selection (auto, manual, or preferred language)
- Speaker labels preservation when available
- Caching of fetched transcripts
"""

import os
import time
import json
import hashlib
from youtube_transcript_api import YouTubeTranscriptApi

# Cache directory for transcripts
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".transcript_cache")


def get_cache_path(video_id, lang=None):
    """Get cache file path for a video transcript."""
    cache_key = f"{video_id}_{lang or 'auto'}"
    filename = hashlib.md5(cache_key.encode()).hexdigest() + ".json"
    return os.path.join(CACHE_DIR, filename)


def load_from_cache(video_id, lang=None):
    """Load transcript from cache if available."""
    cache_path = get_cache_path(video_id, lang)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check cache age (7 days max)
                if time.time() - data.get('timestamp', 0) < 7 * 24 * 60 * 60:
                    return data.get('transcript')
        except:
            pass
    return None


def save_to_cache(video_id, transcript, lang=None):
    """Save transcript to cache."""
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


def list_available_languages(video_id):
    """
    List all available transcript languages for a video.

    Returns:
        List of dicts with 'code' and 'name' for each available language
    """
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

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


def get_transcript(video_id, lang=None, preserve_speakers=False, use_cache=True):
    """
    Get the transcript for a YouTube video.

    Args:
        video_id: YouTube video ID
        lang: Preferred language code (e.g., 'en', 'es', 'de').
              If None, uses auto-generated or first available.
        preserve_speakers: If True, preserves speaker labels when available
        use_cache: If True, uses cached transcripts when available

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
        ytt_api = YouTubeTranscriptApi()

        # Get available transcripts
        transcript_list = ytt_api.list(video_id)

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
        if preserve_speakers:
            # Try to preserve speaker information if available
            full_text = ""
            current_speaker = None

            for segment in transcript_data:
                text = segment.text

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
            full_text = " ".join(segment.text for segment in transcript_data)

        result = full_text.strip()

        # Cache the result
        if use_cache:
            save_to_cache(video_id, result, lang)

        return result

    except Exception as e:
        print(f"  Error getting transcript: {e}")
        return None


def get_transcripts_for_videos(videos, lang=None, preserve_speakers=False):
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


def clear_cache():
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
