"""
Shared test fixtures for readtube tests.
"""

import os
import sys
import pytest
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_video():
    """Sample video data for testing."""
    return {
        "title": "Test Video Title",
        "video_id": "dQw4w9WgXcQ",
        "description": "This is a test video description.",
        "channel": "Test Channel",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        "duration": 213,
    }


@pytest.fixture
def sample_video_with_transcript(sample_video):
    """Sample video with transcript attached."""
    video = sample_video.copy()
    video["transcript"] = """
    Hello everyone, welcome to this video. Today we're going to talk about
    something really interesting. This is a test transcript that contains
    multiple sentences and should be enough for testing purposes.

    We'll cover several topics including testing, software development, and
    best practices for writing clean code. Let's get started!
    """
    return video


@pytest.fixture
def sample_article():
    """Sample article for testing EPUB creation."""
    return {
        "title": "Test Video Title",
        "channel": "Test Channel",
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        "article": """# The Art of Testing

This is a sample article created for testing purposes.

## Why Testing Matters

Testing is crucial for software quality. It helps us catch bugs early
and ensures our code works as expected.

## Key Points

- Write tests early
- Aim for good coverage
- Test edge cases

> "Code without tests is broken by design." — Jacob Kaplan-Moss

## Conclusion

Remember to always test your code before deploying.
"""
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)


@pytest.fixture
def mock_yt_dlp_video(mocker):
    """Mock yt-dlp for video info extraction."""
    mock_result = {
        "id": "dQw4w9WgXcQ",
        "title": "Rick Astley - Never Gonna Give You Up",
        "description": "The official video for Rick Astley",
        "channel": "Rick Astley",
        "uploader": "Rick Astley",
        "thumbnail": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        "duration": 213,
        "thumbnails": [
            {"url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg"},
            {"url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"},
        ]
    }

    mock_ydl = mocker.MagicMock()
    mock_ydl.__enter__ = mocker.MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = mocker.MagicMock(return_value=False)
    mock_ydl.extract_info = mocker.MagicMock(return_value=mock_result)

    mocker.patch('yt_dlp.YoutubeDL', return_value=mock_ydl)

    return mock_result


@pytest.fixture
def mock_transcript(mocker):
    """Mock youtube-transcript-api for transcript extraction."""
    mock_transcript_data = [
        mocker.MagicMock(text="Hello everyone"),
        mocker.MagicMock(text="welcome to this video"),
        mocker.MagicMock(text="today we will discuss testing"),
    ]

    mock_transcript_obj = mocker.MagicMock()
    mock_transcript_obj.fetch = mocker.MagicMock(return_value=mock_transcript_data)
    mock_transcript_obj.is_generated = False

    mock_transcript_list = mocker.MagicMock()
    mock_transcript_list.__iter__ = mocker.MagicMock(return_value=iter([mock_transcript_obj]))
    mock_transcript_list.find_transcript = mocker.MagicMock(return_value=mock_transcript_obj)

    mock_api = mocker.MagicMock()
    mock_api.list = mocker.MagicMock(return_value=mock_transcript_list)

    # Patch at the location where it's used, not where it's defined
    mocker.patch('get_transcripts.YouTubeTranscriptApi', return_value=mock_api)

    return "Hello everyone welcome to this video today we will discuss testing"
