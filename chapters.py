"""
Chapter-aware transcript processing.
Splits transcripts by video chapters for better article structure.

Features:
- Split transcript by chapter timestamps
- Generate chapter summaries
- Create structured article outlines
- Handle videos without chapters gracefully
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class Chapter:
    """A video chapter with associated transcript content."""
    title: str
    start_time: float
    end_time: float
    transcript: str = ""
    word_count: int = 0


@dataclass
class ChapteredTranscript:
    """A transcript split into chapters."""
    chapters: List[Chapter] = field(default_factory=list)
    total_word_count: int = 0
    has_chapters: bool = False


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def parse_timestamp(timestamp: str) -> float:
    """Parse a timestamp string to seconds."""
    parts = timestamp.strip().split(':')
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(parts[0])
    except (ValueError, IndexError):
        return 0.0


def extract_timestamps_from_transcript(transcript: str) -> List[Tuple[float, str]]:
    """
    Extract timestamps from transcript text.

    Looks for patterns like:
    - [00:00] text
    - [0:00:00] text
    - (00:00) text

    Returns:
        List of (timestamp_seconds, text) tuples
    """
    pattern = r'\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(.+?)(?=\[?\d{1,2}:\d{2}|$)'
    matches = re.findall(pattern, transcript, re.DOTALL)

    results = []
    for timestamp_str, text in matches:
        seconds = parse_timestamp(timestamp_str)
        text = text.strip()
        if text:
            results.append((seconds, text))

    return results


def split_transcript_by_chapters(
    transcript: str,
    chapters: List[Dict[str, Any]],
    timestamped: bool = False,
) -> ChapteredTranscript:
    """
    Split a transcript into chapters based on video chapter markers.

    Args:
        transcript: The full transcript text
        chapters: List of chapter dicts with 'title', 'start_time', 'end_time'
        timestamped: If True, transcript has timestamps like [00:00]

    Returns:
        ChapteredTranscript with content split by chapters
    """
    result = ChapteredTranscript()

    if not chapters:
        # No chapters, create a single "chapter" with all content
        result.has_chapters = False
        result.chapters = [Chapter(
            title="Full Video",
            start_time=0,
            end_time=0,
            transcript=transcript,
            word_count=len(transcript.split()),
        )]
        result.total_word_count = result.chapters[0].word_count
        return result

    result.has_chapters = True

    # Sort chapters by start time
    sorted_chapters = sorted(chapters, key=lambda c: c.get('start_time', 0))

    if timestamped:
        # Transcript has timestamps, split by matching them to chapters
        segments = extract_timestamps_from_transcript(transcript)

        for i, ch_data in enumerate(sorted_chapters):
            start = ch_data.get('start_time', 0)
            end = ch_data.get('end_time', 0)

            # If no end time, use next chapter's start or a large number
            if not end and i < len(sorted_chapters) - 1:
                end = sorted_chapters[i + 1].get('start_time', float('inf'))
            elif not end:
                end = float('inf')

            # Collect segments for this chapter
            chapter_text = []
            for seg_time, seg_text in segments:
                if start <= seg_time < end:
                    chapter_text.append(seg_text)

            chapter = Chapter(
                title=ch_data.get('title', f'Chapter {i + 1}'),
                start_time=start,
                end_time=end if end != float('inf') else 0,
                transcript=' '.join(chapter_text),
            )
            chapter.word_count = len(chapter.transcript.split())
            result.chapters.append(chapter)
    else:
        # No timestamps in transcript, split roughly by word count proportion
        words = transcript.split()
        total_words = len(words)

        # Calculate total duration
        total_duration = max(c.get('end_time', 0) for c in sorted_chapters) if sorted_chapters else 1

        word_index = 0
        for i, ch_data in enumerate(sorted_chapters):
            start = ch_data.get('start_time', 0)
            end = ch_data.get('end_time', 0)

            if not end and i < len(sorted_chapters) - 1:
                end = sorted_chapters[i + 1].get('start_time', total_duration)
            elif not end:
                end = total_duration

            # Calculate proportional word count for this chapter
            duration = end - start
            proportion = duration / total_duration if total_duration > 0 else 1.0 / len(sorted_chapters)
            chapter_word_count = int(total_words * proportion)

            # Get words for this chapter
            end_index = min(word_index + chapter_word_count, total_words)
            chapter_text = ' '.join(words[word_index:end_index])

            chapter = Chapter(
                title=ch_data.get('title', f'Chapter {i + 1}'),
                start_time=start,
                end_time=end,
                transcript=chapter_text,
                word_count=end_index - word_index,
            )
            result.chapters.append(chapter)
            word_index = end_index

        # Handle any remaining words
        if word_index < total_words and result.chapters:
            result.chapters[-1].transcript += ' ' + ' '.join(words[word_index:])
            result.chapters[-1].word_count = len(result.chapters[-1].transcript.split())

    result.total_word_count = sum(c.word_count for c in result.chapters)
    return result


def generate_chapter_outline(chapters: List[Dict[str, Any]]) -> str:
    """
    Generate a formatted chapter outline for article structure.

    Args:
        chapters: List of chapter dicts with 'title', 'start_time'

    Returns:
        Formatted outline string
    """
    if not chapters:
        return ""

    lines = ["## Table of Contents\n"]
    for i, ch in enumerate(chapters, 1):
        timestamp = format_timestamp(ch.get('start_time', 0))
        title = ch.get('title', f'Chapter {i}')
        lines.append(f"{i}. **{title}** [{timestamp}]")

    return '\n'.join(lines)


def generate_chapter_article_template(chaptered: ChapteredTranscript) -> str:
    """
    Generate an article template based on chapter structure.

    Args:
        chaptered: ChapteredTranscript with split content

    Returns:
        Markdown template with chapter sections
    """
    if not chaptered.has_chapters:
        return "## Main Content\n\n[Article content here]"

    lines = []
    for i, chapter in enumerate(chaptered.chapters, 1):
        lines.append(f"## {chapter.title}")
        lines.append("")
        lines.append(f"[Summary of this section - approximately {chapter.word_count} words in source]")
        lines.append("")

    return '\n'.join(lines)


def format_chapters_for_prompt(chapters: List[Dict[str, Any]]) -> str:
    """
    Format chapters for inclusion in an LLM prompt.

    Args:
        chapters: List of chapter dicts

    Returns:
        Formatted string for the prompt
    """
    if not chapters:
        return "(No chapters available)"

    lines = []
    for ch in chapters:
        timestamp = format_timestamp(ch.get('start_time', 0))
        title = ch.get('title', 'Untitled')
        lines.append(f"- [{timestamp}] {title}")

    return '\n'.join(lines)


def estimate_reading_time(word_count: int, wpm: int = 200) -> str:
    """
    Estimate reading time based on word count.

    Args:
        word_count: Number of words
        wpm: Words per minute (default 200)

    Returns:
        Human-readable time estimate
    """
    minutes = word_count / wpm
    if minutes < 1:
        return "less than 1 min"
    elif minutes < 60:
        return f"{int(minutes)} min"
    else:
        hours = int(minutes / 60)
        remaining_mins = int(minutes % 60)
        if remaining_mins:
            return f"{hours}h {remaining_mins}m"
        return f"{hours}h"


def get_chapter_stats(chapters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about video chapters.

    Returns:
        Dict with chapter statistics
    """
    if not chapters:
        return {
            'count': 0,
            'total_duration': 0,
            'avg_duration': 0,
            'chapters': [],
        }

    stats = {
        'count': len(chapters),
        'chapters': [],
    }

    total_duration = 0
    for i, ch in enumerate(chapters):
        start = ch.get('start_time', 0)
        end = ch.get('end_time', 0)

        # Calculate end from next chapter if not available
        if not end and i < len(chapters) - 1:
            end = chapters[i + 1].get('start_time', start)

        duration = end - start
        total_duration += duration

        stats['chapters'].append({
            'title': ch.get('title', f'Chapter {i + 1}'),
            'start': format_timestamp(start),
            'duration': format_timestamp(duration) if duration > 0 else 'N/A',
        })

    stats['total_duration'] = format_timestamp(total_duration)
    stats['avg_duration'] = format_timestamp(total_duration / len(chapters)) if chapters else '0:00'

    return stats


if __name__ == '__main__':
    # Test with sample data
    sample_chapters = [
        {'title': 'Introduction', 'start_time': 0, 'end_time': 120},
        {'title': 'Main Topic', 'start_time': 120, 'end_time': 600},
        {'title': 'Deep Dive', 'start_time': 600, 'end_time': 1200},
        {'title': 'Conclusion', 'start_time': 1200, 'end_time': 1500},
    ]

    sample_transcript = """
    This is the introduction where we set up the topic.
    We'll be discussing various aspects of the subject matter.
    Now let's move into the main topic area.
    Here we explore the core concepts in detail.
    The deep dive section gets into technical specifics.
    We examine edge cases and advanced scenarios.
    Finally, in conclusion, we summarize our findings.
    Thanks for watching and see you next time.
    """ * 10  # Repeat for more content

    print("Chapter Stats:")
    print(get_chapter_stats(sample_chapters))

    print("\nChapter Outline:")
    print(generate_chapter_outline(sample_chapters))

    print("\nSplit Transcript:")
    chaptered = split_transcript_by_chapters(sample_transcript, sample_chapters)
    for ch in chaptered.chapters:
        print(f"  {ch.title}: {ch.word_count} words")

    print("\nReading Time:", estimate_reading_time(chaptered.total_word_count))
