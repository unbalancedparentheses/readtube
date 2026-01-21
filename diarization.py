"""
Speaker diarization for podcast transcripts.
Identifies and labels different speakers in multi-speaker content.

Features:
- Heuristic-based speaker detection from transcript patterns
- Support for YouTube's built-in speaker labels
- Chapter-based speaker hints
- Configurable speaker names
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class Speaker:
    """Represents a speaker in the transcript."""
    id: str
    name: str
    speaking_time: float = 0.0
    segment_count: int = 0


@dataclass
class SpeakerSegment:
    """A segment of transcript with speaker information."""
    speaker_id: str
    speaker_name: str
    text: str
    start_time: float
    end_time: float


@dataclass
class DiarizationResult:
    """Result of speaker diarization."""
    segments: List[SpeakerSegment] = field(default_factory=list)
    speakers: Dict[str, Speaker] = field(default_factory=dict)
    is_multi_speaker: bool = False
    confidence: float = 0.0


# Common patterns for speaker identification in transcripts
SPEAKER_PATTERNS = [
    # [Speaker Name]: text
    r'^\[([^\]]+)\]:\s*(.+)$',
    # Speaker Name: text (at start of line)
    r'^([A-Z][a-zA-Z\s]{1,30}):\s+(.+)$',
    # SPEAKER NAME: text
    r'^([A-Z][A-Z\s]{1,30}):\s+(.+)$',
    # >> Speaker: text (common in auto-generated)
    r'^>>\s*([^:]+):\s*(.+)$',
]

# Words that indicate a speaker change
TURN_INDICATORS = [
    'yeah', 'yes', 'no', 'right', 'exactly', 'absolutely',
    'so', 'well', 'okay', 'ok', "i think", "i believe",
    "that's", "you know", "you're", "i'm", "i've",
]

# Common podcast host/guest patterns
PODCAST_PATTERNS = [
    (r'\b(host|interviewer)\b', 'Host'),
    (r'\b(guest|interviewee)\b', 'Guest'),
    (r'\bspeaker\s*(\d+)\b', 'Speaker {0}'),
]


def detect_speaker_labels(transcript: str) -> List[Tuple[str, str]]:
    """
    Detect speaker labels in transcript text.

    Returns:
        List of (speaker_name, text) tuples
    """
    segments = []
    current_speaker = None
    current_text = []

    for line in transcript.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Try each speaker pattern
        matched = False
        for pattern in SPEAKER_PATTERNS:
            match = re.match(pattern, line)
            if match:
                # Save previous speaker's text
                if current_speaker and current_text:
                    segments.append((current_speaker, ' '.join(current_text)))

                current_speaker = match.group(1).strip()
                text = match.group(2).strip() if len(match.groups()) > 1 else ''
                current_text = [text] if text else []
                matched = True
                break

        if not matched and current_speaker:
            current_text.append(line)

    # Don't forget the last segment
    if current_speaker and current_text:
        segments.append((current_speaker, ' '.join(current_text)))

    return segments


def estimate_speaker_changes(transcript: str) -> List[int]:
    """
    Estimate where speaker changes might occur based on linguistic cues.

    Returns:
        List of character positions where speaker changes might occur
    """
    changes = []
    sentences = re.split(r'[.!?]\s+', transcript)

    position = 0
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower().strip()

        # Check for turn-taking indicators at the start of sentences
        for indicator in TURN_INDICATORS:
            if sentence_lower.startswith(indicator):
                changes.append(position)
                break

        position += len(sentence) + 2  # +2 for punctuation and space

    return changes


def identify_speakers_from_description(description: str) -> Dict[str, str]:
    """
    Try to identify speaker names from video description.

    Returns:
        Dict mapping speaker roles to names
    """
    speakers = {}

    # Common patterns in descriptions
    patterns = [
        # "with John Smith" or "featuring John Smith"
        r'(?:with|featuring|ft\.?|feat\.?)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)',
        # "Guest: John Smith"
        r'(?:guest|speaker|host):\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)',
        # "John Smith (host)"
        r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s*\((?:host|guest|speaker)\)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, description, re.IGNORECASE)
        for i, match in enumerate(matches):
            role = f"Speaker {i + 1}" if i > 0 else "Guest"
            speakers[role] = match

    return speakers


def diarize_transcript(
    transcript: str,
    description: Optional[str] = None,
    chapters: Optional[List[Dict[str, Any]]] = None,
    speaker_names: Optional[Dict[str, str]] = None,
) -> DiarizationResult:
    """
    Perform speaker diarization on a transcript.

    Args:
        transcript: The raw transcript text
        description: Optional video description for speaker hints
        chapters: Optional list of chapters for context
        speaker_names: Optional mapping of speaker IDs to names

    Returns:
        DiarizationResult with identified speakers and segments
    """
    result = DiarizationResult()

    # First, try to detect explicit speaker labels
    labeled_segments = detect_speaker_labels(transcript)

    if labeled_segments:
        # Transcript has speaker labels
        result.is_multi_speaker = len(set(s[0] for s in labeled_segments)) > 1

        for speaker_label, text in labeled_segments:
            speaker_id = speaker_label.lower().replace(' ', '_')

            # Apply custom name mapping if provided
            if speaker_names and speaker_id in speaker_names:
                speaker_name = speaker_names[speaker_id]
            else:
                speaker_name = speaker_label

            # Create or update speaker
            if speaker_id not in result.speakers:
                result.speakers[speaker_id] = Speaker(
                    id=speaker_id,
                    name=speaker_name,
                )

            result.speakers[speaker_id].segment_count += 1

            # Add segment (we don't have timing info from labels)
            result.segments.append(SpeakerSegment(
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                text=text,
                start_time=0.0,
                end_time=0.0,
            ))

        result.confidence = 0.9 if result.is_multi_speaker else 0.5

    else:
        # No explicit labels, try to detect from description
        if description:
            desc_speakers = identify_speakers_from_description(description)
            if desc_speakers:
                for role, name in desc_speakers.items():
                    speaker_id = role.lower().replace(' ', '_')
                    result.speakers[speaker_id] = Speaker(id=speaker_id, name=name)

        # Estimate speaker changes heuristically
        changes = estimate_speaker_changes(transcript)

        if len(changes) > 5:
            # Likely multi-speaker content
            result.is_multi_speaker = True
            result.confidence = 0.4

            # Create generic speakers if none found
            if not result.speakers:
                result.speakers['speaker_1'] = Speaker(id='speaker_1', name='Speaker 1')
                result.speakers['speaker_2'] = Speaker(id='speaker_2', name='Speaker 2')
        else:
            result.is_multi_speaker = False
            result.confidence = 0.6

    return result


def format_diarized_transcript(result: DiarizationResult, style: str = 'markdown') -> str:
    """
    Format a diarized transcript for reading.

    Args:
        result: DiarizationResult from diarize_transcript
        style: Output style ('markdown', 'plain', 'html')

    Returns:
        Formatted transcript string
    """
    if not result.segments:
        return ""

    lines = []
    current_speaker = None

    for segment in result.segments:
        if segment.speaker_id != current_speaker:
            current_speaker = segment.speaker_id

            if style == 'markdown':
                lines.append(f"\n**{segment.speaker_name}:**")
            elif style == 'html':
                lines.append(f'\n<p class="speaker"><strong>{segment.speaker_name}:</strong></p>')
            else:
                lines.append(f"\n{segment.speaker_name}:")

        if style == 'html':
            lines.append(f'<p>{segment.text}</p>')
        else:
            lines.append(segment.text)

    return '\n'.join(lines).strip()


def get_speaker_stats(result: DiarizationResult) -> Dict[str, Any]:
    """
    Get statistics about speakers in the transcript.

    Returns:
        Dict with speaker statistics
    """
    stats = {
        'speaker_count': len(result.speakers),
        'is_multi_speaker': result.is_multi_speaker,
        'confidence': result.confidence,
        'speakers': {},
    }

    total_segments = len(result.segments)

    for speaker_id, speaker in result.speakers.items():
        speaker_segments = [s for s in result.segments if s.speaker_id == speaker_id]
        word_count = sum(len(s.text.split()) for s in speaker_segments)

        stats['speakers'][speaker.name] = {
            'segment_count': len(speaker_segments),
            'segment_percentage': len(speaker_segments) / total_segments * 100 if total_segments else 0,
            'word_count': word_count,
        }

    return stats


if __name__ == '__main__':
    # Test with sample transcript
    sample = """
    [Host]: Welcome to the show! Today we're talking about AI.
    [Guest]: Thanks for having me. I'm excited to discuss this topic.
    [Host]: So let's start with the basics. What is machine learning?
    [Guest]: Machine learning is a subset of AI that allows systems to learn from data.
    [Host]: That's fascinating. And how does deep learning fit in?
    [Guest]: Deep learning uses neural networks with many layers to learn complex patterns.
    """

    result = diarize_transcript(sample)
    print("Speaker Stats:", get_speaker_stats(result))
    print("\nFormatted Transcript:")
    print(format_diarized_transcript(result))
