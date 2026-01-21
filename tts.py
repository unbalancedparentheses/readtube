#!/usr/bin/env python3
"""
Text-to-Speech audio narration for Readtube.
Generates audiobook versions of articles.

Supports multiple TTS backends:
- pyttsx3 (offline, cross-platform)
- gTTS (Google TTS, requires internet)
- macOS 'say' command
- Edge TTS (Microsoft, high quality)

Usage:
    python tts.py article.md --output article.mp3
    python tts.py article.md --backend gtts --lang en
"""

import os
import re
import sys
import argparse
import tempfile
from pathlib import Path
from typing import Optional, List
from abc import ABC, abstractmethod

from config import logger


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""

    @abstractmethod
    def synthesize(self, text: str, output_path: str, **kwargs) -> bool:
        """Synthesize speech from text."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class Pyttsx3Backend(TTSBackend):
    """Offline TTS using pyttsx3."""

    def is_available(self) -> bool:
        try:
            import pyttsx3
            return True
        except ImportError:
            return False

    def synthesize(self, text: str, output_path: str, rate: int = 150, **kwargs) -> bool:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty('rate', rate)

        # Save to file
        engine.save_to_file(text, output_path)
        engine.runAndWait()

        return os.path.exists(output_path)


class GTTSBackend(TTSBackend):
    """Google Text-to-Speech backend."""

    def is_available(self) -> bool:
        try:
            from gtts import gTTS
            return True
        except ImportError:
            return False

    def synthesize(self, text: str, output_path: str, lang: str = 'en', slow: bool = False, **kwargs) -> bool:
        from gtts import gTTS

        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(output_path)

        return os.path.exists(output_path)


class MacOSSayBackend(TTSBackend):
    """macOS 'say' command backend."""

    def is_available(self) -> bool:
        import platform
        import shutil
        return platform.system() == 'Darwin' and shutil.which('say') is not None

    def synthesize(self, text: str, output_path: str, voice: str = 'Samantha', rate: int = 175, **kwargs) -> bool:
        import subprocess

        # say command outputs AIFF, we need to convert
        aiff_path = output_path.replace('.mp3', '.aiff')

        result = subprocess.run([
            'say', '-v', voice, '-r', str(rate), '-o', aiff_path, text
        ], capture_output=True)

        if result.returncode != 0:
            return False

        # Convert to MP3 if ffmpeg is available
        if output_path.endswith('.mp3'):
            import shutil
            if shutil.which('ffmpeg'):
                subprocess.run([
                    'ffmpeg', '-i', aiff_path, '-y', output_path
                ], capture_output=True)
                os.remove(aiff_path)
            else:
                # Just rename to output path
                os.rename(aiff_path, output_path.replace('.mp3', '.aiff'))
                output_path = output_path.replace('.mp3', '.aiff')

        return os.path.exists(output_path)


class EdgeTTSBackend(TTSBackend):
    """Microsoft Edge TTS backend (high quality)."""

    def is_available(self) -> bool:
        try:
            import edge_tts
            return True
        except ImportError:
            return False

    def synthesize(self, text: str, output_path: str, voice: str = 'en-US-AriaNeural', **kwargs) -> bool:
        import asyncio
        import edge_tts

        async def _synthesize():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        asyncio.run(_synthesize())
        return os.path.exists(output_path)


# Available backends
BACKENDS = {
    'pyttsx3': Pyttsx3Backend,
    'gtts': GTTSBackend,
    'macos': MacOSSayBackend,
    'edge': EdgeTTSBackend,
}


def get_available_backends() -> List[str]:
    """Get list of available TTS backends."""
    available = []
    for name, backend_class in BACKENDS.items():
        backend = backend_class()
        if backend.is_available():
            available.append(name)
    return available


def preprocess_text(text: str) -> str:
    """
    Preprocess text for better TTS output.
    - Remove markdown formatting
    - Expand abbreviations
    - Handle special characters
    """
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove markdown emphasis
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Remove markdown links, keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove code blocks
    text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)

    # Remove bullet points
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)

    # Expand common abbreviations
    abbreviations = {
        'e.g.': 'for example',
        'i.e.': 'that is',
        'etc.': 'et cetera',
        'vs.': 'versus',
        'Dr.': 'Doctor',
        'Mr.': 'Mister',
        'Mrs.': 'Misses',
        'Ms.': 'Miss',
        'Jr.': 'Junior',
        'Sr.': 'Senior',
    }
    for abbr, expansion in abbreviations.items():
        text = text.replace(abbr, expansion)

    # Add pauses after sentences
    text = re.sub(r'\.(\s)', r'. \1', text)

    # Clean up whitespace
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()


def text_to_speech(
    text: str,
    output_path: str,
    backend: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Convert text to speech audio file.

    Args:
        text: Text to convert
        output_path: Output audio file path
        backend: TTS backend to use (auto-select if None)
        **kwargs: Backend-specific options

    Returns:
        True if successful
    """
    # Preprocess text
    text = preprocess_text(text)

    # Select backend
    if backend is None:
        available = get_available_backends()
        if not available:
            logger.error("No TTS backend available. Install one of: pip install pyttsx3, pip install gtts, pip install edge-tts")
            return False
        backend = available[0]
        logger.info(f"Auto-selected TTS backend: {backend}")

    if backend not in BACKENDS:
        logger.error(f"Unknown backend: {backend}. Available: {', '.join(BACKENDS.keys())}")
        return False

    backend_instance = BACKENDS[backend]()
    if not backend_instance.is_available():
        logger.error(f"Backend {backend} is not available")
        return False

    logger.info(f"Generating audio with {backend}...")
    success = backend_instance.synthesize(text, output_path, **kwargs)

    if success:
        file_size = os.path.getsize(output_path)
        logger.info(f"Audio saved to {output_path} ({file_size / 1024 / 1024:.1f} MB)")
    else:
        logger.error("Failed to generate audio")

    return success


def main():
    parser = argparse.ArgumentParser(description="Text-to-Speech for Readtube")
    parser.add_argument("input", help="Input text or markdown file")
    parser.add_argument("--output", "-o", help="Output audio file (default: input.mp3)")
    parser.add_argument("--backend", "-b", choices=list(BACKENDS.keys()), help="TTS backend")
    parser.add_argument("--lang", default="en", help="Language code (for gtts)")
    parser.add_argument("--voice", help="Voice name (backend-specific)")
    parser.add_argument("--rate", type=int, default=175, help="Speech rate")
    parser.add_argument("--list-backends", action="store_true", help="List available backends")

    args = parser.parse_args()

    if args.list_backends:
        available = get_available_backends()
        print("Available TTS backends:")
        for name in available:
            print(f"  - {name}")
        if not available:
            print("  (none - install pyttsx3, gtts, or edge-tts)")
        return

    # Read input
    if os.path.exists(args.input):
        with open(args.input, 'r') as f:
            text = f.read()
    else:
        text = args.input

    # Output path
    output_path = args.output
    if not output_path:
        if os.path.exists(args.input):
            output_path = Path(args.input).stem + '.mp3'
        else:
            output_path = 'output.mp3'

    # Generate audio
    kwargs = {'lang': args.lang, 'rate': args.rate}
    if args.voice:
        kwargs['voice'] = args.voice

    success = text_to_speech(text, output_path, backend=args.backend, **kwargs)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
