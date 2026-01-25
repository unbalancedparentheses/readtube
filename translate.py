#!/usr/bin/env python3
"""
Translation support for Readtube.
Translate transcripts between languages using various backends.

Supported backends:
- google: Google Translate (via googletrans)
- deepl: DeepL API (requires API key)
- libre: LibreTranslate (self-hosted or public API)

Usage:
    python translate.py input.txt --to es --output translated.txt
    python translate.py --text "Hello world" --to de
"""

import os
import sys
import argparse
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from config import logger


class TranslationBackend(ABC):
    """Abstract base class for translation backends."""

    @abstractmethod
    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Translate text to target language."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def supported_languages(self) -> List[str]:
        """List supported language codes."""
        pass


class GoogleTranslateBackend(TranslationBackend):
    """Google Translate backend using googletrans library."""

    def is_available(self) -> bool:
        try:
            from googletrans import Translator
            return True
        except ImportError:
            return False

    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        from googletrans import Translator

        translator = Translator()

        # Handle long text by splitting into chunks
        max_length = 5000
        if len(text) <= max_length:
            result = translator.translate(text, dest=target_lang, src=source_lang or 'auto')
            return result.text

        # Split by paragraphs and translate each
        paragraphs = text.split('\n\n')
        translated_parts = []

        for para in paragraphs:
            if len(para) <= max_length:
                result = translator.translate(para, dest=target_lang, src=source_lang or 'auto')
                translated_parts.append(result.text)
            else:
                # Split long paragraphs by sentences
                sentences = para.replace('. ', '.\n').split('\n')
                chunk = ""
                for sent in sentences:
                    if len(chunk) + len(sent) < max_length:
                        chunk += sent + " "
                    else:
                        if chunk:
                            result = translator.translate(chunk.strip(), dest=target_lang, src=source_lang or 'auto')
                            translated_parts.append(result.text)
                        chunk = sent + " "
                if chunk:
                    result = translator.translate(chunk.strip(), dest=target_lang, src=source_lang or 'auto')
                    translated_parts.append(result.text)

        return '\n\n'.join(translated_parts)

    def supported_languages(self) -> List[str]:
        return [
            'af', 'sq', 'am', 'ar', 'hy', 'az', 'eu', 'be', 'bn', 'bs', 'bg', 'ca',
            'ceb', 'zh-CN', 'zh-TW', 'co', 'hr', 'cs', 'da', 'nl', 'en', 'eo', 'et',
            'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'el', 'gu', 'ht', 'ha', 'haw', 'he',
            'hi', 'hmn', 'hu', 'is', 'ig', 'id', 'ga', 'it', 'ja', 'jv', 'kn', 'kk',
            'km', 'rw', 'ko', 'ku', 'ky', 'lo', 'la', 'lv', 'lt', 'lb', 'mk', 'mg',
            'ms', 'ml', 'mt', 'mi', 'mr', 'mn', 'my', 'ne', 'no', 'ny', 'or', 'ps',
            'fa', 'pl', 'pt', 'pa', 'ro', 'ru', 'sm', 'gd', 'sr', 'st', 'sn', 'sd',
            'si', 'sk', 'sl', 'so', 'es', 'su', 'sw', 'sv', 'tl', 'tg', 'ta', 'tt',
            'te', 'th', 'tr', 'tk', 'uk', 'ur', 'ug', 'uz', 'vi', 'cy', 'xh', 'yi',
            'yo', 'zu'
        ]


class DeepLBackend(TranslationBackend):
    """DeepL API backend (requires API key)."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('DEEPL_API_KEY')

    def is_available(self) -> bool:
        try:
            import deepl
            return self.api_key is not None
        except ImportError:
            return False

    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        import deepl

        translator = deepl.Translator(self.api_key)

        # DeepL uses different language codes
        target_lang = target_lang.upper()
        if target_lang == 'EN':
            target_lang = 'EN-US'

        result = translator.translate_text(
            text,
            target_lang=target_lang,
            source_lang=source_lang.upper() if source_lang else None
        )

        return result.text

    def supported_languages(self) -> List[str]:
        return [
            'BG', 'CS', 'DA', 'DE', 'EL', 'EN', 'ES', 'ET', 'FI', 'FR', 'HU',
            'ID', 'IT', 'JA', 'KO', 'LT', 'LV', 'NB', 'NL', 'PL', 'PT', 'RO',
            'RU', 'SK', 'SL', 'SV', 'TR', 'UK', 'ZH'
        ]


class LibreTranslateBackend(TranslationBackend):
    """LibreTranslate backend (self-hosted or public API)."""

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or os.environ.get('LIBRETRANSLATE_URL', 'https://libretranslate.com')
        self.api_key = api_key or os.environ.get('LIBRETRANSLATE_API_KEY')

    def is_available(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.api_url}/languages", timeout=5)
            return response.status_code == 200
        except:
            return False

    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        import requests

        data = {
            'q': text,
            'source': source_lang or 'auto',
            'target': target_lang,
            'format': 'text'
        }

        if self.api_key:
            data['api_key'] = self.api_key

        response = requests.post(f"{self.api_url}/translate", data=data)
        response.raise_for_status()

        return response.json()['translatedText']

    def supported_languages(self) -> List[str]:
        import requests
        try:
            response = requests.get(f"{self.api_url}/languages")
            languages = response.json()
            return [lang['code'] for lang in languages]
        except:
            return ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']


# Available backends
BACKENDS = {
    'google': GoogleTranslateBackend,
    'deepl': DeepLBackend,
    'libre': LibreTranslateBackend,
}


def get_available_backends() -> List[str]:
    """Get list of available translation backends."""
    available = []
    for name, backend_class in BACKENDS.items():
        backend = backend_class()
        if backend.is_available():
            available.append(name)
    return available


def translate_text(
    text: str,
    target_lang: str,
    source_lang: Optional[str] = None,
    backend: Optional[str] = None
) -> Optional[str]:
    """
    Translate text to target language.

    Args:
        text: Text to translate
        target_lang: Target language code (e.g., 'es', 'de', 'fr')
        source_lang: Source language code (auto-detect if None)
        backend: Translation backend to use (auto-select if None)

    Returns:
        Translated text or None if failed
    """
    # Select backend
    if backend is None:
        available = get_available_backends()
        if not available:
            logger.error("No translation backend available. Install googletrans: pip install googletrans==4.0.0-rc1")
            return None
        backend = available[0]
        logger.info(f"Auto-selected translation backend: {backend}")

    if backend not in BACKENDS:
        logger.error(f"Unknown backend: {backend}. Available: {', '.join(BACKENDS.keys())}")
        return None

    backend_instance = BACKENDS[backend]()
    if not backend_instance.is_available():
        logger.error(f"Backend {backend} is not available")
        return None

    try:
        logger.info(f"Translating to {target_lang} using {backend}...")
        result = backend_instance.translate(text, target_lang, source_lang)
        logger.info(f"Translation complete: {len(text)} -> {len(result)} chars")
        return result
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return None


def translate_transcript(
    transcript: str,
    target_lang: str,
    preserve_timestamps: bool = True,
    backend: Optional[str] = None
) -> Optional[str]:
    """
    Translate a transcript while preserving timestamps if present.

    Args:
        transcript: Transcript text (may include [timestamp] markers)
        target_lang: Target language code
        preserve_timestamps: Whether to preserve timestamp markers
        backend: Translation backend to use

    Returns:
        Translated transcript or None if failed
    """
    import re

    if not preserve_timestamps:
        return translate_text(transcript, target_lang, backend=backend)

    # Check if transcript has timestamps
    timestamp_pattern = r'\[(\d+:\d+(?::\d+)?)\]'
    has_timestamps = bool(re.search(timestamp_pattern, transcript))

    if not has_timestamps:
        return translate_text(transcript, target_lang, backend=backend)

    # Split by timestamp markers and translate each part
    parts = re.split(timestamp_pattern, transcript)
    translated_parts = []

    for i, part in enumerate(parts):
        if i % 2 == 1:  # This is a timestamp
            translated_parts.append(f"[{part}]")
        elif part.strip():  # This is text to translate
            translated = translate_text(part.strip(), target_lang, backend=backend)
            if translated:
                translated_parts.append(translated)
            else:
                translated_parts.append(part)  # Keep original on failure

    return ' '.join(translated_parts)


# Language name to code mapping
LANGUAGE_NAMES = {
    'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
    'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja',
    'korean': 'ko', 'chinese': 'zh-CN', 'arabic': 'ar', 'hindi': 'hi',
    'dutch': 'nl', 'polish': 'pl', 'turkish': 'tr', 'swedish': 'sv',
    'danish': 'da', 'norwegian': 'no', 'finnish': 'fi', 'greek': 'el',
    'hebrew': 'he', 'thai': 'th', 'vietnamese': 'vi', 'indonesian': 'id',
    'malay': 'ms', 'tagalog': 'tl', 'swahili': 'sw', 'czech': 'cs',
    'romanian': 'ro', 'hungarian': 'hu', 'ukrainian': 'uk', 'bulgarian': 'bg',
}


def normalize_language(lang: str) -> str:
    """Convert language name to code."""
    lang_lower = lang.lower()
    return LANGUAGE_NAMES.get(lang_lower, lang_lower)


def main():
    parser = argparse.ArgumentParser(description="Translate text or files")
    parser.add_argument("input", nargs="?", help="Input file path")
    parser.add_argument("--text", "-t", help="Text to translate (instead of file)")
    parser.add_argument("--to", "-l", required=True, help="Target language (code or name)")
    parser.add_argument("--from", dest="source", help="Source language (auto-detect if not specified)")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--backend", "-b", choices=list(BACKENDS.keys()), help="Translation backend")
    parser.add_argument("--list-backends", action="store_true", help="List available backends")
    parser.add_argument("--list-languages", action="store_true", help="List supported languages")

    args = parser.parse_args()

    if args.list_backends:
        print("Available translation backends:")
        for name in get_available_backends():
            print(f"  - {name}")
        if not get_available_backends():
            print("  (none - install googletrans: pip install googletrans==4.0.0-rc1)")
        return

    if args.list_languages:
        print("Supported languages:")
        for name, code in sorted(LANGUAGE_NAMES.items()):
            print(f"  {name}: {code}")
        return

    # Get text to translate
    if args.text:
        text = args.text
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Error: Provide either --text or input file")
        sys.exit(1)

    target_lang = normalize_language(args.to)
    source_lang = normalize_language(args.source) if args.source else None

    # Translate
    result = translate_text(text, target_lang, source_lang, args.backend)

    if result is None:
        sys.exit(1)

    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Translation saved to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
