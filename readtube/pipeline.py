"""Core pipeline: URL → transcript → article → output."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

from .cache import Cache
from .chapters import (
    estimate_reading_time,
    format_chaptered_transcript,
    format_chapters_for_prompt,
    split_transcript_by_chapters,
)
from .clean import clean_transcript
from .config import Config, progress, resolve_llm_config
from .errors import ReadtubeError
from .genre import detect_genre, get_genre_system_prompt
from .llm import get_backend
from .prompts import build_custom_prompt, get_prompt
from .render import render_output
from .sponsorblock import remove_sponsors
from .transcript import get_transcript, get_transcript_segments
from .video import VideoInfo, get_playlist_videos, get_video_info, is_playlist_url


def process_single(
    url: str,
    config: Config,
    output_path: Optional[str] = None,
    mode: Optional[str] = None,
    fmt: Optional[str] = None,
    timestamps: bool = False,
    use_chapters: bool = True,
    lang: Optional[str] = None,
    cli_backend: Optional[str] = None,
    cli_model: Optional[str] = None,
    verbose: bool = True,
    theme: str = "default",
    custom_prompt: Optional[str] = None,
    raw: bool = False,
    no_sponsorblock: bool = False,
    genre: Optional[str] = None,
) -> str:
    """Process a single video URL. Returns the rendered content."""

    cache = Cache(config.cache.dir, config.cache.ttl_days)
    mode = mode or config.output.default_mode
    timestamps = timestamps or config.output.timestamps
    use_chapters = use_chapters if use_chapters is not None else config.output.chapters

    # 1. Fetch video info
    progress("fetching video info...", verbose)
    video = get_video_info(url)
    progress(f"  {video['title']}", verbose)
    progress(f"  {video['channel']}", verbose)

    # Cache metadata
    cache.set_metadata(video["video_id"], dict(video))

    # 2. Transcript mode - no LLM needed
    if mode == "transcript":
        progress("downloading transcript...", verbose)
        cached = cache.get_transcript(video["video_id"], lang)
        if cached:
            progress("  (from cache)", verbose)
            transcript = cached
        else:
            transcript = get_transcript(video["video_id"], lang=lang, include_timestamps=timestamps)
            cache.set_transcript(video["video_id"], transcript, lang)

        word_count = len(transcript.split())
        progress(f"  {word_count} words", verbose)

        if output_path:
            return render_output(
                transcript, video["title"], video["channel"],
                video["url"], output_path, fmt or "md", theme,
            )
        return transcript

    # 3. Fetch transcript
    progress("downloading transcript...", verbose)
    cached = cache.get_transcript(video["video_id"], lang)
    if cached:
        progress("  (from cache)", verbose)
        transcript = cached
    else:
        transcript = get_transcript(video["video_id"], lang=lang, include_timestamps=timestamps)
        cache.set_transcript(video["video_id"], transcript, lang)

    word_count = len(transcript.split())
    progress(f"  {word_count} words", verbose)

    # 3b. SponsorBlock filtering
    if not no_sponsorblock:
        try:
            timed_segments = get_transcript_segments(video["video_id"], lang=lang)
            filtered = remove_sponsors(video["video_id"], transcript, timed_segments)
            if filtered != transcript:
                removed_words = len(transcript.split()) - len(filtered.split())
                progress(f"  sponsorblock: removed ~{removed_words} words", verbose)
                transcript = filtered
        except Exception:
            pass  # graceful fallback

    # 3c. Transcript cleaning
    if not raw:
        transcript = clean_transcript(transcript)
        cleaned_count = len(transcript.split())
        if cleaned_count < word_count:
            progress(f"  cleaned: {word_count} → {cleaned_count} words", verbose)

    # 4. Chapter splitting
    chapters = video.get("chapters", [])
    chapters_text = ""
    has_chapter_structure = False

    if use_chapters and chapters:
        chaptered = split_transcript_by_chapters(transcript, chapters)
        if chaptered.has_chapters:
            transcript = format_chaptered_transcript(chaptered)
            has_chapter_structure = True
            progress(f"  split into {len(chaptered.chapters)} chapters", verbose)
    elif chapters:
        chapters_text = format_chapters_for_prompt(chapters)

    # 4b. Genre detection
    detected_genre = genre or detect_genre(
        video["title"], video["channel"], video.get("description", ""),
    )
    progress(f"  genre: {detected_genre}", verbose)

    # 5. Build prompt
    if custom_prompt:
        system_prompt, user_prompt = build_custom_prompt(
            prompt=custom_prompt,
            transcript=transcript,
            title=video["title"],
            channel=video["channel"],
        )
    else:
        system_prompt, user_prompt = get_prompt(
            mode=mode,
            transcript=transcript,
            title=video["title"],
            channel=video["channel"],
            chapters_text=chapters_text,
            has_chapter_structure=has_chapter_structure,
            timestamps=timestamps,
        )
        # Override system prompt with genre-specific one
        try:
            system_prompt = get_genre_system_prompt(detected_genre)
        except KeyError:
            pass  # keep default system prompt

    # 6. Resolve LLM backend
    backend_name, model_name, api_key = resolve_llm_config(config, cli_backend, cli_model)
    progress(f"generating article ({backend_name}/{model_name or 'default'})...", verbose)

    backend = get_backend(backend_name, model=model_name, api_key=api_key, url=config.llm.url)

    # 7. Generate — stream if writing to stdout (TTY)
    is_stdout = output_path is None
    is_tty = sys.stdout.isatty()

    if is_stdout and is_tty:
        # Stream to terminal
        chunks = []
        for chunk in backend.stream(user_prompt, system_prompt):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            chunks.append(chunk)
        sys.stdout.write("\n")
        article_md = "".join(chunks)
    else:
        article_md = backend.generate(user_prompt, system_prompt)

    # 8. Post-process timestamps into links
    if timestamps:
        article_md = _linkify_timestamps(article_md, video["video_id"])

    reading_time = estimate_reading_time(len(article_md.split()))
    progress(f"done ({reading_time})", verbose)

    # 9. Render output
    if output_path:
        result = render_output(
            article_md, video["title"], video["channel"],
            video["url"], output_path, fmt, theme, video.get("thumbnail"),
        )
        size_kb = Path(output_path).stat().st_size / 1024
        progress(f"saved: {output_path} ({size_kb:.0f} KB)", verbose)
        return result

    # If we already streamed to stdout, don't print again
    if is_stdout and is_tty:
        return article_md

    return article_md


def process_playlist(
    url: str,
    config: Config,
    output_dir: str = ".",
    max_videos: Optional[int] = None,
    **kwargs,
) -> list[str]:
    """Process all videos in a playlist."""
    progress(f"fetching playlist...", kwargs.get("verbose", True))
    videos = get_playlist_videos(url, max_videos=max_videos)
    progress(f"  found {len(videos)} videos", kwargs.get("verbose", True))

    results = []
    for i, video in enumerate(videos, 1):
        progress(f"\n[{i}/{len(videos)}] {video['title'][:60]}...", kwargs.get("verbose", True))

        filename = _sanitize_filename(f"{video['channel']} - {video['title']}")
        fmt = kwargs.get("fmt") or config.output.default_format
        ext = fmt if fmt != "md" else "md"
        output_path = str(Path(output_dir) / f"{filename}.{ext}")

        try:
            result = process_single(
                video["url"], config, output_path=output_path, **kwargs,
            )
            results.append(result)
        except ReadtubeError as e:
            progress(f"  skipped: {e.message}", kwargs.get("verbose", True))

    return results


def process_batch(
    urls_file: str,
    config: Config,
    output_dir: str = ".",
    **kwargs,
) -> list[str]:
    """Process URLs from a text file (one per line)."""
    lines = Path(urls_file).read_text().strip().splitlines()
    urls = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

    progress(f"processing {len(urls)} URLs...", kwargs.get("verbose", True))

    results = []
    for i, url in enumerate(urls, 1):
        progress(f"\n[{i}/{len(urls)}] {url}", kwargs.get("verbose", True))

        try:
            video = get_video_info(url)
            filename = _sanitize_filename(f"{video['channel']} - {video['title']}")
            fmt = kwargs.get("fmt") or config.output.default_format
            ext = fmt if fmt != "md" else "md"
            output_path = str(Path(output_dir) / f"{filename}.{ext}")

            result = process_single(url, config, output_path=output_path, **kwargs)
            results.append(result)
        except ReadtubeError as e:
            progress(f"  skipped: {e.message}", kwargs.get("verbose", True))

    return results


def _linkify_timestamps(text: str, video_id: str) -> str:
    """Convert [MM:SS] markers into clickable YouTube links."""
    def replace_ts(match):
        ts = match.group(1)
        parts = ts.split(":")
        if len(parts) == 3:
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            seconds = int(parts[0]) * 60 + int(parts[1])
        else:
            return match.group(0)
        return f"[{ts}](https://youtube.com/watch?v={video_id}&t={seconds})"

    return re.sub(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]", replace_ts, text)


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Remove/replace problematic characters
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = name.strip(". ")
    # Truncate
    if len(name) > 200:
        name = name[:200]
    return name or "untitled"
