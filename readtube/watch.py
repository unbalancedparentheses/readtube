"""Watch: track new appearances of people you follow on trusted channels.

Design: channel-primary, search-secondary.

  - Steady state: for each trusted channel, fetch recent uploads via
    `yt-dlp --flat-playlist -j` (reverse-chronological, anonymous, no
    bot-flag risk). For each video not in `seen.txt`, route it to a
    creator if any creator alias appears in the title. Channels with
    `default_creator` route every video (own channels, dedicated shows).

  - Discover (--discover): run YouTube searches per creator to surface
    channels you haven't added yet. Manual review, then expand config.

The view-count/duration/blocklist heuristics from the search-based
prototype are gone: trusting the channel list is the filter.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import CACHE_DIR, CONFIG_DIR, _parse_toml, progress
from .errors import ReadtubeError

WATCH_CONFIG_FILE = CONFIG_DIR / "watch.toml"
WATCH_STATE_DIR = CACHE_DIR / "watch"
SEEN_FILE = WATCH_STATE_DIR / "seen.txt"


@dataclass
class Creator:
    name: str
    aliases: list[str]                  # case-insensitive substrings to match in title
    queries: list[str] = field(default_factory=list)  # only used by --discover


@dataclass
class Channel:
    handle: str                         # @SomeHandle or full URL
    default_creator: Optional[str] = None  # if set, every video routes here, skip title match


@dataclass
class WatchConfig:
    creators: list[Creator] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=list)
    recent_per_channel: int = 20
    min_duration: int = 600              # seconds; drops shorts and clips even on trusted channels
    mode: str = "takeaways"
    output_dir: str = ""

    def __post_init__(self) -> None:
        if not self.output_dir:
            self.output_dir = str(Path.home() / "readtube" / "watch")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "WatchConfig":
        config_path = path or WATCH_CONFIG_FILE
        if not config_path.exists():
            raise ReadtubeError(
                f"watch config not found: {config_path}",
                "create one with: readtube watch --init",
            )
        data = _parse_toml(config_path.read_text())

        defaults = data.get("defaults", {})
        cfg = cls(
            recent_per_channel=defaults.get("recent_per_channel", 20),
            min_duration=defaults.get("min_duration", 600),
            mode=defaults.get("mode", "takeaways"),
            output_dir=defaults.get("output_dir", ""),
        )
        cfg.creators = [
            Creator(
                name=c["name"],
                aliases=[a.lower() for a in c.get("aliases", [c["name"]])],
                queries=list(c.get("queries", [])),
            )
            for c in data.get("creators", [])
        ]
        cfg.channels = [
            Channel(
                handle=ch["handle"],
                default_creator=ch.get("default_creator"),
            )
            for ch in data.get("channels", [])
        ]
        if not cfg.creators:
            raise ReadtubeError(
                "watch config has no creators",
                f"add [[creators]] entries to {config_path}",
            )
        if not cfg.channels:
            raise ReadtubeError(
                "watch config has no channels",
                f"add [[channels]] entries to {config_path}",
            )

        # Validate default_creator references
        creator_names = {c.name for c in cfg.creators}
        for ch in cfg.channels:
            if ch.default_creator and ch.default_creator not in creator_names:
                raise ReadtubeError(
                    f"channel {ch.handle}: default_creator '{ch.default_creator}' is not a defined creator",
                    "check spelling against [[creators]] name fields",
                )
        return cfg


def load_seen(path: Path = SEEN_FILE) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_seen(video_ids: list[str], path: Path = SEEN_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for vid in video_ids:
            f.write(f"{vid}\n")


def _channel_url(handle: str) -> str:
    """Normalize a handle into a yt-dlp-friendly URL.

    `@Foo` → `https://www.youtube.com/@Foo/videos`
    Bare URL → used as-is
    """
    h = handle.strip()
    if h.startswith("http"):
        return h
    if h.startswith("@"):
        return f"https://www.youtube.com/{h}/videos"
    return f"https://www.youtube.com/@{h}/videos"


def fetch_channel_recent(handle: str, limit: int, verbose: bool = True) -> list[dict]:
    """Return recent uploads for a channel, newest first."""
    url = _channel_url(handle)
    cmd = [
        "yt-dlp", "-j", "--flat-playlist", "--no-warnings",
        "--playlist-end", str(limit),
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except FileNotFoundError:
        raise ReadtubeError("yt-dlp not found", "install with: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        progress(f"  timeout: {handle}", verbose)
        return []

    if proc.returncode != 0 and not proc.stdout:
        msg = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        progress(f"  failed: {handle}: {msg[:200]}", verbose)
        return []

    entries = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def search_query(query: str, max_results: int, verbose: bool = True) -> list[dict]:
    """yt-dlp YouTube search, used by --discover only."""
    cmd = [
        "yt-dlp", "-j", "--flat-playlist", "--no-warnings",
        f"ytsearch{max_results}:{query}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except FileNotFoundError:
        raise ReadtubeError("yt-dlp not found", "install with: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        progress(f"  timeout: {query}", verbose)
        return []

    if proc.returncode != 0 and not proc.stdout:
        progress(f"  search failed: {query}", verbose)
        return []

    entries = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def match_creator(title: str, creators: list[Creator]) -> Optional[Creator]:
    """First creator whose any alias is a case-insensitive substring of title."""
    if not title:
        return None
    lower = title.lower()
    for creator in creators:
        for alias in creator.aliases:
            if alias in lower:
                return creator
    return None


def discover(
    cfg: WatchConfig,
    seen: set[str],
    verbose: bool = True,
) -> list[tuple[Creator, dict, Channel]]:
    """Walk all trusted channels, return (creator, entry, channel) for new matches."""
    found: dict[str, tuple[Creator, dict, Channel]] = {}
    creator_by_name = {c.name: c for c in cfg.creators}

    for channel in cfg.channels:
        progress(f"\n[{channel.handle}]", verbose)
        entries = fetch_channel_recent(channel.handle, cfg.recent_per_channel, verbose)
        progress(f"  got {len(entries)} recent uploads", verbose)

        titles_this_channel: set[str] = set()  # dedupe re-uploads with identical titles
        for entry in entries:
            vid = entry.get("id")
            if not vid or vid in seen or vid in found:
                continue
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            if title.lower() in titles_this_channel:
                continue

            dur = entry.get("duration")
            if dur is None or dur < cfg.min_duration:
                continue

            if channel.default_creator:
                creator = creator_by_name.get(channel.default_creator)
                if creator is None:
                    continue
            else:
                creator = match_creator(title, cfg.creators)
                if creator is None:
                    continue

            titles_this_channel.add(title.lower())
            found[vid] = (creator, entry, channel)

    return list(found.values())


def discover_search(
    cfg: WatchConfig,
    verbose: bool = True,
    max_results: int = 25,
) -> list[tuple[Creator, dict]]:
    """Run YouTube searches per creator. Returns (creator, entry) for hits
    whose channel is NOT already in cfg.channels. For manual curation.
    """
    trusted = {_normalize_handle(ch.handle) for ch in cfg.channels}
    found: dict[str, tuple[Creator, dict]] = {}

    for creator in cfg.creators:
        if not creator.queries:
            continue
        progress(f"\n[{creator.name}]", verbose)
        for query in creator.queries:
            progress(f"  search: {query!r}", verbose)
            entries = search_query(query, max_results, verbose)
            for entry in entries:
                vid = entry.get("id")
                if not vid or vid in found:
                    continue
                ch_handle = (entry.get("uploader_id") or entry.get("channel_id") or "")
                if _normalize_handle(ch_handle) in trusted:
                    continue
                title = entry.get("title") or ""
                if not match_creator(title, [creator]):
                    continue
                found[vid] = (creator, entry)
    return list(found.values())


def _normalize_handle(handle: str) -> str:
    """Lowercase, strip @ and URL parts, for comparison."""
    h = handle.strip().lower()
    h = re.sub(r"^https?://(www\.)?youtube\.com/", "", h)
    h = h.split("/", 1)[0]
    return h.lstrip("@")


def format_summary(creator: Creator, entry: dict, channel: Optional[Channel] = None) -> str:
    title = entry.get("title") or "?"
    dur = entry.get("duration")
    dur_str = f"{int(dur)//60:>3}m" if dur else "  ?m"
    src = channel.handle if channel else (entry.get("channel") or entry.get("uploader") or "?")
    return f"  [{creator.name}] {dur_str} | {src} — {title}"


def run_watch(
    config_path: Optional[str] = None,
    dry_run: bool = False,
    print_urls: bool = False,
    do_discover: bool = False,
    verbose: bool = True,
    cli_backend: Optional[str] = None,
    cli_model: Optional[str] = None,
) -> int:
    from datetime import date
    from .config import Config
    from .errors import ReadtubeError as RTE
    from .pipeline import process_single

    cfg_path = Path(config_path).expanduser() if config_path else None
    cfg = WatchConfig.load(cfg_path)
    seen = load_seen()

    if do_discover:
        progress(f"discover: {len(cfg.creators)} creators, {len(cfg.channels)} trusted channels", verbose)
        hits = discover_search(cfg, verbose=verbose)
        if not hits:
            progress("\nno new channels surfaced by search", verbose)
            return 0
        progress(f"\n{len(hits)} hit(s) on UNTRUSTED channels (review and add good ones to watch.toml):", verbose)
        # Group by channel for readability
        by_channel: dict[str, list[tuple[Creator, dict]]] = {}
        for creator, entry in hits:
            ch = entry.get("channel") or entry.get("uploader") or "?"
            by_channel.setdefault(ch, []).append((creator, entry))
        for ch, items in sorted(by_channel.items(), key=lambda kv: -len(kv[1])):
            progress(f"\n  {ch}  ({len(items)} hits)", verbose)
            for creator, entry in items[:3]:
                progress(format_summary(creator, entry), verbose)
        return len(hits)

    progress(f"watch: {len(cfg.creators)} creators, {len(cfg.channels)} channels, {len(seen)} seen", verbose)
    matches = discover(cfg, seen, verbose=verbose)

    if not matches:
        progress("\nno new videos matched", verbose)
        return 0

    progress(f"\nmatched {len(matches)} new video(s):", verbose)
    for creator, entry, channel in matches:
        progress(format_summary(creator, entry, channel), verbose)

    if print_urls:
        for _, entry, _ in matches:
            url = entry.get("webpage_url") or entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}"
            print(url)
        return len(matches)

    if dry_run:
        progress("\n(dry-run, nothing processed, seen.txt unchanged)", verbose)
        return len(matches)

    config = Config.load()
    out_root = Path(cfg.output_dir).expanduser() / date.today().isoformat()
    out_root.mkdir(parents=True, exist_ok=True)
    progress(f"\nwriting to {out_root}", verbose)

    processed_ids = []
    for i, (creator, entry, _channel) in enumerate(matches, 1):
        vid = entry["id"]
        url = entry.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}"
        title = entry.get("title") or vid
        safe_title = _sanitize(f"{creator.name} - {title}")
        fmt = config.output.default_format
        ext = fmt if fmt != "md" else "md"
        out_path = out_root / f"{safe_title}.{ext}"

        progress(f"\n[{i}/{len(matches)}] {creator.name}: {title[:60]}", verbose)
        try:
            process_single(
                url=url,
                config=config,
                output_path=str(out_path),
                mode=cfg.mode,
                cli_backend=cli_backend,
                cli_model=cli_model,
                verbose=verbose,
            )
            processed_ids.append(vid)
        except RTE as e:
            progress(f"  skipped: {e.message}", verbose)
            processed_ids.append(vid)  # avoid retrying broken videos forever

    append_seen(processed_ids)
    progress(f"\nprocessed {len(processed_ids)} video(s)", verbose)
    return len(processed_ids)


def init_watch_config() -> Path:
    WATCH_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if WATCH_CONFIG_FILE.exists():
        print(f"watch config already exists: {WATCH_CONFIG_FILE}", file=sys.stderr)
        return WATCH_CONFIG_FILE
    WATCH_CONFIG_FILE.write_text(DEFAULT_WATCH_TOML)
    print(f"created: {WATCH_CONFIG_FILE}", file=sys.stderr)
    return WATCH_CONFIG_FILE


def _sanitize(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = name.strip(". ")
    if len(name) > 200:
        name = name[:200]
    return name or "untitled"


DEFAULT_WATCH_TOML = """\
# Readtube watch — channel-primary discovery
#
# Steady state: `readtube watch` walks trusted channels, matches video
# titles against creator aliases, writes articles for new hits.
#
# Periodically: `readtube watch --discover` runs YouTube searches per
# creator and shows you new channels worth adding here.

[defaults]
recent_per_channel = 20         # how many recent uploads to scan per channel
min_duration = 600              # seconds; drops shorts and clip-cuts from own channels
mode = "takeaways"              # article | tldr | takeaways
# output_dir = "~/readtube/watch"

# ──────────────────────────────────────────────────────────────────
# Creators — alias = case-insensitive substring matched in titles.
# Add typo variants only if you're sure they're not spammer signals.
# ──────────────────────────────────────────────────────────────────

[[creators]]
name = "Lyn Alden"
aliases = ["lyn alden"]
queries = ["lyn alden interview"]

[[creators]]
name = "Arthur Hayes"
aliases = ["arthur hayes"]
queries = ["arthur hayes interview"]

[[creators]]
name = "Druckenmiller"
aliases = ["druckenmiller"]
queries = ["druckenmiller interview", "stanley druckenmiller"]

[[creators]]
name = "Ray Dalio"
aliases = ["ray dalio", "dalio"]
queries = ["ray dalio interview"]

[[creators]]
name = "Luke Gromen"
aliases = ["luke gromen", "gromen"]
queries = ["luke gromen interview"]

[[creators]]
name = "Raoul Pal"
aliases = ["raoul pal"]
queries = ["raoul pal interview"]

[[creators]]
name = "Paul Tudor Jones"
aliases = ["paul tudor jones", "tudor jones"]
queries = ["paul tudor jones interview"]

[[creators]]
name = "Howard Marks"
aliases = ["howard marks"]
queries = ["howard marks interview"]

[[creators]]
name = "Russell Napier"
aliases = ["russell napier"]
queries = ["russell napier interview"]

[[creators]]
name = "Mohamed El-Erian"
aliases = ["el-erian", "el erian", "mohamed el"]
queries = ["mohamed el-erian interview"]

[[creators]]
name = "Balaji Srinivasan"
aliases = ["balaji"]
queries = ["balaji srinivasan interview"]

[[creators]]
name = "Jim Rogers"
aliases = ["jim rogers"]
queries = ["jim rogers interview"]

# ──────────────────────────────────────────────────────────────────
# Channels — recent uploads scanned each run.
# `default_creator` skips title-match and routes every video there
# (use for own channels or single-person shows).
# Edit handles if any fail to resolve.
# ──────────────────────────────────────────────────────────────────

# Own / dedicated channels — every video routes to default_creator
[[channels]]
handle = "@LynAldenMedia"
default_creator = "Lyn Alden"

[[channels]]
handle = "@principlesbyraydalio"
default_creator = "Ray Dalio"

# Network State Podcast (Balaji)
[[channels]]
handle = "https://www.youtube.com/channel/UCKrpnfpTwncQ050VFXcVkuQ/videos"
default_creator = "Balaji Srinivasan"

# Macro podcasts
# Macro Voices
[[channels]]
handle = "https://www.youtube.com/channel/UCICRehoZjq3ZtAWgRJX118A/videos"

# Forward Guidance (Blockworks)
[[channels]]
handle = "https://www.youtube.com/channel/UCkrwgzhIBKccuDsi_SvZtnQ/videos"

[[channels]]
handle = "@Wealthion"

# Palisades Gold Radio
[[channels]]
handle = "https://www.youtube.com/channel/UC6X0ttmzTAJt_2ebcqcIbYw/videos"

# Real Vision
[[channels]]
handle = "https://www.youtube.com/channel/UCGXWKlq1Oxr3ddEtmKhAkPg/videos"

# Hidden Forces (Demetri Kofinas)
[[channels]]
handle = "https://www.youtube.com/channel/UC8URhgYos5fjHqFSO4RSIEg/videos"

# Monetary Matters (Jack Farley)
[[channels]]
handle = "https://www.youtube.com/channel/UCeyqw1Ns_cnhSJh5XvXPWgw/videos"

# Crypto-macro
# What Bitcoin Did
[[channels]]
handle = "https://www.youtube.com/channel/UCtvg5cXLY_tHDJeBoRySBtg/videos"

# The Peter McCormack Show
[[channels]]
handle = "https://www.youtube.com/channel/UCzrWKkFIRS0kjZf7x24GdGg/videos"

[[channels]]
handle = "@TheBitcoinLayer"

[[channels]]
handle = "@Anthonypompliano"

[[channels]]
handle = "@nataliebrunell"

[[channels]]
handle = "@CoinBureau"

[[channels]]
handle = "@BitcoinMagazine"

[[channels]]
handle = "@Cointelegraph"

# Long-form interview shows
[[channels]]
handle = "@lexfridman"

[[channels]]
handle = "@TheDiaryOfACEO"

[[channels]]
handle = "@allin"

# Events / institutional
# Sohn Conference Foundation
[[channels]]
handle = "https://www.youtube.com/channel/UCq4ajL72ndl4yPxyzSMLeMg/videos"

# Norges Bank — In Good Company (Nicolai Tangen)
[[channels]]
handle = "https://www.youtube.com/channel/UCRhQsN8AVIfZuBNeRV1A37w/videos"

# How Leaders Lead with David Novak
[[channels]]
handle = "https://www.youtube.com/channel/UCa4HLorpafz21UwJem_OnGg/videos"
"""
