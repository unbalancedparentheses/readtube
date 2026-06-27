"""Readtube CLI — YouTube videos in, readable articles out."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from .config import Config, init_config, print_config, progress
from .errors import ReadtubeError, EXIT_INVALID_ARGS, die

SUBCOMMANDS = {"playlist", "batch", "config", "cache", "watch"}

EPILOG = """\
examples:
  readtube https://youtube.com/watch?v=xxx          # article to stdout
  readtube URL -o article.md                         # save as markdown
  readtube URL -o article.epub                       # save as EPUB
  readtube URL --mode tldr                           # 3-5 bullet points
  readtube URL --mode transcript                     # raw transcript, no LLM
  readtube URL --timestamps                          # linked timestamps
  readtube playlist URL -o ./articles/               # process playlist
  readtube batch urls.txt -o ./out/                  # batch from file
  readtube config                                    # show config
  readtube config --init                             # create config file
  readtube cache stats                               # cache info
  readtube cache clear                               # wipe cache
  readtube watch --init                              # create watch config
  readtube watch --dry-run                           # discover, don't process
  readtube watch                                     # discover and write articles
"""


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by single/playlist/batch commands."""
    parser.add_argument("-o", "--output", help="output path")
    parser.add_argument("--format", choices=["md", "epub", "pdf", "html"], help="output format")
    parser.add_argument("--mode", choices=["article", "tldr", "takeaways", "transcript"], help="output mode")
    parser.add_argument("--timestamps", action="store_true", help="include linked timestamps")
    parser.add_argument("--no-chapters", action="store_true", help="disable chapter splitting")
    parser.add_argument("--lang", help="preferred transcript language code")
    parser.add_argument("--backend", choices=["ollama", "claude", "claude-code", "openai"], help="LLM backend")
    parser.add_argument("--model", help="model name")
    parser.add_argument("--theme", default="default", help="theme for HTML/EPUB output")
    parser.add_argument("--prompt", help="custom prompt (replaces mode)")
    parser.add_argument("--prompt-file", help="path to custom prompt file")
    parser.add_argument("--raw", action="store_true", help="skip transcript cleaning")
    parser.add_argument("--no-sponsorblock", action="store_true", help="disable SponsorBlock filtering")
    parser.add_argument("--genre", choices=["interview", "tutorial", "lecture", "documentary", "debate", "conference"], help="override genre detection")
    parser.add_argument("--list-genres", action="store_true", help="show available genres")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress even when piping")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress all progress")


def _build_single_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="readtube",
        description="Turn YouTube videos into readable articles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    parser.add_argument("url", help="YouTube video URL")
    _add_common_args(parser)
    return parser


def _build_playlist_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="readtube playlist")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("--max", type=int, help="max videos to process")
    _add_common_args(parser)
    return parser


def _build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="readtube batch")
    parser.add_argument("file", help="text file with one URL per line")
    _add_common_args(parser)
    return parser


def _build_config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="readtube config")
    parser.add_argument("--init", action="store_true", help="create default config file")
    return parser


def _build_cache_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="readtube cache")
    parser.add_argument("action", nargs="?", choices=["clear", "stats"], help="cache action")
    return parser


def _build_watch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="readtube watch")
    parser.add_argument("--init", action="store_true", help="create starter watch.toml")
    parser.add_argument("--config", help="path to watch.toml (default: ~/.config/readtube/watch.toml)")
    parser.add_argument("--dry-run", action="store_true", help="discover only, no LLM, no state update")
    parser.add_argument("--print-urls", action="store_true", help="print matching URLs to stdout, no processing")
    parser.add_argument("--discover", action="store_true", help="search YouTube for hits on untrusted channels (for curation)")
    parser.add_argument("--backend", choices=["ollama", "claude", "claude-code", "openai"], help="LLM backend")
    parser.add_argument("--model", help="model name")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress even when piping")
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress all progress")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args_list = argv if argv is not None else sys.argv[1:]

    # No arguments — show help
    if not args_list:
        _build_single_parser().print_help()
        sys.exit(EXIT_INVALID_ARGS)

    # Route based on first argument
    first = args_list[0]

    try:
        if first == "playlist":
            args = _build_playlist_parser().parse_args(args_list[1:])
            verbose = _should_show_progress(args)
            _cmd_playlist(args, verbose)

        elif first == "batch":
            args = _build_batch_parser().parse_args(args_list[1:])
            verbose = _should_show_progress(args)
            _cmd_batch(args, verbose)

        elif first == "config":
            args = _build_config_parser().parse_args(args_list[1:])
            _cmd_config(args)

        elif first == "cache":
            args = _build_cache_parser().parse_args(args_list[1:])
            _cmd_cache(args)

        elif first == "watch":
            args = _build_watch_parser().parse_args(args_list[1:])
            verbose = _should_show_progress(args)
            _cmd_watch(args, verbose)

        elif first in ("-h", "--help"):
            _build_single_parser().print_help()
            sys.exit(0)

        else:
            # Treat as a URL (single video mode)
            args = _build_single_parser().parse_args(args_list)
            if getattr(args, "list_genres", False):
                from .genre import list_genres
                for g in list_genres():
                    print(g)
                return
            verbose = _should_show_progress(args)
            _cmd_single(args, verbose)

    except ReadtubeError as e:
        die(e)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)


def _should_show_progress(args) -> bool:
    if getattr(args, "quiet", False):
        return False
    if getattr(args, "verbose", False):
        return True
    if getattr(args, "output", None):
        return True
    return sys.stderr.isatty()


def _resolve_custom_prompt(args) -> Optional[str]:
    """Read custom prompt from --prompt or --prompt-file."""
    if getattr(args, "prompt", None):
        return args.prompt
    if getattr(args, "prompt_file", None):
        from pathlib import Path
        return Path(args.prompt_file).read_text().strip()
    return None


def _cmd_single(args, verbose: bool) -> None:
    from .pipeline import process_single

    config = Config.load()
    custom_prompt = _resolve_custom_prompt(args)
    result = process_single(
        url=args.url,
        config=config,
        output_path=args.output,
        mode=args.mode,
        fmt=args.format,
        timestamps=args.timestamps,
        use_chapters=not args.no_chapters,
        lang=args.lang,
        cli_backend=args.backend,
        cli_model=args.model,
        verbose=verbose,
        theme=args.theme,
        custom_prompt=custom_prompt,
        raw=args.raw,
        no_sponsorblock=args.no_sponsorblock,
        genre=args.genre,
    )

    # Print to stdout if no output file
    if not args.output and result:
        # If streaming already printed to TTY, skip
        if not (sys.stdout.isatty() and args.mode != "transcript"):
            print(result)


def _cmd_playlist(args, verbose: bool) -> None:
    from pathlib import Path
    from .pipeline import process_playlist

    config = Config.load()
    output_dir = args.output or "."
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    custom_prompt = _resolve_custom_prompt(args)
    results = process_playlist(
        url=args.url,
        config=config,
        output_dir=output_dir,
        max_videos=args.max,
        mode=args.mode,
        fmt=args.format,
        timestamps=args.timestamps,
        use_chapters=not args.no_chapters,
        lang=args.lang,
        cli_backend=args.backend,
        cli_model=args.model,
        verbose=verbose,
        theme=args.theme,
        custom_prompt=custom_prompt,
        raw=args.raw,
        no_sponsorblock=args.no_sponsorblock,
        genre=args.genre,
    )

    progress(f"\nprocessed {len(results)} videos", verbose)


def _cmd_batch(args, verbose: bool) -> None:
    from pathlib import Path
    from .pipeline import process_batch

    config = Config.load()
    output_dir = args.output or "."
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    custom_prompt = _resolve_custom_prompt(args)
    results = process_batch(
        urls_file=args.file,
        config=config,
        output_dir=output_dir,
        mode=args.mode,
        fmt=args.format,
        timestamps=args.timestamps,
        use_chapters=not args.no_chapters,
        lang=args.lang,
        cli_backend=args.backend,
        cli_model=args.model,
        verbose=verbose,
        theme=args.theme,
        custom_prompt=custom_prompt,
        raw=args.raw,
        no_sponsorblock=args.no_sponsorblock,
        genre=args.genre,
    )

    progress(f"\nprocessed {len(results)} URLs", verbose)


def _cmd_config(args) -> None:
    if args.init:
        init_config()
    else:
        config = Config.load()
        print_config(config)


def _cmd_watch(args, verbose: bool) -> None:
    from .watch import init_watch_config, run_watch

    if args.init:
        init_watch_config()
        return

    run_watch(
        config_path=args.config,
        dry_run=args.dry_run,
        print_urls=args.print_urls,
        do_discover=args.discover,
        verbose=verbose,
        cli_backend=args.backend,
        cli_model=args.model,
    )


def _cmd_cache(args) -> None:
    from .cache import Cache
    config = Config.load()
    cache = Cache(config.cache.dir, config.cache.ttl_days)

    if args.action == "clear":
        count = cache.clear()
        print(f"cleared {count} cached entries", file=sys.stderr)
    elif args.action == "stats":
        stats = cache.stats()
        print(f"cache dir: {stats['cache_dir']}")
        print(f"entries: {stats['total_entries']}")
        print(f"size: {stats['total_size_kb']} KB")
        for name, info in stats["by_type"].items():
            print(f"  {name}: {info['count']} entries ({info['size_kb']} KB)")
    else:
        print("usage: readtube cache [clear|stats]", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)
