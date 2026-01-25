#!/usr/bin/env python3
"""
Scheduled fetching for Readtube.
Automatically fetch new videos from channels/playlists on a schedule.

Usage:
    python scheduler.py config.yaml --interval 3600  # Run every hour
    python scheduler.py config.yaml --cron "0 8 * * *"  # Run daily at 8am
    python scheduler.py --daemon  # Run as background daemon
"""

import os
import sys
import time
import json
import signal
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from threading import Thread, Event

from config import BatchConfig, logger

# Try to import schedule library
try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

# Try to import croniter for cron expressions
try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False


class FetchHistory:
    """Track fetch history to avoid re-fetching."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path.home() / '.config' / 'readtube' / 'fetch_history.json'
        self.history: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load history from file."""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'fetched': {}, 'last_run': None}

    def _save(self) -> None:
        """Save history to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def is_fetched(self, video_id: str) -> bool:
        """Check if a video has been fetched."""
        return video_id in self.history.get('fetched', {})

    def mark_fetched(self, video_id: str, data: Optional[Dict] = None) -> None:
        """Mark a video as fetched."""
        self.history.setdefault('fetched', {})[video_id] = {
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        self._save()

    def get_last_run(self) -> Optional[datetime]:
        """Get timestamp of last run."""
        last_run = self.history.get('last_run')
        if last_run:
            return datetime.fromisoformat(last_run)
        return None

    def update_last_run(self) -> None:
        """Update last run timestamp."""
        self.history['last_run'] = datetime.now().isoformat()
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Get fetch statistics."""
        fetched = self.history.get('fetched', {})
        return {
            'total_fetched': len(fetched),
            'last_run': self.history.get('last_run'),
        }


class Scheduler:
    """Schedule and run fetch jobs."""

    def __init__(
        self,
        config_path: Path,
        history: Optional[FetchHistory] = None,
        on_fetch: Optional[Callable[[Dict], None]] = None
    ):
        self.config_path = config_path
        self.history = history or FetchHistory()
        self.on_fetch = on_fetch
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    def fetch_new_videos(self) -> List[Dict[str, Any]]:
        """
        Fetch new videos that haven't been processed yet.

        Returns:
            List of new video data
        """
        from get_videos import get_video_info, get_videos_from_playlist, is_playlist_url
        from get_transcripts import get_transcript

        batch_config = BatchConfig.load(self.config_path)
        new_videos = []

        for job in batch_config.jobs:
            try:
                if is_playlist_url(job.url):
                    videos = get_videos_from_playlist(job.url)
                else:
                    video = get_video_info(job.url)
                    videos = [video] if video else []

                for video in videos:
                    if video and not self.history.is_fetched(video['video_id']):
                        # Fetch transcript
                        transcript = get_transcript(
                            video['video_id'],
                            lang=job.language or batch_config.default_language
                        )

                        if transcript:
                            video['transcript'] = transcript
                            video['word_count'] = len(transcript.split())
                            new_videos.append(video)

                            # Mark as fetched
                            self.history.mark_fetched(video['video_id'], {
                                'title': video.get('title'),
                                'channel': video.get('channel'),
                            })

                            logger.info(f"New video: {video.get('title', 'Unknown')[:50]}")

                            # Callback
                            if self.on_fetch:
                                self.on_fetch(video)

                # Rate limiting
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error processing {job.url}: {e}")

        self.history.update_last_run()
        return new_videos

    def run_once(self) -> List[Dict[str, Any]]:
        """Run a single fetch cycle."""
        logger.info(f"Starting fetch at {datetime.now().isoformat()}")
        videos = self.fetch_new_videos()
        logger.info(f"Fetch complete: {len(videos)} new videos")
        return videos

    def run_interval(self, interval_seconds: int) -> None:
        """
        Run fetch at regular intervals.

        Args:
            interval_seconds: Seconds between fetches
        """
        logger.info(f"Starting scheduler with {interval_seconds}s interval")

        while not self._stop_event.is_set():
            self.run_once()

            # Wait for next interval or stop event
            self._stop_event.wait(timeout=interval_seconds)

        logger.info("Scheduler stopped")

    def run_cron(self, cron_expression: str) -> None:
        """
        Run fetch on a cron schedule.

        Args:
            cron_expression: Cron expression (e.g., "0 8 * * *" for daily at 8am)
        """
        if not HAS_CRONITER:
            raise ImportError("croniter required for cron schedules: pip install croniter")

        logger.info(f"Starting scheduler with cron: {cron_expression}")

        cron = croniter(cron_expression, datetime.now())

        while not self._stop_event.is_set():
            # Calculate next run time
            next_run = cron.get_next(datetime)
            wait_seconds = (next_run - datetime.now()).total_seconds()

            logger.info(f"Next run at {next_run.isoformat()}")

            # Wait for next run or stop event
            if self._stop_event.wait(timeout=max(0, wait_seconds)):
                break

            self.run_once()

        logger.info("Scheduler stopped")

    def run_schedule_library(self) -> None:
        """
        Run using the schedule library (requires: pip install schedule).
        Call this after setting up schedules with the schedule library.
        """
        if not HAS_SCHEDULE:
            raise ImportError("schedule library required: pip install schedule")

        logger.info("Starting scheduler with schedule library")

        while not self._stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

        logger.info("Scheduler stopped")

    def start_background(self, interval_seconds: int) -> None:
        """Start scheduler in background thread."""
        self._thread = Thread(target=self.run_interval, args=(interval_seconds,), daemon=True)
        self._thread.start()
        logger.info("Scheduler started in background")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")


def create_systemd_service(config_path: str, interval: int = 3600) -> str:
    """
    Generate a systemd service file for running the scheduler.

    Args:
        config_path: Path to batch config file
        interval: Fetch interval in seconds

    Returns:
        Systemd service file content
    """
    python_path = sys.executable
    script_path = Path(__file__).resolve()
    working_dir = script_path.parent

    return f"""[Unit]
Description=Readtube Scheduled Fetcher
After=network.target

[Service]
Type=simple
User={os.getlogin()}
WorkingDirectory={working_dir}
ExecStart={python_path} {script_path} {config_path} --interval {interval}
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
"""


def create_launchd_plist(config_path: str, interval: int = 3600) -> str:
    """
    Generate a macOS launchd plist for running the scheduler.

    Args:
        config_path: Path to batch config file
        interval: Fetch interval in seconds

    Returns:
        Launchd plist content
    """
    python_path = sys.executable
    script_path = Path(__file__).resolve()
    working_dir = script_path.parent

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.readtube.scheduler</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
        <string>{config_path}</string>
        <string>--interval</string>
        <string>{interval}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/readtube-scheduler.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/readtube-scheduler.log</string>
</dict>
</plist>
"""


def main():
    parser = argparse.ArgumentParser(description="Scheduled video fetching")
    parser.add_argument("config", nargs="?", type=Path, help="Batch config file")
    parser.add_argument("--interval", "-i", type=int, help="Fetch interval in seconds")
    parser.add_argument("--cron", help="Cron expression for scheduling")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (background)")
    parser.add_argument("--stats", action="store_true", help="Show fetch statistics")
    parser.add_argument("--generate-systemd", action="store_true", help="Generate systemd service file")
    parser.add_argument("--generate-launchd", action="store_true", help="Generate macOS launchd plist")

    args = parser.parse_args()

    # Show stats
    if args.stats:
        history = FetchHistory()
        stats = history.get_stats()
        print(f"Total videos fetched: {stats['total_fetched']}")
        print(f"Last run: {stats['last_run'] or 'Never'}")
        return

    # Generate service files
    if args.generate_systemd:
        if not args.config:
            print("Error: config file required")
            sys.exit(1)
        print(create_systemd_service(str(args.config), args.interval or 3600))
        return

    if args.generate_launchd:
        if not args.config:
            print("Error: config file required")
            sys.exit(1)
        print(create_launchd_plist(str(args.config), args.interval or 3600))
        return

    # Validate config
    if not args.config:
        print("Error: config file required")
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    # Create scheduler
    scheduler = Scheduler(args.config)

    # Handle signals
    def signal_handler(signum, frame):
        logger.info("Received signal, stopping...")
        scheduler.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    if args.once:
        videos = scheduler.run_once()
        print(f"Fetched {len(videos)} new videos")

    elif args.cron:
        scheduler.run_cron(args.cron)

    elif args.interval:
        if args.daemon:
            scheduler.start_background(args.interval)
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                scheduler.stop()
        else:
            scheduler.run_interval(args.interval)

    else:
        print("Error: specify --interval, --cron, or --once")
        sys.exit(1)


if __name__ == "__main__":
    main()
