# Readtube

Turn YouTube videos into beautifully typeset ebooks.

Readtube fetches transcripts from YouTube videos and transforms them into polished, magazine-style articles ready for your e-reader. No YouTube API key needed—just paste a URL.

## Why Readtube?

- **Read instead of watch**: Convert long-form videos into articles you can read anywhere
- **Better retention**: Reading lets you highlight, annotate, and revisit key ideas
- **Offline access**: EPUBs work on any e-reader without internet
- **Professional typography**: Clean, readable formatting based on Practical Typography principles

## Quick Start

```bash
git clone https://github.com/unbalancedparentheses/readtube.git
cd readtube && make install

python fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"
```

That's it. You'll get an EPUB in the current directory.

## What It Does

1. **Fetches** video metadata and transcript using yt-dlp
2. **Transforms** the raw transcript into a well-structured article
3. **Generates** an ebook with proper typography and optional cover art

## Output Formats

- **EPUB** — For Kindle, Kobo, Apple Books, and other e-readers
- **PDF** — For printing or reading on tablets
- **HTML** — For web publishing
- **MOBI/AZW3** — Direct Kindle formats (requires Calibre)

## CLI Options

```bash
# Single video
python fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# Playlist (limit to 10 videos)
python fetch_transcript.py "https://youtube.com/playlist?list=ID" --max 10

# Choose transcript language
python fetch_transcript.py "URL" --lang es

# Generate a short summary instead of full article
python fetch_transcript.py "URL" --summary

# Save to specific directory
python fetch_transcript.py "URL" --output-dir ./ebooks

# See available transcript languages
python fetch_transcript.py "URL" --list-languages
```

## Python API

```python
from create_epub import create_ebook

articles = [{
    "title": "How to Build a Startup",
    "channel": "Y Combinator",
    "url": "https://youtube.com/watch?v=...",
    "thumbnail": "https://i.ytimg.com/...",  # becomes the cover
    "article": "# Introduction\n\nYour markdown content here..."
}]

create_ebook(articles, format="epub")
```

## Batch Processing

Process multiple videos from a config file:

```yaml
# batch.yaml
output_dir: ./ebooks
default_format: epub

jobs:
  - url: https://youtube.com/watch?v=VIDEO1
  - url: https://youtube.com/watch?v=VIDEO2
    output_format: pdf
  - url: https://youtube.com/playlist?list=PLAYLIST
    summary_mode: true
```

```bash
python batch.py batch.yaml
```

## Scheduled Fetching

Automatically fetch new videos on a schedule:

```bash
# Run every hour
python scheduler.py batch.yaml --interval 3600

# Run daily at 8am
python scheduler.py batch.yaml --cron "0 8 * * *"

# Generate a systemd service file
python scheduler.py batch.yaml --generate-systemd > /etc/systemd/system/readtube.service
```

## Additional Features

**Themes**: Choose from `default`, `dark`, `modern`, or `minimal` styles.

**Translation**: Translate transcripts to other languages using Google Translate, DeepL, or LibreTranslate.

**Text-to-Speech**: Generate audio versions of articles.

**RSS Feeds**: Create podcast-style feeds from your converted articles.

**Image Extraction**: Pull thumbnails or extract frames at specific timestamps.

**Readwise Integration**: Send articles directly to your Readwise account.

## Typography

The generated ebooks follow [Practical Typography](https://practicaltypography.com/) principles:

- Line length of 65 characters for optimal readability
- 1.4 line height for comfortable reading
- Charter and Georgia fonts
- First-line paragraph indents
- Proper heading hierarchy

## Installation

**Basic:**
```bash
git clone https://github.com/unbalancedparentheses/readtube.git
cd readtube
make install
```

**With PDF support** (requires system libraries):
```bash
# macOS
brew install pango cairo glib

# Ubuntu/Debian
sudo apt install libpango-1.0-0 libcairo2 libglib2.0-0

make install-pdf
```

**With Nix** (handles all dependencies):
```bash
make shell
```

## Requirements

- Python 3.8+
- yt-dlp
- youtube-transcript-api
- ebooklib
- markdown

## Development

```bash
make test      # Run tests
make test-cov  # Tests with coverage
make lint      # Run linter
```

## License

MIT

---

Typography principles from [Practical Typography](https://practicaltypography.com/) by Matthew Butterick.
