<p align="center">
  <h1 align="center">Readtube</h1>
  <p align="center">
    <strong>Turn YouTube videos into beautifully typeset ebooks</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a>
  </p>
</p>

---

**Readtube** extracts transcripts from YouTube videos and transforms them into polished, magazine-style articles. Output as EPUB for e-readers, PDF for printing, or HTML for the web.

No YouTube API key required. Just paste a URL and get a beautifully formatted ebook.

```bash
python fetch_transcript.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/unbalancedparentheses/readtube.git
cd readtube && make install

# Convert a video to EPUB
python fetch_transcript.py "https://youtube.com/watch?v=dQw4w9WgXcQ"

# Convert a playlist
python fetch_transcript.py "https://youtube.com/playlist?list=PLxyz" --max 5

# Get a summary instead of full article
python fetch_transcript.py "URL" --summary
```

## Features

| Feature | Description |
|---------|-------------|
| **Multiple Sources** | Single videos, playlists, or entire channels |
| **Output Formats** | EPUB, PDF, HTML, MOBI, AZW3 (Kindle) |
| **Smart Chapters** | Video chapters become article sections |
| **Cover Art** | Video thumbnails as book covers |
| **Multi-language** | Select preferred transcript language |
| **Caching** | 7-day transcript cache for speed |
| **No API Keys** | Uses yt-dlp, no YouTube API needed |

### Typography

Ebooks follow [Practical Typography](https://practicaltypography.com/) principles:

- **Line length**: 65 characters (optimal readability)
- **Line height**: 1.4 (comfortable reading)
- **Fonts**: Charter, Georgia (elegant serifs)
- **Paragraphs**: First-line indents, proper spacing

### Advanced Features

<details>
<summary><strong>Batch Processing</strong></summary>

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
</details>

<details>
<summary><strong>Custom Themes</strong></summary>

Built-in themes: `default`, `dark`, `modern`, `minimal`

```python
from themes import get_theme, list_themes

print(list_themes())  # ['default', 'dark', 'modern', 'minimal']
theme = get_theme("dark")
```
</details>

<details>
<summary><strong>RSS/Atom Feeds</strong></summary>

Generate podcast-style feeds from your articles:

```python
from rss import generate_rss_feed, generate_atom_feed

generate_rss_feed(articles, output_path="feed.xml")
generate_atom_feed(articles, output_path="feed.atom")
```
</details>

<details>
<summary><strong>Text-to-Speech</strong></summary>

Convert articles to audio:

```bash
python tts.py article.md --output article.mp3 --backend edge
python tts.py --list-backends  # pyttsx3, gtts, edge, macos
```
</details>

<details>
<summary><strong>Translation</strong></summary>

Translate transcripts to other languages:

```bash
python translate.py transcript.txt --to spanish --output es.txt
python translate.py --text "Hello" --to de --backend google
```
</details>

<details>
<summary><strong>Scheduled Fetching</strong></summary>

Automate fetching on a schedule:

```bash
# Every hour
python scheduler.py batch.yaml --interval 3600

# Daily at 8am
python scheduler.py batch.yaml --cron "0 8 * * *"

# Generate systemd service
python scheduler.py batch.yaml --generate-systemd > readtube.service
```
</details>

<details>
<summary><strong>Image Extraction</strong></summary>

Extract thumbnails and frames:

```bash
python images.py "URL" --thumbnails
python images.py "URL" --timestamps 0:30,1:00,2:30
python images.py "URL" --chapters
```
</details>

<details>
<summary><strong>Readwise Integration</strong></summary>

Send articles to Readwise:

```bash
export READWISE_TOKEN="your_token"
python -c "from integrations import send_to_readwise; send_to_readwise(article)"
```
</details>

## Installation

### Standard

```bash
git clone https://github.com/unbalancedparentheses/readtube.git
cd readtube
make install
```

### With PDF Support

PDF generation requires system libraries (Pango, Cairo, GLib):

```bash
# macOS
brew install pango cairo glib

# Ubuntu/Debian
sudo apt install libpango-1.0-0 libcairo2 libglib2.0-0

# Then install Python deps
make install-pdf
```

### With Nix

Nix handles all dependencies automatically:

```bash
make shell
```

## Usage

### Command Line

```bash
# Basic usage
python fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# Playlist (limit to 5 videos)
python fetch_transcript.py "https://youtube.com/playlist?list=ID" --max 5

# Specific language
python fetch_transcript.py "URL" --lang es

# Summary mode
python fetch_transcript.py "URL" --summary

# Custom output directory
python fetch_transcript.py "URL" --output-dir ./ebooks

# List available languages
python fetch_transcript.py "URL" --list-languages
```

### Python API

```python
from create_epub import create_ebook

articles = [{
    "title": "Video Title",
    "channel": "Channel Name",
    "url": "https://youtube.com/watch?v=...",
    "thumbnail": "https://...",
    "article": "# Your markdown content..."
}]

create_ebook(articles, format="epub")  # or pdf, html, mobi, azw3
```

### With Claude Code

Use Readtube as a Claude Code skill:

```
Create an ebook from https://www.youtube.com/watch?v=VIDEO_ID
```

Claude will fetch the transcript, write the article, and generate the ebook.

## Project Structure

```
readtube/
├── fetch_transcript.py   # Main CLI
├── create_epub.py        # EPUB/PDF/HTML generation
├── get_videos.py         # YouTube video fetching
├── get_transcripts.py    # Transcript extraction
├── batch.py              # Batch processing
├── themes.py             # CSS themes
├── scheduler.py          # Cron-like scheduling
├── translate.py          # Translation
├── tts.py                # Text-to-speech
├── images.py             # Image extraction
├── rss.py                # RSS/Atom feeds
├── integrations.py       # Readwise integration
├── config.py             # Configuration
├── async_fetch.py        # Async support
└── tests/                # Test suite
```

## Development

```bash
make help      # Show all commands
make test      # Run tests
make test-cov  # Tests with coverage
make lint      # Run linter
make clean     # Clean generated files
```

## Requirements

- Python 3.8+
- yt-dlp
- youtube-transcript-api
- ebooklib
- markdown

## License

MIT

---

<p align="center">
  Typography principles from <a href="https://practicaltypography.com/">Practical Typography</a> by Matthew Butterick.
</p>
