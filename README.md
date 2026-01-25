# Readtube

Turn YouTube videos into beautifully typeset ebooks.

Readtube extracts transcripts from YouTube videos and transforms them into well-written, magazine-style articles. Output as EPUB for e-readers, PDF for printing, or HTML for the web.

## Features

**Core:**
- **Video sources**: Single videos, playlists, or channel feeds
- **Output formats**: EPUB, PDF, HTML
- **Video chapters**: Extracted to help structure articles
- **Thumbnail covers**: Video thumbnails become book covers
- **Multiple languages**: Select preferred transcript language
- **Summary mode**: Short summaries instead of full articles
- **Transcript caching**: 7-day cache for faster repeated requests
- **Timestamps**: Include timestamps in transcripts for reference
- **Professional typography**: Based on [Practical Typography](https://practicaltypography.com/)
- **No API keys**: Uses yt-dlp (no YouTube API key needed)

**Advanced:**
- **Batch processing**: Process multiple videos from YAML/JSON config
- **Custom themes**: Built-in themes (default, dark, modern, minimal) or custom CSS
- **Web dashboard**: Simple Flask web UI for conversions
- **RSS/Atom feeds**: Generate podcast-style feeds from articles
- **Text-to-speech**: Audio narration (pyttsx3, gTTS, Edge TTS, macOS say)
- **Readwise integration**: Send articles and highlights to Readwise
- **Progress indicators**: tqdm progress bars for batch processing
- **Retry logic**: Automatic retry with exponential backoff
- **Image extraction**: Extract thumbnails and frames at timestamps

## Installation

```bash
# Clone
git clone https://github.com/unbalancedparentheses/readtube.git
cd readtube

# Install
make install

# For PDF support (requires system libraries)
make install-pdf

# Or use Nix (handles all dependencies)
make shell
```

## Usage

### With Claude Code

The easiest way to use Readtube is as a Claude Code skill:

```
Create an ebook from https://www.youtube.com/watch?v=VIDEO_ID
```

Claude will fetch the transcript, write an article, and generate the ebook.

### Command Line

```bash
# Fetch transcript from a video
python fetch_transcript.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Fetch from a playlist (max 5 videos)
python fetch_transcript.py "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max 5

# List available languages
python fetch_transcript.py "URL" --list-languages

# Fetch in Spanish
python fetch_transcript.py "URL" --lang es

# Request summary mode
python fetch_transcript.py "URL" --summary

# Custom output directory
python fetch_transcript.py "URL" --output-dir ./ebooks
```

### Programmatic

```python
from create_epub import create_ebook

articles = [{
    "title": "Video Title",
    "channel": "Channel Name",
    "url": "https://youtube.com/watch?v=...",
    "thumbnail": "https://...",  # Optional: becomes cover
    "article": "# Markdown article content..."
}]

# Create EPUB
create_ebook(articles, format="epub")

# Create PDF (requires weasyprint + system deps)
create_ebook(articles, format="pdf")

# Create HTML
create_ebook(articles, format="html")
```

### Batch Processing

```bash
# Create a batch config (batch.yaml)
cat > batch.yaml << EOF
output_dir: ./ebooks
default_format: epub

jobs:
  - url: https://www.youtube.com/watch?v=VIDEO1
  - url: https://www.youtube.com/watch?v=VIDEO2
    output_format: pdf
  - url: https://www.youtube.com/playlist?list=PLAYLIST
    summary_mode: true
EOF

# Process batch
python batch.py batch.yaml
```

### Web Dashboard

```bash
# Start web UI
python web.py --port 8080
# Open http://localhost:8080
```

### Custom Themes

```python
from themes import get_theme, list_themes

# List available themes
print(list_themes())  # default, dark, modern, minimal

# Get a theme
theme = get_theme("dark")
```

### RSS/Atom Feeds

```python
from rss import generate_rss_feed, generate_atom_feed

articles = [...]  # Your articles
generate_rss_feed(articles, output_path="feed.xml")
generate_atom_feed(articles, output_path="feed.atom")
```

### Text-to-Speech

```bash
# List available TTS backends
python tts.py --list-backends

# Generate audio from article
python tts.py article.md --output article.mp3 --backend edge
```

### Readwise Integration

```bash
export READWISE_TOKEN="your_token"
python -c "from integrations import send_to_readwise; send_to_readwise(article)"
```

### Image Extraction

```bash
# Download all thumbnails
python images.py "https://youtube.com/watch?v=VIDEO_ID" --thumbnails

# Extract frames at specific timestamps
python images.py "URL" --timestamps 0:30,1:00,2:30 --output ./frames

# Extract frames at 60 second intervals
python images.py "URL" --interval 60 --max-frames 10

# Extract frame at each chapter start
python images.py "URL" --chapters
```

## Typography

Ebooks are typeset following [Practical Typography](https://practicaltypography.com/) principles:

- **Line length**: 45-90 characters (65ch default)
- **Line spacing**: 140% (within 120-145% optimal range)
- **Paragraph style**: First-line indents, no space between
- **Headings**: Subtle size hierarchy, emphasis via spacing
- **Fonts**: Charter, Georgia (serif) for body text
- **Kerning**: Always enabled

## Development

```bash
make help        # Show all commands
make test        # Run tests
make test-cov    # Run with coverage
make lint        # Run linter
make clean       # Remove generated files
```

## Project Structure

```
readtube/
├── fetch_transcript.py   # CLI: fetch video + transcript
├── create_epub.py        # Generate EPUB/PDF/HTML
├── get_videos.py         # yt-dlp video fetching
├── get_transcripts.py    # Transcript extraction + caching
├── config.py             # Central configuration
├── batch.py              # Batch processing
├── themes.py             # Custom CSS themes
├── web.py                # Flask web UI
├── rss.py                # RSS/Atom feed generation
├── tts.py                # Text-to-speech audio
├── images.py             # Image/frame extraction
├── integrations.py       # Readwise integration
├── SKILL.md              # Claude Code skill definition
├── tests/                # Test suite
├── Makefile              # Build commands
├── shell.nix             # Nix development environment
├── Dockerfile            # Docker support
└── requirements.txt      # Python dependencies
```

## Requirements

- Python 3.8+
- yt-dlp
- youtube-transcript-api
- ebooklib
- markdown

**Optional for PDF:**
- weasyprint
- System libraries: Pango, Cairo, GLib

## Acknowledgments

Typography principles from [Practical Typography](https://practicaltypography.com/) by Matthew Butterick.

## License

MIT
