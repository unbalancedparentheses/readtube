---
name: readtube
description: Transform YouTube videos into magazine-style ebook articles
---

# Readtube

Transform YouTube videos into well-written magazine-style articles, delivered as EPUB, PDF, or HTML.

## Setup

**Option 1: Add as a Claude Code skill (easiest)**
```
/add-skill https://raw.githubusercontent.com/unbalancedparentheses/readtube/main/SKILL.md
```
Then ask: "Create an ebook from https://youtube.com/watch?v=VIDEO_ID"

**Option 2: Clone and work from the repo**
```bash
git clone https://github.com/unbalancedparentheses/readtube.git
cd readtube
pip install -r requirements.txt
```
Then ask Claude Code: "Create an ebook from https://youtube.com/watch?v=VIDEO_ID"

## How to Use This Skill

When the user wants to create an ebook from YouTube videos, follow this workflow:

### Step 1: Fetch Video and Transcript

```bash
# Single video
python fetch_transcript.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Multiple videos
python fetch_transcript.py "URL1" "URL2" "URL3"

# Playlist (all videos)
python fetch_transcript.py "https://www.youtube.com/playlist?list=PLxxx"

# Playlist with limit
python fetch_transcript.py "https://www.youtube.com/playlist?list=PLxxx" --max 5

# From configured channels
python fetch_transcript.py --channels

# With specific language
python fetch_transcript.py "URL" --lang es

# List available languages
python fetch_transcript.py "URL" --list-languages

# Summary mode (short 2-3 paragraph summary)
python fetch_transcript.py "URL" --summary

# Custom output directory
python fetch_transcript.py "URL" --output-dir ./ebooks
```

### Step 2: Write the Article

Using the transcript output, write a magazine-style article following these guidelines:

**Article Writing Guidelines:**
- Start with an engaging headline (different from the video title)
- The audience is a curious individual who is smart but not a specialist
- Make it highly engaging and readable - think New Yorker or The Atlantic
- Explain any jargon or obscure references
- Capture key insights, contrarian viewpoints, memorable anecdotes
- Preserve key quotes (clean up filler words or transcription errors)
- Use the video title/description to correct transcription errors
- Do NOT include phrases like "In this video" - write standalone
- Format in clean markdown
- Length depends on content and insight density

### Optional: Local LLM Drafting (no Claude Code)

If Claude Code isn't available, you can draft locally:

```bash
python fetch_transcript.py "URL" --output-json video.json
python write_article.py video.json --backend claude --output-dir ./drafts
# or use llama.cpp
python write_article.py video.json --backend llama-cpp --model /path/to/model.gguf --output-dir ./drafts
```

### Step 3: Create the Ebook

```python
from create_epub import create_ebook

articles = [{
    "title": "Original Video Title",
    "channel": "Channel Name",
    "url": "https://youtube.com/watch?v=...",
    "thumbnail": "https://...",  # Optional: used as cover
    "article": """# Your Article Headline

Your article content in markdown...
"""
}]

# Create EPUB (default)
create_ebook(articles, format="epub")

# Create PDF (requires weasyprint)
create_ebook(articles, format="pdf")

# Create HTML
create_ebook(articles, format="html")
```

## Features

- **Playlist support**: Process entire playlists with one URL
- **Multiple output formats**: EPUB, PDF, HTML
- **Thumbnail covers**: Automatically uses video thumbnail as cover
- **Chapter support**: Extracts video chapters for article structure hints
- **Language selection**: Choose preferred transcript language
- **Summary mode**: Request short summaries instead of full articles
- **Transcript caching**: Speeds up repeated requests (7-day cache)
- **Professional typography**: Inspired by Practical Typography
- **Speaker labels**: Preserves speaker identification when available
- **Configurable output**: Custom output directories

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.8+
- yt-dlp
- youtube-transcript-api
- ebooklib
- markdown

Optional for PDF: `pip install weasyprint`

**No API keys required!**

## Testing

```bash
make test        # Run all tests
make test-cov    # Run with coverage report
make test-e2e    # Run end-to-end tests
```

## Key Files

```
readtube/
├── fetch_transcript.py  # Fetch video info + transcript
├── create_epub.py       # Create EPUB/PDF/HTML from articles
├── get_videos.py        # Video fetching (yt-dlp)
├── get_transcripts.py   # Transcript extraction
├── tests/               # Test suite
├── Makefile             # Common commands
└── SKILL.md             # This file
```

## Configuring Channels

Edit `CHANNELS` in `get_videos.py`:

```python
CHANNELS = [
    "@LatentSpacePod",
    "@ycombinator",
    "@DwarkeshPatel",
]
```

## Troubleshooting

### No transcript available
Some videos don't have captions. Try `--list-languages` to see options.

### yt-dlp errors
Keep updated: `pip install --upgrade yt-dlp`

### PDF generation fails
Install weasyprint system dependencies. See: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html
