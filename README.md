# Readtube

Turn YouTube videos into beautifully typeset ebooks.

Readtube fetches transcripts from YouTube videos and transforms them into polished, magazine-style articles ready for your e-reader. No YouTube API key needed—just paste a URL.

## Installation

**Homebrew (macOS/Linux):**
```bash
brew tap unbalancedparentheses/readtube
brew install readtube
```

**Nix:**
```bash
git clone https://github.com/unbalancedparentheses/readtube.git
cd readtube && make shell
```

**pipx:**
```bash
pipx install git+https://github.com/unbalancedparentheses/readtube.git
```

**pip:**
```bash
pip install git+https://github.com/unbalancedparentheses/readtube.git
```

**One-liner:**
```bash
curl -sSL https://raw.githubusercontent.com/unbalancedparentheses/readtube/main/install.sh | bash
```

## Usage

### With Claude Code (recommended)

The easiest way to use Readtube is with Claude Code.

**Setup (choose one):**
```bash
# Option 1: Add as a skill (easiest - run this in Claude Code)
/add-skill https://raw.githubusercontent.com/unbalancedparentheses/readtube/main/SKILL.md

# Option 2: Clone the repo
git clone https://github.com/unbalancedparentheses/readtube.git
cd readtube
```

**Then just ask:**
```
Create an ebook from https://www.youtube.com/watch?v=VIDEO_ID
```

Claude will:
1. Fetch the video transcript
2. Write a polished, magazine-style article
3. Generate an EPUB with professional typography

Also works with ChatGPT/Codex and other AI coding assistants.

### Command Line

Fetch transcripts to process yourself:

```bash
# Fetch transcript and metadata
readtube "https://youtube.com/watch?v=VIDEO_ID"

# Fetch from a playlist
readtube "https://youtube.com/playlist?list=ID" --max 10

# Choose transcript language
readtube "URL" --lang es

# See available languages
readtube "URL" --list-languages

# Save as JSON for LLM processing
readtube "URL" --output-json video.json
```

### Local LLM Support

Readtube supports multiple LLM backends for article generation when not using Claude Code:

| Backend | Setup | Best For |
|---------|-------|----------|
| **llama-cpp** | `pip install llama-cpp-python` + download a `.gguf` model | Offline, privacy-focused |
| **Ollama** | Install from [ollama.ai](https://ollama.ai), run `ollama serve` | Easy local setup |
| **Claude API** | `pip install anthropic` + set `ANTHROPIC_API_KEY` | Best quality |
| **OpenAI** | Set `OPENAI_API_KEY` | GPT models |

**Auto-detection:** When running inside Claude Code or ChatGPT/Codex, Readtube automatically uses the host AI. Otherwise, it falls back to local LLMs in this order: llama-cpp → Ollama → Claude API → OpenAI.

**Environment variables:**
```
LLM_MAX_TOKENS=2048
LLM_TEMPERATURE=0.7
LLAMA_MODEL_PATH=/path/to/model.gguf
LLAMA_CTX=4096
LLAMA_GPU_LAYERS=-1
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

**Generate articles with local LLMs:**
```bash
# First, fetch the transcript
readtube "URL" --output-json video.json

# Generate with llama.cpp
python write_article.py video.json --backend llama-cpp --model /path/to/model.gguf --n-ctx 4096 --n-gpu-layers -1 --max-tokens 1600 --temperature 0.7

# Generate with Ollama
python write_article.py video.json --backend ollama --model llama3.2 --base-url http://localhost:11434

# Generate with Claude API
python write_article.py video.json --backend claude-api --model claude-sonnet-4-20250514

# Generate with OpenAI-compatible API (LM Studio / vLLM / OpenAI)
python write_article.py video.json --backend openai --model gpt-4o --base-url https://api.openai.com/v1

# Just create a prompt file (to paste into any LLM)
python write_article.py video.json --prompt-only
```

**Check available backends:**
```bash
python llm.py --list-backends
```

### Python API

```python
from create_epub import create_ebook

articles = [{
    "title": "How to Build a Startup",
    "channel": "Y Combinator",
    "url": "https://youtube.com/watch?v=...",
    "thumbnail": "https://i.ytimg.com/...",
    "article": "# Your markdown article here..."
}]

create_ebook(articles, format="epub")  # or pdf, html, mobi, azw3
```

## Why Readtube?

- **Read instead of watch**: Convert long-form videos into articles you can read anywhere
- **Better retention**: Reading lets you highlight, annotate, and revisit key ideas
- **Offline access**: EPUBs work on any e-reader without internet
- **Professional typography**: Clean, readable formatting based on Practical Typography principles

## Output Formats

- **EPUB** — For Kindle, Kobo, Apple Books, and other e-readers
- **PDF** — For printing or reading on tablets
- **HTML** — For web publishing
- **MOBI/AZW3** — Direct Kindle formats (requires Calibre)

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
```

```bash
python batch.py batch.yaml
```

## Additional Features

- **Themes**: `default`, `dark`, `modern`, `minimal`
- **Translation**: Google Translate, DeepL, LibreTranslate
- **Text-to-Speech**: Generate audio versions
- **RSS Feeds**: Podcast-style feeds from articles
- **Image Extraction**: Thumbnails and frames
- **Readwise Integration**: Sync to Readwise
- **Scheduled Fetching**: Cron/systemd/launchd support

## Typography

Ebooks follow [Practical Typography](https://practicaltypography.com/) principles:

- 65 character line length
- 1.4 line height
- Charter and Georgia fonts
- First-line paragraph indents

## PDF Support

PDF generation requires system libraries:

```bash
# macOS
brew install pango cairo glib

# Ubuntu/Debian
sudo apt install libpango-1.0-0 libcairo2 libglib2.0-0

# Then install with PDF support
pip install "readtube[pdf] @ git+https://github.com/unbalancedparentheses/readtube.git"
```

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
