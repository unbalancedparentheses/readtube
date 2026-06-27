# Readtube

Turn YouTube videos into articles you actually want to read.

Readtube fetches transcripts from YouTube videos and transforms them into
polished, well-structured articles using any LLM. Local-first, Unix-native,
zero accounts required.

```bash
$ readtube "https://youtube.com/watch?v=VIDEO_ID"
# The Art of Doing Science and Engineering

Richard Hamming's legendary lecture series at the Naval Postgraduate School...
```

## Install

```bash
# Minimal (Ollama + markdown output)
pip install git+https://github.com/unbalancedparentheses/readtube.git

# With EPUB support
pip install "readtube[epub] @ git+https://github.com/unbalancedparentheses/readtube.git"

# With Claude API
pip install "readtube[claude] @ git+https://github.com/unbalancedparentheses/readtube.git"

# Everything
pip install "readtube[all] @ git+https://github.com/unbalancedparentheses/readtube.git"
```

**Requirements:** Python 3.10+ and [yt-dlp](https://github.com/yt-dlp/yt-dlp).

## Quick Start

```bash
# Article to terminal (streams in real time)
readtube "https://youtube.com/watch?v=VIDEO_ID"

# Save as EPUB for your e-reader
readtube "https://youtube.com/watch?v=VIDEO_ID" -o article.epub

# Quick summary
readtube "https://youtube.com/watch?v=VIDEO_ID" --mode tldr

# Key takeaways
readtube "https://youtube.com/watch?v=VIDEO_ID" --mode takeaways

# Just the transcript, no LLM needed
readtube "https://youtube.com/watch?v=VIDEO_ID" --mode transcript
```

## Output Modes

| Mode | What you get |
|------|-------------|
| `article` | Full magazine-style article with headings and sections (default) |
| `tldr` | 3-5 bullet points capturing the key ideas |
| `takeaways` | Structured list grouped by topic with specific claims and data |
| `transcript` | Cleaned transcript — no LLM required |

## Output Formats

| Format | Command | Notes |
|--------|---------|-------|
| Markdown | `readtube URL` | Default, streams to stdout |
| Markdown file | `readtube URL -o article.md` | |
| EPUB | `readtube URL -o article.epub` | Requires `readtube[epub]` |
| PDF | `readtube URL -o article.pdf` | Requires `readtube[pdf]` |
| HTML | `readtube URL -o article.html` | Standalone with embedded CSS |

## LLM Backends

Readtube supports four backends. It auto-detects which one to use:

| Backend | Setup | Best for |
|---------|-------|----------|
| **Claude Code** | None — run `readtube` from inside a [Claude Code](https://claude.com/claude-code) session | Zero-config, no API key |
| **Ollama** | [Install Ollama](https://ollama.ai), run `ollama serve` | Free, private, local |
| **Claude** | Set `ANTHROPIC_API_KEY` | Best quality |
| **OpenAI** | Set `OPENAI_API_KEY` | GPT models |

When run from Claude Code, readtube shells out to the `claude` CLI and borrows
the host session's auth — so it just works with no key or local model.

**Auto-detection order:** Claude Code session (`CLAUDECODE` set + `claude` on PATH) → Ollama running locally → `ANTHROPIC_API_KEY` set → `OPENAI_API_KEY` set.

Override with flags:

```bash
readtube URL --backend claude-code                 # uses the `claude` CLI; --model optional
readtube URL --backend claude --model claude-sonnet-4-20250514
readtube URL --backend ollama --model llama3.2
readtube URL --backend openai --model gpt-4o
```

## Features

### Timestamps

Link every key point back to the exact moment in the video:

```bash
readtube URL --timestamps
```

Produces:

```markdown
Neural networks learn through backpropagation [4:32](https://youtube.com/watch?v=xxx&t=272),
adjusting weights layer by layer...
```

### Chapter-Aware Articles

When a video has chapters, readtube uses them as article sections:

```bash
readtube URL --chapters
```

### Playlists

```bash
readtube playlist "https://youtube.com/playlist?list=PLxxx" -o ./articles/
readtube playlist "https://youtube.com/playlist?list=PLxxx" --max 10
```

### Batch Processing

```bash
# urls.txt: one URL per line, # comments, blank lines ignored
readtube batch urls.txt -o ./articles/
```

### Transcript Mode (No LLM)

Fetch and clean transcripts without any LLM:

```bash
readtube URL --mode transcript              # to stdout
readtube URL --mode transcript | wc -w      # word count
readtube URL --mode transcript | llm "summarize this"  # pipe to another tool
```

## Configuration

```bash
readtube config --init   # create default config
readtube config          # show current config
```

Config lives at `~/.config/readtube/config.toml`:

```toml
[llm]
backend = "ollama"           # ollama | claude | claude-code | openai
model = "llama3.2"
# api_key_env = "ANTHROPIC_API_KEY"  # env var name, never the key itself

[output]
default_format = "md"        # md | epub | pdf | html
default_mode = "article"     # article | tldr | takeaways | transcript
timestamps = false
chapters = true

[cache]
dir = "~/.cache/readtube"
ttl_days = 30
```

**Precedence:** CLI flags > env vars > config file > auto-detect.

## Unix Pipeline Friendly

- **stdout** = content (article text)
- **stderr** = progress (fetching, generating, etc.)
- **Exit codes**: 0 success, 1 general, 2 bad args, 3 video not found, 4 no transcript, 5 LLM error, 6 output format error

```bash
readtube URL | head -20                     # first 20 lines
readtube URL > article.md 2>/dev/null       # suppress progress
readtube URL --mode transcript | grep -i "neural"  # search transcript
```

## Cache

Transcripts are cached locally to avoid re-fetching:

```bash
readtube cache stats    # show cache size
readtube cache clear    # wipe cache
```

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
git clone https://github.com/unbalancedparentheses/readtube.git
cd readtube
pip install -e ".[all]"
pytest
```

87 tests. 13 source files. 2 core dependencies.

## License

MIT
