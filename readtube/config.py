"""TOML-based configuration with CLI flag override support."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".config" / "readtube"
CONFIG_FILE = CONFIG_DIR / "config.toml"
CACHE_DIR = Path.home() / ".cache" / "readtube"


@dataclass
class LLMConfig:
    backend: Optional[str] = None
    model: Optional[str] = None
    api_key_env: Optional[str] = None
    url: Optional[str] = None


@dataclass
class OutputConfig:
    default_format: str = "md"
    default_mode: str = "article"
    timestamps: bool = False
    chapters: bool = True


@dataclass
class CacheConfig:
    dir: str = ""
    ttl_days: int = 30

    def __post_init__(self):
        if not self.dir:
            self.dir = str(CACHE_DIR)


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Config:
        """Load config from TOML file, or return defaults."""
        config_path = path or CONFIG_FILE
        config = cls()

        if not config_path.exists():
            return config

        try:
            text = config_path.read_text()
            data = _parse_toml(text)
        except Exception:
            return config

        if "llm" in data:
            llm = data["llm"]
            config.llm = LLMConfig(
                backend=llm.get("backend"),
                model=llm.get("model"),
                api_key_env=llm.get("api_key_env"),
                url=llm.get("url"),
            )

        if "output" in data:
            out = data["output"]
            config.output = OutputConfig(
                default_format=out.get("default_format", "md"),
                default_mode=out.get("default_mode", "article"),
                timestamps=out.get("timestamps", False),
                chapters=out.get("chapters", True),
            )

        if "cache" in data:
            c = data["cache"]
            config.cache = CacheConfig(
                dir=c.get("dir", str(CACHE_DIR)),
                ttl_days=c.get("ttl_days", 30),
            )

        return config


def _parse_toml(text: str) -> dict:
    """Parse TOML using stdlib (3.11+) or tomli fallback."""
    try:
        import tomllib
        return tomllib.loads(text)
    except ImportError:
        pass
    try:
        import tomli
        return tomli.loads(text)
    except ImportError:
        raise ImportError("Python 3.11+ or 'tomli' package required for TOML config")


def resolve_llm_config(
    config: Config,
    cli_backend: Optional[str] = None,
    cli_model: Optional[str] = None,
) -> tuple[str, str, Optional[str]]:
    """Resolve LLM backend, model, and API key from CLI flags > env > config > auto-detect.

    Returns:
        (backend, model, api_key) tuple. Raises LLMError if nothing found.
    """
    from .errors import LLMError

    # 1. CLI flags
    backend = cli_backend
    model = cli_model

    # 2. Environment variables
    if not backend:
        backend = os.environ.get("READTUBE_BACKEND")
    if not model:
        model = os.environ.get("READTUBE_MODEL")

    # 3. Config file
    if not backend and config.llm.backend:
        backend = config.llm.backend
    if not model and config.llm.model:
        model = config.llm.model

    # 4. Auto-detect
    if not backend:
        backend = _auto_detect_backend()

    if not backend:
        raise LLMError(
            "none",
            "no LLM backend configured.\n"
            "  → run from Claude Code (uses the `claude` CLI automatically),\n"
            "  → or set ANTHROPIC_API_KEY, or run: readtube config --init",
        )

    # Resolve API key from env var name in config
    api_key = None
    if config.llm.api_key_env:
        api_key = os.environ.get(config.llm.api_key_env)

    # Fallback: check common env vars based on backend
    if not api_key:
        if backend == "claude":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif backend == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")

    # Default models per backend. claude-code is intentionally absent: leave
    # the model unset so the Claude Code CLI picks its own session default.
    if not model and backend != "claude-code":
        defaults = {
            "ollama": "llama3.2",
            "claude": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
        }
        model = defaults.get(backend, "llama3.2")

    # Resolve URL
    url = config.llm.url
    if backend == "ollama" and not url:
        url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    return backend, model, api_key


def _auto_detect_backend() -> Optional[str]:
    """Try to auto-detect an available LLM backend."""
    # Prefer the host Claude Code session — zero-config when run from Claude Code.
    if os.environ.get("CLAUDECODE"):
        import shutil

        if shutil.which("claude"):
            return "claude-code"

    # Check if Ollama is running
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2):
            return "ollama"
    except Exception:
        pass

    # Check for API keys
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"

    return None


def print_config(config: Config) -> None:
    """Print current configuration to stdout."""
    print("[llm]")
    print(f"  backend = {config.llm.backend or '(auto-detect)'}")
    print(f"  model = {config.llm.model or '(default)'}")
    if config.llm.url:
        print(f"  url = {config.llm.url}")
    print()
    print("[output]")
    print(f"  default_format = {config.output.default_format}")
    print(f"  default_mode = {config.output.default_mode}")
    print(f"  timestamps = {config.output.timestamps}")
    print(f"  chapters = {config.output.chapters}")
    print()
    print("[cache]")
    print(f"  dir = {config.cache.dir}")
    print(f"  ttl_days = {config.cache.ttl_days}")


DEFAULT_CONFIG_TOML = """\
# Readtube configuration
# See: https://github.com/unbalancedparentheses/readtube

[llm]
# backend = "ollama"          # ollama | claude | claude-code | openai
# model = "llama3.2"          # model name
# api_key_env = "ANTHROPIC_API_KEY"  # env var name (never the key itself)
# url = "http://localhost:11434"     # for ollama or openai-compatible

[output]
default_format = "md"          # md | epub | pdf | html
default_mode = "article"       # article | tldr | takeaways | transcript
timestamps = false
chapters = true

[cache]
# dir = "~/.cache/readtube"
ttl_days = 30
"""


def init_config() -> Path:
    """Create default config file. Returns path."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        print(f"config already exists: {CONFIG_FILE}", file=sys.stderr)
        return CONFIG_FILE
    CONFIG_FILE.write_text(DEFAULT_CONFIG_TOML)
    print(f"created: {CONFIG_FILE}", file=sys.stderr)
    return CONFIG_FILE


def progress(msg: str, verbose: bool = True) -> None:
    """Print progress message to stderr."""
    if verbose:
        print(msg, file=sys.stderr)
