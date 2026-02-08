#!/usr/bin/env python3
"""
LLM support for Readtube.
Supports multiple backends for article generation.

Backends:
- claude-code: Use within Claude Code (no API key needed)
- llama-cpp: Local llama.cpp with GGUF models (no server needed)
- ollama: Local Ollama server
- claude-api: Anthropic Claude API (requires ANTHROPIC_API_KEY)
- openai: OpenAI-compatible APIs

Usage:
    from llm import generate_article

    # Auto-select best available backend
    article = generate_article(transcript, title, channel)

    # Use specific backend
    article = generate_article(transcript, title, channel, backend="llama-cpp", model_path="./model.gguf")
"""

import os
import json
import time
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path

from config import logger


def _retry_sleep(attempt: int, base_delay: float = 0.5) -> None:
    time.sleep(base_delay * (2 ** attempt))


def _request_with_retries(func, attempts: int = 3):
    last_exc = None
    for attempt in range(attempts):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            logger.warning(f"LLM request failed (attempt {attempt + 1}/{attempts}): {exc}")
            if attempt < attempts - 1:
                _retry_sleep(attempt)
    if last_exc:
        raise last_exc


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class AICodeEnvironmentBackend(LLMBackend):
    """
    Backend for use within AI coding environments (Claude Code, OpenAI Codex, etc.).
    This is a placeholder - when running in these environments, the AI handles generation directly.
    """

    def is_available(self) -> bool:
        # Check if we're running inside Claude Code
        if os.environ.get("CLAUDE_CODE") == "1" or os.environ.get("ANTHROPIC_CLAUDE_CODE") == "1":
            return True
        # Check if we're running inside OpenAI Codex/ChatGPT
        if os.environ.get("OPENAI_CODEX") == "1" or os.environ.get("CODEX_SANDBOX") == "1":
            return True
        # Check for Jupyter/IPython environment (common in Codex)
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                # Could be Codex or regular Jupyter
                return os.environ.get("OPENAI_API_KEY") is not None
        except:
            pass
        return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        # When running in AI code environment, the AI handles generation
        logger.info("Running in AI coding environment - generation handled by host AI")
        return None


class LlamaCppBackend(LLMBackend):
    """
    llama.cpp backend using llama-cpp-python.
    Loads GGUF models directly, no server needed.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,  # -1 = use all GPU layers
    ):
        self.model_path = model_path or os.environ.get("LLAMA_MODEL_PATH")
        self.n_ctx = int(os.environ.get("LLAMA_CTX", n_ctx))
        self.n_gpu_layers = int(os.environ.get("LLAMA_GPU_LAYERS", n_gpu_layers))
        self._llm = None

    def is_available(self) -> bool:
        try:
            from llama_cpp import Llama
            if self.model_path and Path(self.model_path).exists():
                return True
            # Check common model locations
            common_paths = [
                Path.home() / ".cache" / "llama.cpp" / "models",
                Path.home() / "models",
                Path("./models"),
            ]
            for path in common_paths:
                if path.exists() and any(path.glob("*.gguf")):
                    return True
            return False
        except ImportError:
            return False

    def _get_model_path(self) -> Optional[str]:
        if self.model_path and Path(self.model_path).exists():
            return self.model_path

        # Search common locations
        common_paths = [
            Path.home() / ".cache" / "llama.cpp" / "models",
            Path.home() / "models",
            Path("./models"),
        ]
        for path in common_paths:
            if path.exists():
                gguf_files = list(path.glob("*.gguf"))
                if gguf_files:
                    # Prefer larger models
                    return str(sorted(gguf_files, key=lambda x: x.stat().st_size, reverse=True)[0])
        return None

    def _load_model(self):
        if self._llm is not None:
            return

        from llama_cpp import Llama

        model_path = self._get_model_path()
        if not model_path:
            raise ValueError("No GGUF model found. Set LLAMA_MODEL_PATH or place models in ~/models/")

        logger.info(f"Loading model: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        try:
            self._load_model()

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            if max_tokens is None:
                max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "2048"))
            if temperature is None:
                temperature = float(os.environ.get("LLM_TEMPERATURE", "0.7"))

            response = self._llm(
                full_prompt,
                max_tokens=max_tokens,
                stop=["</s>", "<|im_end|>", "<|end|>"],
                temperature=temperature,
                echo=False,
            )

            return response["choices"][0]["text"].strip()

        except Exception as e:
            logger.error(f"llama.cpp error: {e}")
            return None


class OllamaBackend(LLMBackend):
    """Ollama local LLM backend."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", base_url).rstrip("/")
        self.model = os.environ.get("OLLAMA_MODEL", model)

    def is_available(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        try:
            import urllib.request

            if max_tokens is None:
                max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "2048"))
            if temperature is None:
                temperature = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
            num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "4096"))

            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    "num_predict": max_tokens,
                },
            }
            if system_prompt:
                data["system"] = system_prompt

            def _do_request():
                req = urllib.request.Request(
                    f"{self.base_url}/api/generate",
                    data=json.dumps(data).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    result = json.load(resp)
                    return result.get("response")

            return _request_with_retries(_do_request, attempts=3)

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return None


class ClaudeAPIBackend(LLMBackend):
    """Anthropic Claude API backend."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = os.environ.get("ANTHROPIC_MODEL", model)

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic
            return True
        except ImportError:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self.api_key)

            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens or int(os.environ.get("LLM_MAX_TOKENS", "2048")),
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            if temperature is not None:
                kwargs["temperature"] = temperature

            def _do_request():
                response = client.messages.create(**kwargs)
                return response.content[0].text

            return _request_with_retries(_do_request, attempts=3)

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return None


class OpenAIBackend(LLMBackend):
    """OpenAI-compatible API backend (works with LM Studio, vLLM, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", base_url).rstrip("/")
        self.model = os.environ.get("OPENAI_MODEL", model)

    def is_available(self) -> bool:
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            try:
                import urllib.request
                req = urllib.request.Request(f"{self.base_url}/models")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    return resp.status == 200
            except:
                return False
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        try:
            import urllib.request

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            data = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or int(os.environ.get("LLM_MAX_TOKENS", "2048")),
                "temperature": temperature if temperature is not None else float(os.environ.get("LLM_TEMPERATURE", "0.7")),
            }

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            def _do_request():
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=json.dumps(data).encode(),
                    headers=headers,
                )
                with urllib.request.urlopen(req, timeout=300) as resp:
                    result = json.load(resp)
                    return result["choices"][0]["message"]["content"]

            return _request_with_retries(_do_request, attempts=3)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None


# Backend registry
BACKENDS: Dict[str, type] = {
    "auto": AICodeEnvironmentBackend,  # Claude Code, Codex, etc.
    "llama-cpp": LlamaCppBackend,
    "ollama": OllamaBackend,
    "claude-api": ClaudeAPIBackend,
    "openai": OpenAIBackend,
}


def get_available_backends() -> list:
    """List available LLM backends."""
    available = []
    for name, backend_class in BACKENDS.items():
        try:
            backend = backend_class()
            if backend.is_available():
                available.append(name)
        except:
            pass
    return available


def get_backend(name: Optional[str] = None, **kwargs) -> Optional[LLMBackend]:
    """Get an LLM backend by name, or auto-select the best available."""
    if name:
        if name not in BACKENDS:
            logger.error(f"Unknown backend: {name}")
            return None
        backend = BACKENDS[name](**kwargs)
        if not backend.is_available():
            logger.error(f"Backend {name} is not available")
            return None
        return backend

    # Auto-select priority: auto (AI env) > llama-cpp > ollama > claude-api > openai
    for backend_name in ["auto", "llama-cpp", "ollama", "claude-api", "openai"]:
        try:
            backend = BACKENDS[backend_name](**kwargs)
            if backend.is_available():
                logger.info(f"Auto-selected LLM backend: {backend_name}")
                return backend
        except:
            pass

    return None


# Default system prompt for article generation
ARTICLE_SYSTEM_PROMPT = """You are a skilled writer who transforms video transcripts into well-written, engaging articles.

Your task is to take a raw transcript and create a polished, magazine-style article that:
- Has a clear structure with headings and sections
- Captures the key ideas and insights
- Is easy to read and engaging
- Maintains the speaker's voice and perspective
- Uses proper formatting (markdown)

Do not include phrases like "In this video" or "The speaker says". Write as if it's an original article."""


ARTICLE_PROMPT_TEMPLATE = """Transform this transcript into a well-written article.

Title: {title}
Channel: {channel}
Description: {description}
Chapters:
{chapters}

Transcript:
{transcript}

Write a polished, magazine-style article based on this content. Use markdown formatting with headers, paragraphs, and occasional quotes from the original."""


ARTICLE_PROMPT_TEMPLATE_CHAPTERS = """Transform this transcript into a well-written article.

Title: {title}
Channel: {channel}
Description: {description}

The transcript below is organized into sections with ## headings. Preserve these headings in your article. Write each section based on the content under that heading.

Transcript:
{transcript}

Write a polished, magazine-style article based on this content. Use markdown formatting, keeping the ## section headings from the transcript. Write engaging prose for each section."""


def generate_article(
    transcript: str,
    title: str,
    channel: str,
    description: Optional[str] = None,
    chapters: Optional[str] = None,
    has_chapters: bool = False,
    backend: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **backend_kwargs,
) -> Optional[str]:
    """
    Generate a polished article from a transcript using an LLM.

    Args:
        transcript: The raw transcript text
        title: Video title
        channel: Channel name
        has_chapters: If True, transcript is pre-structured with ## headings
        backend: LLM backend to use or None for auto-select
        **backend_kwargs: Additional arguments for the backend

    Returns:
        Generated article in markdown format, or None if failed
    """
    llm = get_backend(backend, **backend_kwargs)
    if not llm:
        logger.error("No LLM backend available.")
        logger.error("Options:")
        logger.error("  1. Run inside Claude Code")
        logger.error("  2. Install llama-cpp-python and download a GGUF model")
        logger.error("  3. Run Ollama locally (ollama serve)")
        logger.error("  4. Set ANTHROPIC_API_KEY for Claude API")
        return None

    if has_chapters:
        prompt = ARTICLE_PROMPT_TEMPLATE_CHAPTERS.format(
            title=title,
            channel=channel,
            description=description or "",
            transcript=transcript[:50000],
        )
    else:
        prompt = ARTICLE_PROMPT_TEMPLATE.format(
            title=title,
            channel=channel,
            description=description or "",
            chapters=chapters or "",
            transcript=transcript[:50000],
        )

    return llm.generate(
        prompt,
        ARTICLE_SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=temperature,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM utilities for Readtube")
    parser.add_argument("--list-backends", action="store_true", help="List available backends")
    parser.add_argument("--test", help="Test a specific backend")
    parser.add_argument("--model-path", help="Path to GGUF model for llama-cpp")

    args = parser.parse_args()

    if args.list_backends:
        available = get_available_backends()
        print("Available LLM backends:")
        for name in available:
            print(f"  - {name}")
        if not available:
            print("  (none available)")
        print("\nTo enable backends:")
        print("  auto:        Run inside Claude Code or OpenAI Codex")
        print("  llama-cpp:   pip install llama-cpp-python && download a .gguf model")
        print("  Ollama:      Install from https://ollama.ai and run 'ollama serve'")
        print("  Claude API:  pip install anthropic && export ANTHROPIC_API_KEY=...")
        print("  OpenAI:      export OPENAI_API_KEY=...")

    elif args.test:
        kwargs = {}
        if args.model_path:
            kwargs["model_path"] = args.model_path

        backend = get_backend(args.test, **kwargs)
        if backend:
            print(f"Testing backend: {args.test}")
            result = backend.generate("Say 'Hello, Readtube!' in a friendly way.")
            if result:
                print(f"Success! Response: {result}")
            else:
                print("Backend failed to generate response")
        else:
            print(f"Backend {args.test} is not available")
