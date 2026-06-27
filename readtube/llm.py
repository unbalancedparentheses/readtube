"""LLM backends with streaming support."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Generator, Optional

from .errors import LLMError, retry_with_backoff, LLM_RETRY_CONFIG


class LLMBackend(ABC):
    """Abstract base for LLM backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """Generate text. Returns full response."""
        ...

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Stream text chunks. Default: yield full response at once."""
        yield self.generate(prompt, system_prompt, max_tokens, temperature)

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class OllamaBackend(LLMBackend):
    def __init__(self, url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.url = os.environ.get("OLLAMA_BASE_URL", url).rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.url}/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, prompt, system_prompt=None, max_tokens=4096, temperature=0.7):
        import urllib.request

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system_prompt:
            data["system"] = system_prompt

        def _do():
            req = urllib.request.Request(
                f"{self.url}/api/generate",
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.load(resp)
                text = result.get("response")
                if not text:
                    raise LLMError("ollama", "empty response")
                return text

        return retry_with_backoff(_do, config=LLM_RETRY_CONFIG)

    def stream(self, prompt, system_prompt=None, max_tokens=4096, temperature=0.7):
        import urllib.request

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system_prompt:
            data["system"] = system_prompt

        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                if line:
                    chunk = json.loads(line)
                    text = chunk.get("response", "")
                    if text:
                        yield text
                    if chunk.get("done"):
                        break


class ClaudeCodeBackend(LLMBackend):
    """Uses the Claude Code CLI (`claude -p`) — no API key needed.

    Active when readtube runs inside a Claude Code session (``CLAUDECODE`` is
    set) and the ``claude`` binary is on PATH. This lets `readtube` borrow the
    host Claude Code session's auth, so users running it from Claude Code get
    article generation with zero configuration.
    """

    def __init__(self, model: Optional[str] = None, **kwargs):
        # None → let the CLI choose its default model.
        self.model = model

    @staticmethod
    def _binary() -> Optional[str]:
        import shutil

        return shutil.which("claude")

    def is_available(self) -> bool:
        return bool(os.environ.get("CLAUDECODE")) and self._binary() is not None

    def generate(self, prompt, system_prompt=None, max_tokens=4096, temperature=0.7):
        import subprocess

        binary = self._binary()
        if not binary:
            raise LLMError("claude-code", "claude CLI not found on PATH")

        cmd = [binary, "-p", prompt]
        if system_prompt:
            cmd += ["--system-prompt", system_prompt]
        if self.model:
            cmd += ["--model", self.model]

        def _do():
            # Drop the parent session id so the child runs as its own session.
            env = dict(os.environ)
            env.pop("CLAUDE_CODE_SESSION_ID", None)
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                raise LLMError("claude-code", "claude CLI timed out")
            except FileNotFoundError:
                raise LLMError("claude-code", "claude CLI not found on PATH")

            if proc.returncode != 0:
                msg = proc.stderr.strip() or f"claude exited with code {proc.returncode}"
                raise LLMError("claude-code", msg)

            text = proc.stdout.strip()
            if not text:
                raise LLMError("claude-code", "empty response")
            return text

        return retry_with_backoff(_do, config=LLM_RETRY_CONFIG)


class ClaudeBackend(LLMBackend):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic
            return True
        except ImportError:
            return False

    def generate(self, prompt, system_prompt=None, max_tokens=4096, temperature=0.7):
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature

        def _do():
            response = client.messages.create(**kwargs)
            return response.content[0].text

        return retry_with_backoff(_do, config=LLM_RETRY_CONFIG)

    def stream(self, prompt, system_prompt=None, max_tokens=4096, temperature=0.7):
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature

        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        api_key: Optional[str] = None,
        url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.url = os.environ.get("OPENAI_BASE_URL", url).rstrip("/")
        self.model = model

    def is_available(self) -> bool:
        if "localhost" in self.url or "127.0.0.1" in self.url:
            try:
                import urllib.request
                req = urllib.request.Request(f"{self.url}/models")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    return resp.status == 200
            except Exception:
                return False
        return bool(self.api_key)

    def generate(self, prompt, system_prompt=None, max_tokens=4096, temperature=0.7):
        import urllib.request

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        def _do():
            req = urllib.request.Request(
                f"{self.url}/chat/completions",
                data=json.dumps(data).encode(),
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.load(resp)
                return result["choices"][0]["message"]["content"]

        return retry_with_backoff(_do, config=LLM_RETRY_CONFIG)

    def stream(self, prompt, system_prompt=None, max_tokens=4096, temperature=0.7):
        import urllib.request

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.url}/chat/completions",
            data=json.dumps(data).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


BACKENDS = {
    "ollama": OllamaBackend,
    "claude": ClaudeBackend,
    "claude-code": ClaudeCodeBackend,
    "openai": OpenAIBackend,
}


def get_backend(
    backend_name: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    url: Optional[str] = None,
) -> LLMBackend:
    """Create an LLM backend instance."""
    if backend_name not in BACKENDS:
        raise LLMError(backend_name, f"unknown backend '{backend_name}'. available: {', '.join(BACKENDS)}")

    kwargs = {}
    if model:
        kwargs["model"] = model
    if api_key:
        kwargs["api_key"] = api_key
    if url:
        kwargs["url"] = url

    backend = BACKENDS[backend_name](**kwargs)

    if not backend.is_available():
        raise LLMError(backend_name, f"backend '{backend_name}' is not available")

    return backend
