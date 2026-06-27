"""Tests for LLM module."""

import pytest
from readtube.llm import (
    get_backend,
    BACKENDS,
    OllamaBackend,
    ClaudeBackend,
    ClaudeCodeBackend,
    OpenAIBackend,
)
from readtube.errors import LLMError


class TestBackendRegistry:
    def test_all_backends_registered(self):
        assert "ollama" in BACKENDS
        assert "claude" in BACKENDS
        assert "claude-code" in BACKENDS
        assert "openai" in BACKENDS

    def test_unknown_backend_raises(self):
        with pytest.raises(LLMError, match="unknown backend"):
            get_backend("nonexistent")


class TestOllamaBackend:
    def test_default_config(self):
        backend = OllamaBackend()
        assert backend.url == "http://localhost:11434"
        assert backend.model == "llama3.2"

    def test_custom_config(self):
        backend = OllamaBackend(url="http://myserver:11434", model="mistral")
        assert backend.url == "http://myserver:11434"
        assert backend.model == "mistral"


class TestClaudeBackend:
    def test_no_key_not_available(self):
        backend = ClaudeBackend(api_key=None)
        # Clear env
        import os
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            backend = ClaudeBackend(api_key=None)
            assert not backend.is_available()
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_with_key_needs_anthropic(self):
        backend = ClaudeBackend(api_key="test-key")
        # May or may not have anthropic installed
        # Just verify it doesn't crash
        backend.is_available()


class TestClaudeCodeBackend:
    def test_unavailable_without_claudecode_env(self, monkeypatch):
        monkeypatch.delenv("CLAUDECODE", raising=False)
        assert not ClaudeCodeBackend().is_available()

    def test_unavailable_without_binary(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setattr(ClaudeCodeBackend, "_binary", staticmethod(lambda: None))
        assert not ClaudeCodeBackend().is_available()

    def test_available_when_in_claude_code(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setattr(ClaudeCodeBackend, "_binary", staticmethod(lambda: "/usr/bin/claude"))
        assert ClaudeCodeBackend().is_available()

    def test_generate_invokes_cli(self, monkeypatch):
        import subprocess
        import types

        monkeypatch.setattr(ClaudeCodeBackend, "_binary", staticmethod(lambda: "/usr/bin/claude"))
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return types.SimpleNamespace(returncode=0, stdout="# Article\n\nbody", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        out = ClaudeCodeBackend(model="opus").generate("transcript here", system_prompt="be a writer")
        assert out == "# Article\n\nbody"
        assert captured["cmd"][:2] == ["/usr/bin/claude", "-p"]
        assert "--system-prompt" in captured["cmd"]
        assert "--model" in captured["cmd"] and "opus" in captured["cmd"]

    def test_generate_raises_on_nonzero_exit(self, monkeypatch):
        import subprocess
        import types

        monkeypatch.setattr(ClaudeCodeBackend, "_binary", staticmethod(lambda: "/usr/bin/claude"))
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda cmd, **kw: types.SimpleNamespace(returncode=1, stdout="", stderr="boom"),
        )
        with pytest.raises(LLMError, match="boom"):
            ClaudeCodeBackend().generate("x")


class TestOpenAIBackend:
    def test_local_server_check(self):
        backend = OpenAIBackend(url="http://localhost:9999")
        # Won't be available since nothing is running there
        assert not backend.is_available()

    def test_remote_with_key(self):
        backend = OpenAIBackend(api_key="test-key")
        assert backend.is_available()
