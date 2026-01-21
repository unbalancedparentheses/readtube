.PHONY: install install-pdf install-all test test-cov test-e2e test-unit lint clean help shell docker

# Default target
help:
	@echo "Readtube - Transform YouTube videos into ebooks"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install Python dependencies"
	@echo "  make install-pdf   Install dependencies including PDF support"
	@echo "  make install-all   Install all optional dependencies"
	@echo "  make shell         Enter Nix development shell (requires Nix)"
	@echo ""
	@echo "Run:"
	@echo "  make demo          Fetch a sample video transcript"
	@echo "  make docker        Build and run with Docker"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make test-unit     Run unit tests only"
	@echo "  make test-e2e      Run end-to-end tests only"
	@echo ""
	@echo "Development:"
	@echo "  make lint          Run linter"
	@echo "  make clean         Remove generated files"
	@echo ""

# Install Python dependencies
install:
	pip install -r requirements.txt
	pip install pytest pytest-mock pytest-cov

# Install with PDF support (requires system dependencies)
install-pdf: install
	pip install weasyprint
	@echo ""
	@echo "Note: PDF generation requires system libraries (Pango, Cairo, GLib)."
	@echo "On macOS: brew install pango cairo glib"
	@echo "On Ubuntu: apt install libpango-1.0-0 libcairo2"
	@echo "Or use: make shell (requires Nix)"

# Enter Nix development shell
shell:
	@if command -v nix-shell > /dev/null; then \
		nix-shell; \
	else \
		echo "Nix is not installed. Install from https://nixos.org/download.html"; \
		echo "Or install dependencies manually:"; \
		echo "  macOS: brew install pango cairo glib"; \
		echo "  Ubuntu: apt install libpango-1.0-0 libcairo2"; \
	fi

# Run all tests
test:
	pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# Run unit tests only
test-unit:
	pytest tests/test_unit.py -v

# Run end-to-end tests only (makes network requests)
test-e2e:
	pytest tests/test_e2e.py -v -s

# Lint
lint:
	@if command -v ruff > /dev/null; then \
		ruff check .; \
	elif command -v flake8 > /dev/null; then \
		flake8 . --max-line-length=127; \
	else \
		echo "No linter installed. Install with: pip install ruff"; \
	fi

# Clean generated files
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf .transcript_cache
	rm -f *.epub *.pdf
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned generated files"

# Demo: fetch a sample transcript
demo:
	python fetch_transcript.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Install all optional dependencies
install-all: install
	pip install weasyprint pyyaml tqdm
	@echo ""
	@echo "Optional TTS backends:"
	@echo "  pip install pyttsx3      # Offline TTS"
	@echo "  pip install gtts         # Google TTS"
	@echo "  pip install edge-tts     # Microsoft Edge TTS"
	@echo ""
	@echo "Optional translation:"
	@echo "  pip install googletrans==4.0.0-rc1  # Google Translate"
	@echo "  pip install deepl        # DeepL (needs API key)"

# Build and run with Docker
docker:
	docker-compose up --build
