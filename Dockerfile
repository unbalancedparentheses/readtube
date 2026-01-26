# Readtube - YouTube to Ebook converter
# Production-ready multi-stage build

# ============================================
# Stage 1: Build dependencies
# ============================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir weasyprint pyyaml tqdm

# ============================================
# Stage 2: Production image
# ============================================
FROM python:3.11-slim as production

# Labels for container metadata
LABEL org.opencontainers.image.title="Readtube"
LABEL org.opencontainers.image.description="Transform YouTube videos into beautifully typeset ebooks"
LABEL org.opencontainers.image.source="https://github.com/unbalancedparentheses/readtube"
LABEL org.opencontainers.image.licenses="MIT"

# Install runtime dependencies for WeasyPrint and PDF generation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 readtube \
    && useradd --uid 1000 --gid readtube --shell /bin/bash --create-home readtube

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=readtube:readtube *.py ./
COPY --chown=readtube:readtube tests/ ./tests/

# Create directories with proper permissions
RUN mkdir -p /app/output /app/cache /app/data \
    && chown -R readtube:readtube /app

# Switch to non-root user
USER readtube

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OUTPUT_DIR=/app/output \
    CACHE_DIR=/app/cache \
    # Disable GPU for llama.cpp (no GPU in container)
    LLAMA_GPU_LAYERS=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import fetch_transcript; print('OK')" || exit 1

# Expose port for potential web dashboard
EXPOSE 5000

# Default entrypoint
ENTRYPOINT ["python"]
CMD ["fetch_transcript.py", "--help"]

# ============================================
# Usage Examples
# ============================================
#
# Build:
#   docker build -t readtube .
#
# Fetch transcript:
#   docker run --rm readtube fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"
#
# Fetch and save JSON:
#   docker run --rm -v $(pwd)/output:/app/output readtube \
#     fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID" --output-json /app/output/video.json
#
# Create EPUB from JSON:
#   docker run --rm -v $(pwd)/output:/app/output readtube \
#     write_article.py /app/output/video.json --format epub --output-dir /app/output
#
# Run with Ollama backend:
#   docker run --rm --network host \
#     -e OLLAMA_BASE_URL=http://localhost:11434 \
#     -v $(pwd)/output:/app/output readtube \
#     write_article.py /app/output/video.json
#
# Run with Claude API:
#   docker run --rm \
#     -e ANTHROPIC_API_KEY=your-key-here \
#     -v $(pwd)/output:/app/output readtube \
#     write_article.py /app/output/video.json
#
# Batch processing:
#   docker run --rm \
#     -v $(pwd)/config.yaml:/app/config.yaml:ro \
#     -v $(pwd)/output:/app/output \
#     readtube batch.py /app/config.yaml
