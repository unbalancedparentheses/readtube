# Readtube - YouTube to Ebook converter
# Multi-stage build for smaller image

FROM python:3.11-slim as base

# Install system dependencies for WeasyPrint
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install Calibre for MOBI conversion (optional, large)
# Uncomment if you need MOBI support
# RUN apt-get update && apt-get install -y --no-install-recommends calibre

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir weasyprint pyyaml tqdm

# Copy application code
COPY *.py ./
COPY themes.py ./

# Create directories
RUN mkdir -p /app/output /app/.transcript_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/app/output

# Default command
CMD ["python", "fetch_transcript.py", "--help"]

# Usage examples:
#
# Build:
#   docker build -t readtube .
#
# Fetch transcript:
#   docker run --rm readtube python fetch_transcript.py "https://youtube.com/watch?v=VIDEO_ID"
#
# Create ebook (mount output directory):
#   docker run --rm -v $(pwd)/output:/app/output readtube python -c "
#     from create_epub import create_ebook
#     articles = [{'title': 'Test', 'channel': 'Ch', 'url': 'http://x', 'article': '# Hello'}]
#     create_ebook(articles, output_path='/app/output/test.epub')
#   "
#
# Run batch processing:
#   docker run --rm -v $(pwd)/config.yaml:/app/config.yaml -v $(pwd)/output:/app/output \
#     readtube python batch.py /app/config.yaml --output-dir /app/output
