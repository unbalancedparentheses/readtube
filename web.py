#!/usr/bin/env python3
"""
Simple web UI for Readtube.
Provides a dashboard to manage video-to-ebook conversions.

Usage:
    python web.py
    python web.py --port 8080
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    from flask import Flask, render_template_string, request, jsonify, send_file
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from get_videos import get_video_info, is_playlist_url
from get_transcripts import get_transcript, list_available_languages
from create_epub import create_ebook
from config import get_config, logger

app = Flask(__name__)

# Store jobs in memory (in production, use a database)
jobs: Dict[str, Dict[str, Any]] = {}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Readtube</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 2rem;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
            color: #111;
        }
        .subtitle {
            color: #666;
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .card h2 {
            font-size: 1.2rem;
            margin-bottom: 1rem;
            color: #333;
        }
        .form-group { margin-bottom: 1rem; }
        label {
            display: block;
            font-weight: 500;
            margin-bottom: 0.3rem;
            color: #555;
        }
        input[type="text"], input[type="url"], select, textarea {
            width: 100%;
            padding: 0.7rem;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 1rem;
        }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #0066cc;
        }
        .btn {
            display: inline-block;
            padding: 0.7rem 1.5rem;
            background: #0066cc;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 1rem;
            cursor: pointer;
            text-decoration: none;
        }
        .btn:hover { background: #0052a3; }
        .btn-secondary {
            background: #666;
        }
        .btn-secondary:hover { background: #555; }
        .job-list { list-style: none; }
        .job-item {
            padding: 1rem;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .job-item:last-child { border-bottom: none; }
        .job-title { font-weight: 500; }
        .job-meta { font-size: 0.85rem; color: #666; }
        .status {
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .status-pending { background: #fff3cd; color: #856404; }
        .status-processing { background: #cce5ff; color: #004085; }
        .status-done { background: #d4edda; color: #155724; }
        .status-error { background: #f8d7da; color: #721c24; }
        .error { color: #dc3545; margin-top: 0.5rem; }
        .success { color: #28a745; margin-top: 0.5rem; }
        .row { display: flex; gap: 1rem; }
        .col { flex: 1; }
        @media (max-width: 600px) {
            .row { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Readtube</h1>
        <p class="subtitle">Transform YouTube videos into beautifully typeset ebooks</p>

        <div class="card">
            <h2>New Conversion</h2>
            <form id="convertForm">
                <div class="form-group">
                    <label for="url">YouTube URL</label>
                    <input type="url" id="url" name="url" placeholder="https://www.youtube.com/watch?v=..." required>
                </div>
                <div class="row">
                    <div class="col form-group">
                        <label for="format">Output Format</label>
                        <select id="format" name="format">
                            <option value="epub">EPUB</option>
                            <option value="pdf">PDF</option>
                            <option value="html">HTML</option>
                        </select>
                    </div>
                    <div class="col form-group">
                        <label for="language">Language</label>
                        <select id="language" name="language">
                            <option value="">Auto-detect</option>
                            <option value="en">English</option>
                            <option value="es">Spanish</option>
                            <option value="de">German</option>
                            <option value="fr">French</option>
                            <option value="pt">Portuguese</option>
                            <option value="ja">Japanese</option>
                            <option value="ko">Korean</option>
                            <option value="zh">Chinese</option>
                        </select>
                    </div>
                </div>
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="summary" name="summary">
                        Summary mode (short version)
                    </label>
                </div>
                <button type="submit" class="btn">Fetch Video</button>
                <div id="formMessage"></div>
            </form>
        </div>

        <div class="card">
            <h2>Recent Jobs</h2>
            <ul class="job-list" id="jobList">
                <li class="job-item">
                    <div>
                        <div class="job-meta">No jobs yet. Enter a YouTube URL above to get started.</div>
                    </div>
                </li>
            </ul>
        </div>

        <div class="card">
            <h2>About</h2>
            <p>Readtube extracts transcripts from YouTube videos and transforms them into magazine-style articles.</p>
            <p style="margin-top: 0.5rem;">
                <a href="https://github.com/unbalancedparentheses/readtube" target="_blank">GitHub</a> ·
                Typography based on <a href="https://practicaltypography.com/" target="_blank">Practical Typography</a>
            </p>
        </div>
    </div>

    <script>
        const form = document.getElementById('convertForm');
        const jobList = document.getElementById('jobList');
        const formMessage = document.getElementById('formMessage');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            formMessage.innerHTML = '<span style="color: #666;">Fetching video info...</span>';

            const formData = new FormData(form);
            const data = {
                url: formData.get('url'),
                format: formData.get('format'),
                language: formData.get('language') || null,
                summary: formData.get('summary') === 'on'
            };

            try {
                const response = await fetch('/api/fetch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                if (result.error) {
                    formMessage.innerHTML = `<span class="error">${result.error}</span>`;
                } else {
                    formMessage.innerHTML = `<span class="success">Video fetched! ${result.word_count} words.</span>`;
                    form.reset();
                    loadJobs();
                }
            } catch (error) {
                formMessage.innerHTML = `<span class="error">Error: ${error.message}</span>`;
            }
        });

        async function loadJobs() {
            try {
                const response = await fetch('/api/jobs');
                const jobs = await response.json();

                if (jobs.length === 0) {
                    jobList.innerHTML = `
                        <li class="job-item">
                            <div class="job-meta">No jobs yet.</div>
                        </li>
                    `;
                    return;
                }

                jobList.innerHTML = jobs.map(job => `
                    <li class="job-item">
                        <div>
                            <div class="job-title">${job.title}</div>
                            <div class="job-meta">${job.channel} · ${job.word_count} words · ${job.reading_time} min read</div>
                        </div>
                        <span class="status status-${job.status}">${job.status}</span>
                    </li>
                `).join('');
            } catch (error) {
                console.error('Error loading jobs:', error);
            }
        }

        // Load jobs on page load
        loadJobs();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/fetch', methods=['POST'])
def api_fetch():
    """Fetch video info and transcript."""
    data = request.json
    url = data.get('url')
    language = data.get('language')
    output_format = data.get('format', 'epub')
    summary_mode = data.get('summary', False)

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        # Fetch video info
        video = get_video_info(url)
        if not video:
            return jsonify({'error': 'Could not fetch video info'}), 400

        # Fetch transcript
        transcript = get_transcript(video['video_id'], lang=language)
        if not transcript:
            return jsonify({'error': 'No transcript available'}), 400

        word_count = len(transcript.split())
        reading_time = max(1, round(word_count / 200))

        # Create job
        job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        jobs[job_id] = {
            'id': job_id,
            'title': video['title'],
            'channel': video['channel'],
            'url': video['url'],
            'thumbnail': video.get('thumbnail'),
            'transcript': transcript,
            'word_count': word_count,
            'reading_time': reading_time,
            'format': output_format,
            'summary_mode': summary_mode,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
        }

        return jsonify({
            'job_id': job_id,
            'title': video['title'],
            'channel': video['channel'],
            'word_count': word_count,
            'reading_time': reading_time,
        })

    except Exception as e:
        logger.error(f"Error fetching video: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/jobs')
def api_jobs():
    """List all jobs."""
    return jsonify([
        {
            'id': job['id'],
            'title': job['title'],
            'channel': job['channel'],
            'word_count': job['word_count'],
            'reading_time': job['reading_time'],
            'status': job['status'],
        }
        for job in sorted(jobs.values(), key=lambda x: x['created_at'], reverse=True)
    ])


@app.route('/api/jobs/<job_id>')
def api_job(job_id):
    """Get job details."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)


def main():
    import argparse

    if not HAS_FLASK:
        print("Flask is required for the web UI. Install with: pip install flask")
        return

    parser = argparse.ArgumentParser(description="Readtube Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print(f"Starting Readtube web UI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
