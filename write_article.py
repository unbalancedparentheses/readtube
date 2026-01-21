#!/usr/bin/env python3
"""
Generate articles from video transcripts using LLMs.

Usage:
    python write_article.py video.json
    python write_article.py video.json --backend llama-cpp --model ./model.gguf
    python write_article.py video.json --prompt-only
    python write_article.py video.json --format epub
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from config import logger
from llm import generate_article, get_available_backends, ARTICLE_SYSTEM_PROMPT, ARTICLE_PROMPT_TEMPLATE


def load_video_data(path: str) -> Optional[Dict[str, Any]]:
    """Load video data from JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None


def generate_prompt_file(video_data: Dict[str, Any], output_dir: str) -> str:
    """Generate a prompt file for manual use with Claude."""
    chapters = video_data.get("chapters")
    chapters_text = ""
    if isinstance(chapters, list):
        parts = []
        for ch in chapters:
            title_text = ch.get("title", "")
            start = ch.get("start_time", "")
            parts.append(f"- {title_text} ({start})")
        chapters_text = "\n".join(parts)

    prompt = ARTICLE_PROMPT_TEMPLATE.format(
        title=video_data.get('title', 'Untitled'),
        channel=video_data.get('channel', 'Unknown'),
        description=video_data.get('description', ''),
        chapters=chapters_text,
        transcript=video_data.get('transcript', '')[:50000],
    )

    content = f"SYSTEM PROMPT:\n{ARTICLE_SYSTEM_PROMPT}\n\n---\n\nUSER PROMPT:\n{prompt}"
    
    output_path = Path(output_dir) / f"{video_data.get('video_id', 'video')}_prompt.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate articles from video transcripts")
    parser.add_argument("input", help="Path to video JSON file")
    parser.add_argument("--backend", choices=["claude-code", "llama-cpp", "ollama", "claude-api", "openai"],
                        help="LLM backend to use (default: auto)")
    parser.add_argument("--model", help="Model path (llama-cpp) or model name (ollama/openai/claude)")
    parser.add_argument("--base-url", help="Base URL for ollama/openai backends")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--n-ctx", type=int, default=None, help="Context size (llama-cpp)")
    parser.add_argument("--n-gpu-layers", type=int, default=None, help="GPU layers (llama-cpp)")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--prompt-only", action="store_true", help="Generate prompt file only")
    parser.add_argument("--format", choices=["md", "epub", "pdf", "html"], default="md",
                        help="Output format")
    parser.add_argument("--list-backends", action="store_true", help="List available backends")
    
    args = parser.parse_args()
    
    if args.list_backends:
        available = get_available_backends()
        print("Available LLM backends:", ", ".join(available) if available else "(none)")
        return
    
    # Load video data
    video_data = load_video_data(args.input)
    if not video_data:
        sys.exit(1)
    
    title = video_data.get('title', 'Untitled')
    channel = video_data.get('channel', 'Unknown')
    transcript = video_data.get('transcript', '')
    description = video_data.get('description', '')
    chapters = video_data.get('chapters')
    chapters_text = ""
    if isinstance(chapters, list):
        parts = []
        for ch in chapters:
            title_text = ch.get("title", "")
            start = ch.get("start_time", "")
            parts.append(f"- {title_text} ({start})")
        chapters_text = "\n".join(parts)
    
    if not transcript:
        logger.error("No transcript found in input file")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate prompt file only
    if args.prompt_only:
        prompt_path = generate_prompt_file(video_data, args.output_dir)
        print(f"Prompt file saved to: {prompt_path}")
        print("Paste this into Claude to generate the article.")
        return
    
    # Generate article with LLM
    backend_kwargs = {}
    if args.model:
        if args.backend == "llama-cpp":
            backend_kwargs["model_path"] = args.model
        else:
            backend_kwargs["model"] = args.model
    if args.base_url:
        backend_kwargs["base_url"] = args.base_url
    if args.n_ctx is not None:
        backend_kwargs["n_ctx"] = args.n_ctx
    if args.n_gpu_layers is not None:
        backend_kwargs["n_gpu_layers"] = args.n_gpu_layers
    
    print(f"Generating article for: {title}")
    article = generate_article(
        transcript,
        title,
        channel,
        description=description,
        chapters=chapters_text,
        backend=args.backend,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        **backend_kwargs,
    )
    
    if not article:
        logger.error("Failed to generate article")
        sys.exit(1)
    
    # Save output
    video_id = video_data.get('video_id', 'video')
    
    if args.format == "md":
        output_path = output_dir / f"{video_id}_article.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(article)
        print(f"Article saved to: {output_path}")
    
    elif args.format in ["epub", "pdf", "html"]:
        from create_epub import create_ebook
        
        articles = [{
            "title": title,
            "channel": channel,
            "url": video_data.get('url', ''),
            "thumbnail": video_data.get('thumbnail'),
            "article": article,
        }]
        
        output_path = output_dir / f"{video_id}.{args.format}"
        result = create_ebook(articles, output_path=str(output_path), format=args.format)
        
        if result:
            print(f"Ebook saved to: {result}")
        else:
            logger.error("Failed to create ebook")
            sys.exit(1)


if __name__ == "__main__":
    main()
