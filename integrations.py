#!/usr/bin/env python3
"""
Third-party integrations for Readtube.
- Readwise
- Pocket (future)
- Instapaper (future)
"""

import os
import json
import requests
from typing import Optional, List, Dict, Any
from datetime import datetime

from config import logger


class ReadwiseClient:
    """
    Readwise API client.
    https://readwise.io/api_deets

    Requires READWISE_TOKEN environment variable or token parameter.
    """

    API_BASE = "https://readwise.io/api/v2"

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("READWISE_TOKEN")
        if not self.token:
            raise ValueError(
                "Readwise token required. Set READWISE_TOKEN environment variable "
                "or pass token parameter. Get your token at https://readwise.io/access_token"
            )

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated API request."""
        url = f"{self.API_BASE}/{endpoint}"
        headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()

        if response.content:
            return response.json()
        return {}

    def create_highlight(
        self,
        text: str,
        title: str,
        author: Optional[str] = None,
        source_url: Optional[str] = None,
        source_type: str = "article",
        category: str = "articles",
        note: Optional[str] = None,
        highlighted_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a highlight in Readwise.

        Args:
            text: The highlight text
            title: Source title
            author: Source author
            source_url: URL of the source
            source_type: Type of source
            category: Category (articles, books, tweets, podcasts)
            note: Optional note
            highlighted_at: ISO timestamp
        """
        highlight = {
            "text": text,
            "title": title,
            "source_type": source_type,
            "category": category,
        }

        if author:
            highlight["author"] = author
        if source_url:
            highlight["source_url"] = source_url
        if note:
            highlight["note"] = note
        if highlighted_at:
            highlight["highlighted_at"] = highlighted_at

        data = {"highlights": [highlight]}

        return self._request("POST", "highlights/", json=data)

    def send_article(
        self,
        title: str,
        content: str,
        url: str,
        author: Optional[str] = None,
        image_url: Optional[str] = None,
        published_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send article to Readwise Reader.

        Note: This requires Readwise Reader access.
        """
        data = {
            "url": url,
            "title": title,
            "html": content,
        }

        if author:
            data["author"] = author
        if image_url:
            data["image_url"] = image_url
        if published_date:
            data["published_date"] = published_date

        # Reader API uses different endpoint
        url = "https://readwise.io/api/v3/save/"
        headers = {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json",
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        return response.json()

    def get_highlights(
        self,
        page_size: int = 100,
        updated_after: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all highlights."""
        params = {"page_size": page_size}
        if updated_after:
            params["updated__gt"] = updated_after

        result = self._request("GET", "highlights/", params=params)
        return result.get("results", [])


def send_to_readwise(
    article: Dict[str, Any],
    token: Optional[str] = None,
    extract_highlights: bool = True,
) -> bool:
    """
    Send an article to Readwise.

    Args:
        article: Article dict with title, channel, url, article content
        token: Readwise API token (or use READWISE_TOKEN env var)
        extract_highlights: Whether to extract blockquotes as highlights

    Returns:
        True if successful
    """
    try:
        client = ReadwiseClient(token=token)

        title = article.get("title", "Untitled")
        author = article.get("channel", "Unknown")
        url = article.get("url", "")
        content = article.get("article", "")
        thumbnail = article.get("thumbnail")

        # Try to send to Reader first
        try:
            import markdown
            html_content = markdown.markdown(content)
            client.send_article(
                title=title,
                content=html_content,
                url=url,
                author=author,
                image_url=thumbnail,
            )
            logger.info(f"Sent article to Readwise Reader: {title}")
        except Exception as e:
            logger.warning(f"Could not send to Reader (may need subscription): {e}")

        # Extract blockquotes as highlights
        if extract_highlights:
            import re
            blockquotes = re.findall(r'^>\s*(.+)$', content, re.MULTILINE)

            for quote in blockquotes[:10]:  # Limit to 10 highlights
                quote = quote.strip()
                if len(quote) > 20:  # Skip very short quotes
                    client.create_highlight(
                        text=quote,
                        title=title,
                        author=author,
                        source_url=url,
                        highlighted_at=datetime.now().isoformat(),
                    )

            if blockquotes:
                logger.info(f"Created {min(len(blockquotes), 10)} highlights in Readwise")

        return True

    except Exception as e:
        logger.error(f"Failed to send to Readwise: {e}")
        return False


# Placeholder for future integrations
class PocketClient:
    """Pocket integration (placeholder)."""

    def __init__(self, consumer_key: str, access_token: str):
        raise NotImplementedError("Pocket integration not yet implemented")


class InstapaperClient:
    """Instapaper integration (placeholder)."""

    def __init__(self, username: str, password: str):
        raise NotImplementedError("Instapaper integration not yet implemented")
