#!/usr/bin/env python3
"""
RSS/Atom feed generation for Readtube.
Generate podcast-style feeds from converted videos.

Usage:
    python rss.py --output feed.xml --title "My Reading List"
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

from config import get_config, logger


def generate_rss_feed(
    articles: List[Dict[str, Any]],
    title: str = "Readtube Feed",
    description: str = "Articles from YouTube videos",
    link: str = "https://github.com/unbalancedparentheses/readtube",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate an RSS 2.0 feed from articles.

    Args:
        articles: List of article dicts with title, channel, url, article, created_at
        title: Feed title
        description: Feed description
        link: Feed link
        output_path: Optional path to save the feed

    Returns:
        RSS XML string
    """
    # Create RSS root
    rss = ET.Element('rss', version='2.0')
    rss.set('xmlns:atom', 'http://www.w3.org/2005/Atom')
    rss.set('xmlns:content', 'http://purl.org/rss/1.0/modules/content/')

    channel = ET.SubElement(rss, 'channel')

    # Channel metadata
    ET.SubElement(channel, 'title').text = title
    ET.SubElement(channel, 'description').text = description
    ET.SubElement(channel, 'link').text = link
    ET.SubElement(channel, 'language').text = 'en-us'
    ET.SubElement(channel, 'lastBuildDate').text = datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000')
    ET.SubElement(channel, 'generator').text = 'Readtube'

    # Add items
    for article in articles:
        item = ET.SubElement(channel, 'item')

        ET.SubElement(item, 'title').text = article.get('title', 'Untitled')
        ET.SubElement(item, 'link').text = article.get('url', '')
        ET.SubElement(item, 'guid', isPermaLink='false').text = article.get('url', '')

        # Publication date
        created_at = article.get('created_at', datetime.now().isoformat())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except:
                created_at = datetime.now()
        ET.SubElement(item, 'pubDate').text = created_at.strftime('%a, %d %b %Y %H:%M:%S +0000')

        # Author (channel)
        if article.get('channel'):
            ET.SubElement(item, 'author').text = article['channel']

        # Description (first 500 chars of article)
        article_text = article.get('article', '')
        # Strip markdown formatting for description
        import re
        clean_text = re.sub(r'[#*_`\[\]()]', '', article_text)
        description_text = clean_text[:500] + '...' if len(clean_text) > 500 else clean_text
        ET.SubElement(item, 'description').text = description_text

        # Full content
        content_encoded = ET.SubElement(item, 'content:encoded')
        content_encoded.text = f"<![CDATA[{article_text}]]>"

        # Thumbnail as enclosure
        if article.get('thumbnail'):
            ET.SubElement(item, 'enclosure',
                url=article['thumbnail'],
                type='image/jpeg'
            )

    # Pretty print
    xml_string = ET.tostring(rss, encoding='unicode')
    parsed = minidom.parseString(xml_string)
    pretty_xml = parsed.toprettyxml(indent='  ')

    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        logger.info(f"RSS feed saved to {output_path}")

    return pretty_xml


def generate_atom_feed(
    articles: List[Dict[str, Any]],
    title: str = "Readtube Feed",
    subtitle: str = "Articles from YouTube videos",
    link: str = "https://github.com/unbalancedparentheses/readtube",
    author: str = "Readtube",
    output_path: Optional[str] = None,
) -> str:
    """
    Generate an Atom feed from articles.

    Args:
        articles: List of article dicts
        title: Feed title
        subtitle: Feed subtitle
        link: Feed link
        author: Feed author
        output_path: Optional path to save the feed

    Returns:
        Atom XML string
    """
    ATOM_NS = 'http://www.w3.org/2005/Atom'

    feed = ET.Element('feed', xmlns=ATOM_NS)

    ET.SubElement(feed, 'title').text = title
    ET.SubElement(feed, 'subtitle').text = subtitle
    ET.SubElement(feed, 'link', href=link)
    ET.SubElement(feed, 'id').text = link
    ET.SubElement(feed, 'updated').text = datetime.now().isoformat() + 'Z'

    author_elem = ET.SubElement(feed, 'author')
    ET.SubElement(author_elem, 'name').text = author

    ET.SubElement(feed, 'generator').text = 'Readtube'

    for article in articles:
        entry = ET.SubElement(feed, 'entry')

        ET.SubElement(entry, 'title').text = article.get('title', 'Untitled')
        ET.SubElement(entry, 'link', href=article.get('url', ''))
        ET.SubElement(entry, 'id').text = article.get('url', '')

        created_at = article.get('created_at', datetime.now().isoformat())
        if isinstance(created_at, str):
            if not created_at.endswith('Z'):
                created_at += 'Z'
        else:
            created_at = created_at.isoformat() + 'Z'
        ET.SubElement(entry, 'updated').text = created_at

        if article.get('channel'):
            author_elem = ET.SubElement(entry, 'author')
            ET.SubElement(author_elem, 'name').text = article['channel']

        # Summary
        article_text = article.get('article', '')
        import re
        clean_text = re.sub(r'[#*_`\[\]()]', '', article_text)
        summary_text = clean_text[:500] + '...' if len(clean_text) > 500 else clean_text
        ET.SubElement(entry, 'summary').text = summary_text

        # Full content
        content = ET.SubElement(entry, 'content', type='html')
        content.text = article_text

    # Pretty print
    xml_string = ET.tostring(feed, encoding='unicode')
    parsed = minidom.parseString(xml_string)
    pretty_xml = parsed.toprettyxml(indent='  ')

    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        logger.info(f"Atom feed saved to {output_path}")

    return pretty_xml


def main():
    parser = argparse.ArgumentParser(description="Generate RSS/Atom feeds")
    parser.add_argument("--output", "-o", default="feed.xml", help="Output file path")
    parser.add_argument("--title", default="Readtube Feed", help="Feed title")
    parser.add_argument("--format", choices=["rss", "atom"], default="rss", help="Feed format")

    args = parser.parse_args()

    # Example usage
    articles = [
        {
            "title": "Example Article",
            "channel": "Example Channel",
            "url": "https://youtube.com/watch?v=example",
            "article": "# Example\n\nThis is an example article.",
            "created_at": datetime.now().isoformat(),
        }
    ]

    if args.format == "atom":
        generate_atom_feed(articles, title=args.title, output_path=args.output)
    else:
        generate_rss_feed(articles, title=args.title, output_path=args.output)


if __name__ == "__main__":
    main()
