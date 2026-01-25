#!/usr/bin/env python3
"""
Create EPUB or PDF ebooks from article content.
Supports thumbnail covers and professional typesetting.

Usage:
    python create_epub.py article.md --title "Title" --channel "Channel" --url "URL"
    python create_epub.py article.md --title "Title" --channel "Channel" --url "URL" --pdf

Or import and use directly:
    from create_epub import create_ebook
    create_ebook(articles, format="epub")  # or format="pdf"
"""

import os
import sys
import argparse
import markdown
import tempfile
import urllib.request
from datetime import datetime
from ebooklib import epub


# Professional typesetting CSS based on Practical Typography
# https://practicaltypography.com/
#
# Key principles applied:
# - Line spacing: 120-145% of point size (using 1.4)
# - Line length: 45-90 characters (max-width ~65ch)
# - First-line indents OR paragraph spacing, not both
# - Headings: subtle size increases, space above/below
# - Serif fonts for body text (Charter, Georgia)
# - Smart quotes handled by markdown 'smarty' extension
TYPOGRAPHY_CSS = """
/* Base typography - Practical Typography principles */
body {
    font-family: Charter, "Bitstream Charter", Georgia, "Times New Roman", serif;
    font-size: 1.1em;
    line-height: 1.4;  /* 140% - within 120-145% optimal range */
    color: #1a1a1a;
    max-width: 65ch;  /* ~65 characters per line - optimal readability */
    margin: 0 auto;
    padding: 2em 1.5em;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    font-kerning: normal;  /* Always enable kerning */
}

/* Paragraphs - use first-line indent, not space between */
p {
    margin-top: 0;
    margin-bottom: 0;
    text-align: left;
    hyphens: auto;
    -webkit-hyphens: auto;
    text-indent: 1.5em;  /* 1-4x point size recommended */
}

/* First paragraph after heading - no indent (it's already obvious) */
h1 + p, h2 + p, h3 + p, h4 + p,
.intro + p,
blockquote + p,
ul + p, ol + p,
pre + p {
    text-indent: 0;
}

/* Headings - subtle size increases, emphasis via space */
h1, h2, h3, h4 {
    font-family: Charter, "Bitstream Charter", Georgia, serif;
    font-weight: 600;
    line-height: 1.25;
    margin-bottom: 0.5em;
    color: #111;
    hyphens: none;  /* No hyphenation in headings */
    -webkit-hyphens: none;
}

h1 {
    font-size: 1.5em;  /* Modest increase from body */
    margin-top: 0;
    padding-bottom: 0.3em;
}

h2 {
    font-size: 1.2em;  /* Only slightly larger than body */
    margin-top: 2em;   /* Space above for emphasis */
}

h3 {
    font-size: 1.05em;  /* Nearly body size, bold provides emphasis */
    margin-top: 1.5em;
}

h4 {
    font-size: 1em;  /* Same as body, bold only */
    margin-top: 1.2em;
}

/* Block quotes - slight indent, no border for cleaner look */
blockquote {
    margin: 1.2em 0;
    margin-left: 1.5em;
    margin-right: 1.5em;
    font-style: italic;
    color: #333;
}

blockquote p {
    text-indent: 0;
    margin-bottom: 0.5em;
}

blockquote p:last-child {
    margin-bottom: 0;
}

/* Lists - proper spacing */
ul, ol {
    margin: 0.8em 0;
    padding-left: 1.5em;
}

li {
    margin-bottom: 0.3em;
    text-indent: 0;
}

li p {
    text-indent: 0;
}

/* Links - subtle underline */
a {
    color: inherit;
    text-decoration: underline;
    text-decoration-color: #999;
    text-underline-offset: 2px;
}

a:hover {
    text-decoration-color: #333;
}

/* Emphasis - use sparingly */
em {
    font-style: italic;
}

strong {
    font-weight: 600;
}

/* Article intro box - clean, minimal */
.intro {
    background: #f9f9f7;
    padding: 1em 1.2em;
    margin-bottom: 1.5em;
    font-size: 0.92em;
    line-height: 1.4;
    border-radius: 2px;
}

.intro p {
    margin: 0;
    text-indent: 0;
}

/* Watch link at bottom */
.watch-link {
    margin-top: 2em;
    padding: 0.8em 1em;
    background: #f5f5f3;
    font-size: 0.88em;
    border-radius: 2px;
}

.watch-link p {
    text-indent: 0;
    margin: 0;
}

/* Code blocks */
code {
    font-family: "SF Mono", Menlo, Monaco, "Consolas", monospace;
    font-size: 0.85em;
    background: #f5f5f5;
    padding: 0.1em 0.25em;
    border-radius: 2px;
}

pre {
    background: #f5f5f5;
    padding: 1em;
    overflow-x: auto;
    border-radius: 2px;
    font-size: 0.85em;
    line-height: 1.4;
}

pre code {
    background: none;
    padding: 0;
}

/* Horizontal rule - subtle */
hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 1.5em 0;
}

/* Small caps for abbreviations - 5-12% extra letterspacing */
.sc {
    font-variant: small-caps;
    letter-spacing: 0.05em;
}

/* Prevent orphans and widows where supported */
p {
    orphans: 2;
    widows: 2;
}
"""


def download_thumbnail(url, output_path):
    """Download thumbnail image from URL."""
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  Could not download thumbnail: {e}")
        return False


def create_epub_book(articles, output_path=None, include_cover=True):
    """
    Create an EPUB ebook from a list of articles.

    Args:
        articles: List of dicts with keys: title, channel, url, article, thumbnail (optional)
        output_path: Optional custom output path
        include_cover: Whether to include thumbnail as cover

    Returns:
        Path to the created EPUB file
    """
    today = datetime.now().strftime("%B %d, %Y")

    if output_path is None:
        filename = f"youtube_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.epub"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    # Create the ebook
    book = epub.EpubBook()
    book.set_identifier(f"youtube-digest-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    book.set_title(f"YouTube Digest - {today}")
    book.set_language("en")
    book.add_author("Readtube")

    # Add CSS
    nav_css = epub.EpubItem(
        uid="style_nav",
        file_name="style/typography.css",
        media_type="text/css",
        content=TYPOGRAPHY_CSS
    )
    book.add_item(nav_css)

    chapters = []

    # Try to add cover from first article's thumbnail
    if include_cover and articles and articles[0].get("thumbnail"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                if download_thumbnail(articles[0]["thumbnail"], tmp.name):
                    with open(tmp.name, "rb") as img_file:
                        book.set_cover("cover.jpg", img_file.read())
                    print(f"  Added cover image from thumbnail")
                os.unlink(tmp.name)
        except Exception as e:
            print(f"  Could not add cover: {e}")

    for i, article in enumerate(articles):
        # Convert markdown to HTML
        article_html = markdown.markdown(
            article['article'],
            extensions=['smarty']  # Smart quotes, dashes
        )

        # Build chapter content - simpler HTML for better compatibility
        chapter_content = f"""<html>
<head>
<title>{article['title'][:50]}</title>
<link rel="stylesheet" type="text/css" href="style/typography.css"/>
</head>
<body>
<div class="intro">
<p>Based on <strong>{article['title']}</strong> from <strong>{article['channel']}</strong></p>
</div>
{article_html}
<p class="watch-link">Original video: {article['url']}</p>
</body>
</html>"""

        # Create chapter title (truncate if too long)
        chapter_title = article['title'][:50]
        if len(article['title']) > 50:
            chapter_title += "..."

        chapter = epub.EpubHtml(
            title=chapter_title,
            file_name=f"chapter_{i+1}.xhtml",
            lang="en"
        )
        chapter.content = chapter_content
        chapter.add_item(nav_css)

        book.add_item(chapter)
        chapters.append(chapter)

    # Create table of contents
    book.toc = tuple(chapters)

    # Add navigation files
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Set the reading order
    book.spine = ["nav"] + chapters

    # Write the EPUB file
    epub.write_epub(output_path, book)

    print(f"EPUB created: {output_path}")
    return output_path


def create_pdf(articles, output_path=None):
    """
    Create a PDF from a list of articles using weasyprint.

    Args:
        articles: List of dicts with keys: title, channel, url, article, thumbnail (optional)
        output_path: Optional custom output path

    Returns:
        Path to the created PDF file
    """
    try:
        from weasyprint import HTML, CSS
    except ImportError:
        print("Error: weasyprint not installed. Install with: pip install weasyprint")
        print("Note: weasyprint requires system dependencies. See: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html")
        return None

    today = datetime.now().strftime("%B %d, %Y")

    if output_path is None:
        filename = f"youtube_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    # Build HTML document
    html_parts = [f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>YouTube Digest - {today}</title>
    <style>
        {TYPOGRAPHY_CSS}

        /* PDF-specific adjustments */
        @page {{
            size: A5;
            margin: 2cm 1.5cm;
        }}

        @page:first {{
            margin: 0;
        }}

        body {{
            font-size: 10pt;
            max-width: none;
            padding: 0;
        }}

        /* Title page styles */
        .title-page {{
            page-break-after: always;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            text-align: center;
            padding: 2cm;
            box-sizing: border-box;
        }}

        .title-page .cover-image {{
            max-width: 100%;
            max-height: 50vh;
            margin-bottom: 2em;
            border-radius: 4px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .title-page h1 {{
            font-size: 18pt;
            margin-bottom: 0.5em;
            line-height: 1.3;
        }}

        .title-page .channel {{
            font-size: 12pt;
            color: #555;
            margin-bottom: 1em;
        }}

        .title-page .date {{
            font-size: 10pt;
            color: #777;
        }}

        .article {{
            page-break-after: always;
        }}

        .article:last-child {{
            page-break-after: auto;
        }}

        h1 {{
            font-size: 16pt;
        }}

        h2 {{
            font-size: 13pt;
        }}
    </style>
</head>
<body>
"""]

    for i, article in enumerate(articles):
        # Add title page with thumbnail for each article
        thumbnail_html = ""
        if article.get('thumbnail'):
            thumbnail_html = f'<img class="cover-image" src="{article["thumbnail"]}" alt="Video thumbnail">'

        html_parts.append(f"""
    <div class="title-page">
        {thumbnail_html}
        <h1>{article['title']}</h1>
        <p class="channel">{article['channel']}</p>
        <p class="date">{today}</p>
    </div>
""")

        article_html = markdown.markdown(
            article['article'],
            extensions=['smarty']
        )

        html_parts.append(f"""
    <div class="article">
        {article_html}
        <p class="watch-link">Original video: {article['url']}</p>
    </div>
""")

    html_parts.append("</body></html>")

    full_html = "".join(html_parts)

    # Generate PDF
    HTML(string=full_html).write_pdf(output_path)

    print(f"PDF created: {output_path}")
    return output_path


def create_html(articles, output_path=None):
    """
    Create a standalone HTML file from a list of articles.

    Args:
        articles: List of dicts with keys: title, channel, url, article
        output_path: Optional custom output path

    Returns:
        Path to the created HTML file
    """
    today = datetime.now().strftime("%B %d, %Y")

    if output_path is None:
        filename = f"youtube_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Digest - {today}</title>
    <style>
        {TYPOGRAPHY_CSS}

        /* HTML-specific styles */
        body {{
            background: #fafaf8;
        }}

        .container {{
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 2em auto;
            padding: 3em 2em;
        }}

        .header {{
            text-align: center;
            margin-bottom: 3em;
            padding-bottom: 1.5em;
            border-bottom: 1px solid #ddd;
        }}

        .header h1 {{
            font-size: 1.5em;
            margin-bottom: 0.3em;
        }}

        .header .date {{
            color: #666;
            font-size: 0.9em;
        }}

        .article {{
            margin-bottom: 4em;
            padding-bottom: 3em;
            border-bottom: 1px solid #eee;
        }}

        .article:last-child {{
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>YouTube Digest</h1>
            <p class="date">{today}</p>
        </div>
"""]

    for article in articles:
        article_html = markdown.markdown(
            article['article'],
            extensions=['smarty']
        )

        html_parts.append(f"""
        <div class="article">
            <div class="intro">
                <p>Based on <strong>{article['title']}</strong> from <strong>{article['channel']}</strong></p>
            </div>
            {article_html}
            <p class="watch-link">Original video: <a href="{article['url']}">{article['url']}</a></p>
        </div>
""")

    html_parts.append("""
    </div>
</body>
</html>""")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"HTML created: {output_path}")
    return output_path


def create_mobi(articles, output_path=None):
    """
    Create a MOBI ebook (Kindle format).

    Requires either Calibre's ebook-convert or Amazon's kindlegen to be installed.

    Args:
        articles: List of dicts with keys: title, channel, url, article
        output_path: Optional custom output path

    Returns:
        Path to the created MOBI file or None if failed
    """
    import shutil
    import subprocess
    import tempfile

    # First create an EPUB
    today = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"youtube_digest_{today}.mobi")

    # Create temp EPUB
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        epub_path = tmp.name

    epub_result = create_epub_book(articles, epub_path)
    if not epub_result:
        return None

    try:
        # Try Calibre's ebook-convert first
        if shutil.which('ebook-convert'):
            result = subprocess.run([
                'ebook-convert', epub_path, output_path
            ], capture_output=True, timeout=120)

            if result.returncode == 0:
                print(f"MOBI created: {output_path}")
                return output_path
            else:
                print(f"  ebook-convert failed: {result.stderr.decode()}")

        # Try kindlegen
        if shutil.which('kindlegen'):
            result = subprocess.run([
                'kindlegen', epub_path, '-o', os.path.basename(output_path)
            ], capture_output=True, timeout=120)

            if result.returncode in (0, 1):  # kindlegen returns 1 for warnings
                # kindlegen outputs to same directory as input
                temp_mobi = epub_path.replace('.epub', '.mobi')
                if os.path.exists(temp_mobi):
                    shutil.move(temp_mobi, output_path)
                    print(f"MOBI created: {output_path}")
                    return output_path

        print("Error: No MOBI converter found. Install Calibre or kindlegen.")
        print("  Calibre: https://calibre-ebook.com/")
        print("  kindlegen: Download from Amazon (deprecated but still works)")
        return None

    except subprocess.TimeoutExpired:
        print("  MOBI conversion timed out")
        return None
    except Exception as e:
        print(f"  MOBI conversion failed: {e}")
        return None
    finally:
        # Clean up temp EPUB
        if os.path.exists(epub_path):
            os.unlink(epub_path)


def create_azw3(articles, output_path=None):
    """
    Create an AZW3 ebook (Kindle Format 8).

    Requires Calibre's ebook-convert to be installed.

    Args:
        articles: List of dicts with keys: title, channel, url, article
        output_path: Optional custom output path

    Returns:
        Path to the created AZW3 file or None if failed
    """
    import shutil
    import subprocess
    import tempfile

    if not shutil.which('ebook-convert'):
        print("Error: Calibre's ebook-convert required for AZW3. Install Calibre.")
        return None

    today = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"youtube_digest_{today}.azw3")

    # Create temp EPUB
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        epub_path = tmp.name

    epub_result = create_epub_book(articles, epub_path)
    if not epub_result:
        return None

    try:
        result = subprocess.run([
            'ebook-convert', epub_path, output_path
        ], capture_output=True, timeout=120)

        if result.returncode == 0:
            print(f"AZW3 created: {output_path}")
            return output_path
        else:
            print(f"  AZW3 conversion failed: {result.stderr.decode()}")
            return None

    except Exception as e:
        print(f"  AZW3 conversion failed: {e}")
        return None
    finally:
        if os.path.exists(epub_path):
            os.unlink(epub_path)


def create_ebook(articles, output_path=None, format="epub"):
    """
    Create an ebook in the specified format.

    Args:
        articles: List of article dicts
        output_path: Optional output path
        format: "epub", "pdf", "html", "mobi", or "azw3"

    Returns:
        Path to created file
    """
    if format == "pdf":
        return create_pdf(articles, output_path)
    elif format == "html":
        return create_html(articles, output_path)
    elif format == "mobi":
        return create_mobi(articles, output_path)
    elif format == "azw3":
        return create_azw3(articles, output_path)
    else:
        return create_epub_book(articles, output_path)


# Alias for backwards compatibility
create_epub_from_articles = create_epub_book


def main():
    parser = argparse.ArgumentParser(description="Create ebook from markdown article")
    parser.add_argument("article_file", help="Path to markdown article file")
    parser.add_argument("--title", required=True, help="Video title")
    parser.add_argument("--channel", required=True, help="Channel name")
    parser.add_argument("--url", required=True, help="Video URL")
    parser.add_argument("--thumbnail", help="Thumbnail URL for cover")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--output-dir", help="Output directory for generated files")
    parser.add_argument("--pdf", action="store_true", help="Output as PDF instead of EPUB")
    parser.add_argument("--html", action="store_true", help="Output as HTML instead of EPUB")
    parser.add_argument("--mobi", action="store_true", help="Output as MOBI (Kindle) instead of EPUB")
    parser.add_argument("--azw3", action="store_true", help="Output as AZW3 (Kindle Format 8) instead of EPUB")

    args = parser.parse_args()

    # Read article content
    with open(args.article_file, 'r') as f:
        article_content = f.read()

    articles = [{
        "title": args.title,
        "channel": args.channel,
        "url": args.url,
        "article": article_content,
        "thumbnail": args.thumbnail,
    }]

    if args.pdf:
        output_format = "pdf"
    elif args.html:
        output_format = "html"
    elif args.mobi:
        output_format = "mobi"
    elif args.azw3:
        output_format = "azw3"
    else:
        output_format = "epub"

    # Determine output path
    output_path = args.output
    if not output_path and args.output_dir:
        filename = f"youtube_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
        output_path = os.path.join(args.output_dir, filename)

    output_path = create_ebook(articles, output_path, format=output_format)
    if output_path:
        print(f"Done! Open {output_path}")


if __name__ == "__main__":
    main()
