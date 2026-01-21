"""
Custom themes for Readtube ebooks.
Based on Practical Typography principles.
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class Theme:
    """A visual theme for ebook output."""
    name: str
    description: str
    css: str


# Default theme - Practical Typography
THEME_DEFAULT = Theme(
    name="default",
    description="Clean serif theme based on Practical Typography",
    css="""
/* Base typography - Practical Typography principles */
body {
    font-family: Charter, "Bitstream Charter", Georgia, "Times New Roman", serif;
    font-size: 1.1em;
    line-height: 1.4;
    color: #1a1a1a;
    max-width: 65ch;
    margin: 0 auto;
    padding: 2em 1.5em;
    text-rendering: optimizeLegibility;
    -webkit-font-smoothing: antialiased;
    font-kerning: normal;
}

p {
    margin-top: 0;
    margin-bottom: 0;
    text-align: left;
    hyphens: auto;
    -webkit-hyphens: auto;
    text-indent: 1.5em;
}

h1 + p, h2 + p, h3 + p, h4 + p,
.intro + p, blockquote + p,
ul + p, ol + p, pre + p {
    text-indent: 0;
}

h1, h2, h3, h4 {
    font-family: Charter, "Bitstream Charter", Georgia, serif;
    font-weight: 600;
    line-height: 1.25;
    margin-bottom: 0.5em;
    color: #111;
    hyphens: none;
    -webkit-hyphens: none;
}

h1 { font-size: 1.5em; margin-top: 0; padding-bottom: 0.3em; }
h2 { font-size: 1.2em; margin-top: 2em; }
h3 { font-size: 1.05em; margin-top: 1.5em; }
h4 { font-size: 1em; margin-top: 1.2em; }

blockquote {
    margin: 1.2em 0;
    margin-left: 1.5em;
    margin-right: 1.5em;
    font-style: italic;
    color: #333;
}

blockquote p { text-indent: 0; margin-bottom: 0.5em; }
blockquote p:last-child { margin-bottom: 0; }

ul, ol { margin: 0.8em 0; padding-left: 1.5em; }
li { margin-bottom: 0.3em; text-indent: 0; }
li p { text-indent: 0; }

a {
    color: inherit;
    text-decoration: underline;
    text-decoration-color: #999;
    text-underline-offset: 2px;
}

em { font-style: italic; }
strong { font-weight: 600; }

.intro {
    background: #f9f9f7;
    padding: 1em 1.2em;
    margin-bottom: 1.5em;
    font-size: 0.92em;
    line-height: 1.4;
    border-radius: 2px;
}

.intro p { margin: 0; text-indent: 0; }

.watch-link {
    margin-top: 2em;
    padding: 0.8em 1em;
    background: #f5f5f3;
    font-size: 0.88em;
    border-radius: 2px;
}

.watch-link p { text-indent: 0; margin: 0; }

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

pre code { background: none; padding: 0; }

hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }

.sc { font-variant: small-caps; letter-spacing: 0.05em; }

p { orphans: 2; widows: 2; }

.reading-time {
    font-size: 0.85em;
    color: #666;
    margin-bottom: 1.5em;
}
"""
)


# Dark theme
THEME_DARK = Theme(
    name="dark",
    description="Dark mode theme for comfortable night reading",
    css="""
body {
    font-family: Charter, "Bitstream Charter", Georgia, "Times New Roman", serif;
    font-size: 1.1em;
    line-height: 1.4;
    color: #e0e0e0;
    background: #1a1a1a;
    max-width: 65ch;
    margin: 0 auto;
    padding: 2em 1.5em;
    text-rendering: optimizeLegibility;
    font-kerning: normal;
}

p {
    margin-top: 0;
    margin-bottom: 0;
    text-indent: 1.5em;
}

h1 + p, h2 + p, h3 + p, h4 + p { text-indent: 0; }

h1, h2, h3, h4 {
    font-family: Charter, "Bitstream Charter", Georgia, serif;
    font-weight: 600;
    line-height: 1.25;
    margin-bottom: 0.5em;
    color: #fff;
}

h1 { font-size: 1.5em; margin-top: 0; }
h2 { font-size: 1.2em; margin-top: 2em; }
h3 { font-size: 1.05em; margin-top: 1.5em; }

blockquote {
    margin: 1.2em 1.5em;
    font-style: italic;
    color: #bbb;
    border-left: 2px solid #444;
    padding-left: 1em;
}

a { color: #6db3f2; text-decoration: none; }

.intro {
    background: #252525;
    padding: 1em 1.2em;
    margin-bottom: 1.5em;
    border-radius: 2px;
}

.watch-link {
    margin-top: 2em;
    padding: 0.8em 1em;
    background: #252525;
    font-size: 0.88em;
    border-radius: 2px;
}

code {
    font-family: "SF Mono", Menlo, Monaco, monospace;
    font-size: 0.85em;
    background: #2d2d2d;
    padding: 0.1em 0.25em;
    border-radius: 2px;
}

pre { background: #2d2d2d; padding: 1em; border-radius: 2px; }

hr { border: none; border-top: 1px solid #444; margin: 1.5em 0; }
"""
)


# Sans-serif modern theme
THEME_MODERN = Theme(
    name="modern",
    description="Clean sans-serif theme with modern aesthetics",
    css="""
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 1.05em;
    line-height: 1.6;
    color: #333;
    max-width: 680px;
    margin: 0 auto;
    padding: 2em 1.5em;
    text-rendering: optimizeLegibility;
}

p {
    margin-top: 0;
    margin-bottom: 1.2em;
}

h1, h2, h3, h4 {
    font-weight: 700;
    line-height: 1.3;
    margin-top: 2em;
    margin-bottom: 0.5em;
    color: #111;
}

h1 { font-size: 2em; margin-top: 0; }
h2 { font-size: 1.5em; }
h3 { font-size: 1.2em; }

blockquote {
    margin: 1.5em 0;
    padding: 1em 1.5em;
    background: #f7f7f7;
    border-left: 4px solid #0066cc;
    font-style: normal;
}

a { color: #0066cc; text-decoration: none; }
a:hover { text-decoration: underline; }

.intro {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5em;
    margin-bottom: 2em;
    border-radius: 8px;
}

.watch-link {
    margin-top: 2em;
    padding: 1em;
    background: #f0f0f0;
    border-radius: 8px;
    text-align: center;
}

code {
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 0.9em;
    background: #f4f4f4;
    padding: 0.2em 0.4em;
    border-radius: 4px;
}

pre {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 1.5em;
    border-radius: 8px;
    overflow-x: auto;
}

hr { border: none; border-top: 2px solid #eee; margin: 2em 0; }
"""
)


# Minimal/distraction-free theme
THEME_MINIMAL = Theme(
    name="minimal",
    description="Ultra-minimal theme for distraction-free reading",
    css="""
body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.2em;
    line-height: 1.8;
    color: #222;
    max-width: 600px;
    margin: 0 auto;
    padding: 3em 2em;
}

p {
    margin-top: 0;
    margin-bottom: 1.5em;
}

h1, h2, h3 {
    font-weight: normal;
    font-style: italic;
    margin-top: 3em;
    margin-bottom: 1em;
}

h1 { font-size: 1.4em; margin-top: 0; }
h2 { font-size: 1.2em; }
h3 { font-size: 1.1em; }

blockquote {
    margin: 2em 0;
    padding-left: 2em;
    border-left: 1px solid #ccc;
    font-style: italic;
    color: #555;
}

a { color: inherit; }

.intro, .watch-link { display: none; }

code, pre { font-family: monospace; }

hr { display: none; }
"""
)


# All available themes
THEMES: Dict[str, Theme] = {
    "default": THEME_DEFAULT,
    "dark": THEME_DARK,
    "modern": THEME_MODERN,
    "minimal": THEME_MINIMAL,
}


def get_theme(name: str) -> Theme:
    """Get a theme by name."""
    if name not in THEMES:
        available = ", ".join(THEMES.keys())
        raise ValueError(f"Unknown theme '{name}'. Available: {available}")
    return THEMES[name]


def list_themes() -> Dict[str, str]:
    """List all available themes with descriptions."""
    return {name: theme.description for name, theme in THEMES.items()}


def register_theme(theme: Theme) -> None:
    """Register a custom theme."""
    THEMES[theme.name] = theme


def load_custom_css(path: str) -> Theme:
    """Load a custom CSS file as a theme."""
    with open(path, 'r') as f:
        css = f.read()
    return Theme(
        name="custom",
        description=f"Custom theme from {path}",
        css=css
    )
