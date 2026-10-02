"""Normalize model inputs while preserving the original evidence text."""

from html.parser import HTMLParser


_BLOCK_TAGS = {
    "article", "blockquote", "br", "dd", "div", "dl", "dt", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "ol",
    "p", "pre", "section", "table", "td", "th", "tr", "ul",
}


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in _BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in _BLOCK_TAGS:
            self.parts.append(" ")


def normalize_text(text):
    """Strip HTML, decode entities, and collapse whitespace without joining paragraphs."""
    parser = _TextParser()
    parser.feed(text)
    parser.close()
    return " ".join("".join(parser.parts).split())
