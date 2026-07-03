from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlparse

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional dependency
    sync_playwright = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag in {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "tr", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self.parts.append(text)


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.netloc)


def _build_search_url(query: str) -> str:
    return f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}"


def extract_text_from_html(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    text = " ".join(part for part in parser.parts if part)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_with_requests(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="ignore")
        return extract_text_from_html(html)


def fetch_page_text(url: str) -> str:
    if sync_playwright is None:
        return _fetch_with_requests(url)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(1500)
            content = page.content()
            browser.close()
            text = extract_text_from_html(content)
            return text if text else _fetch_with_requests(url)
    except Exception:
        return _fetch_with_requests(url)


def summarize_text(text: str, max_sentences: int = 4) -> str:
    if not text or not text.strip():
        return "No readable content was found on the page."

    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > 6000:
        cleaned = cleaned[:6000]

    if os.getenv("OPEN_ROUTER_KEY") and ChatOpenAI is not None:
        try:
            llm = ChatOpenAI(
                model="anthropic/claude-3-haiku",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPEN_ROUTER_KEY"),
            )
            prompt = (
                "Summarize the following web content in "
                f"{max_sentences} concise sentences:\n\n{cleaned}"
            )
            response = llm.invoke(prompt)
            return getattr(response, "content", str(response))
        except Exception:
            pass

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return "No readable content was found on the page."
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    return " ".join(sentences[:max_sentences])


def summarize_url_or_query(query: str) -> dict:
    if not query or not query.strip():
        raise ValueError("Please enter a URL or search query.")

    target = query.strip()
    if not _is_url(target):
        target = _build_search_url(target)

    page_text = fetch_page_text(target)
    summary = summarize_text(page_text)
    return {
        "url": target,
        "summary": summary,
        "text_length": len(page_text),
    }
