from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urlparse
from functools import lru_cache

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - optional dependency
    sync_playwright = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None

# Global browser instance for reuse
_BROWSER_INSTANCE = None
_BROWSER_CONTEXT = None

# Simple cache for web summaries to avoid re-fetching
_web_cache: dict[str, dict] = {}
_CACHE_MAX_SIZE = 50


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


def _get_browser():
    """Get or create a persistent browser instance for reuse."""
    global _BROWSER_INSTANCE, _BROWSER_CONTEXT
    if sync_playwright is None:
        return None
    
    if _BROWSER_INSTANCE is None or not _BROWSER_INSTANCE.is_connected():
        try:
            playwright = sync_playwright().start()
            _BROWSER_INSTANCE = playwright.chromium.launch(headless=True)
            _BROWSER_CONTEXT = _BROWSER_INSTANCE.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
        except Exception:
            return None
    return _BROWSER_CONTEXT


def _close_browser():
    """Close the persistent browser instance."""
    global _BROWSER_INSTANCE, _BROWSER_CONTEXT
    try:
        if _BROWSER_CONTEXT:
            _BROWSER_CONTEXT.close()
        if _BROWSER_INSTANCE:
            _BROWSER_INSTANCE.close()
    except Exception:
        pass
    finally:
        _BROWSER_INSTANCE = None
        _BROWSER_CONTEXT = None


def fetch_page_text(url: str) -> str:
    if sync_playwright is None:
        return _fetch_with_requests(url)

    try:
        context = _get_browser()
        if context:
            page = context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(1000)  # Reduced from 1500ms
                content = page.content()
                text = extract_text_from_html(content)
                page.close()
                return text if text else _fetch_with_requests(url)
            except Exception:
                page.close()
                return _fetch_with_requests(url)
        else:
            return _fetch_with_requests(url)
    except Exception:
        return _fetch_with_requests(url)


# Cache LLM instance to avoid recreating it
_cached_llm = None


def _get_llm():
    """Get or create cached LLM instance."""
    global _cached_llm
    if _cached_llm is None and os.getenv("OPEN_ROUTER_KEY") and ChatOpenAI is not None:
        try:
            _cached_llm = ChatOpenAI(
                model="anthropic/claude-3-haiku",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPEN_ROUTER_KEY"),
                temperature=0.7,
                max_tokens=512,  # Limit for faster responses
            )
        except Exception:
            pass
    return _cached_llm


def summarize_text(text: str, max_sentences: int = 4) -> str:
    if not text or not text.strip():
        return "No readable content was found on the page."

    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) > 6000:
        cleaned = cleaned[:6000]

    llm = _get_llm()
    if llm:
        try:
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

    # Check cache first
    global _web_cache
    if target in _web_cache:
        return _web_cache[target]

    page_text = fetch_page_text(target)
    summary = summarize_text(page_text)
    result = {
        "url": target,
        "summary": summary,
        "text_length": len(page_text),
    }

    # Cache the result
    _web_cache[target] = result
    if len(_web_cache) > _CACHE_MAX_SIZE:
        # Remove oldest entry
        _web_cache.pop(next(iter(_web_cache)))

    return result


# Register cleanup on exit
import atexit
atexit.register(_close_browser)
