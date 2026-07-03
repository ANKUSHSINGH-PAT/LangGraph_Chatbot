from __future__ import annotations

import os
import sys
from mcp.server.fastmcp import FastMCP

from browser_summary import summarize_url_or_query

mcp = FastMCP("document-intelligence")


@mcp.tool()
def summarize_web_content(query: str) -> dict:
    """Summarize a webpage or search query using the browser-based helper."""
    return summarize_url_or_query(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
