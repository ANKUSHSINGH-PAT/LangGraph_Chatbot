"""
MCP client wrapper for the Puppeteer MCP server.
Provides browser automation tools as LangGraph-compatible functions.

Uses direct JSON-RPC over subprocess stdio (avoids anyio/async issues
with LangGraph's thread-pool workers).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from typing import Any, Optional

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Synchronous JSON-RPC MCP client over subprocess stdio
# ---------------------------------------------------------------------------


class PuppeteerMCPClient:
    """Manages a subprocess connection to the Puppeteer MCP server."""

    def __init__(self, server_command: Optional[list[str]] = None) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        # Response map: request_id -> (event, response_dict)
        self._pending: dict[str, tuple[threading.Event, dict]] = {}
        self._pending_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

    # ------------------------------------------------------------------
    # Static helpers for finding executables
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_server_command() -> list[str]:
        """Build the command to launch the MCP Puppeteer server."""
        node_path = PuppeteerMCPClient._resolve_node_path()
        script = PuppeteerMCPClient._resolve_server_script()
        if script:
            return [node_path, script]
        return ["npx", "-y", "@modelcontextprotocol/server-puppeteer"]

    @staticmethod
    def _resolve_server_script() -> str:
        """Path to globally installed server-puppeteer dist/index.js."""
        node_path = PuppeteerMCPClient._resolve_node_path()
        if node_path != "node":
            node_dir = os.path.dirname(node_path)
            script = os.path.join(
                node_dir, "node_modules",
                "@modelcontextprotocol", "server-puppeteer",
                "dist", "index.js"
            )
            if os.path.isfile(script):
                return script
        return ""

    @staticmethod
    def _resolve_chrome_path() -> Optional[str]:
        """Find Chrome/Chromium executable on common install paths."""
        candidates = [
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Chromium", "Application", "chrome.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Microsoft", "Edge", "Application", "msedge.exe"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _resolve_node_path() -> str:
        """Return path to a modern Node.js binary (prefer nvm's newest)."""
        import glob as _glob

        nvm_root = os.environ.get("NVM_HOME", "")
        if not nvm_root:
            nvm_root = os.path.join(os.environ.get("LOCALAPPDATA", ""), "nvm")
        if os.path.isdir(nvm_root):
            version_dirs = sorted(
                (d for d in _glob.glob(os.path.join(nvm_root, "v*")) if os.path.isdir(d)),
                key=lambda d: [int(x) for x in os.path.basename(d).lstrip("v").split(".") if x.isdigit()] or [0],
                reverse=True,
            )
            for vdir in version_dirs:
                node_exe = os.path.join(vdir, "node.exe")
                if os.path.isfile(node_exe):
                    return node_exe

        nvm_symlink = os.environ.get("NVM_SYMLINK")
        if nvm_symlink:
            sym_node = os.path.join(nvm_symlink, "node.exe")
            if os.path.isfile(sym_node):
                return sym_node

        return "node"

    # ------------------------------------------------------------------
    # Background stdout reader thread
    # ------------------------------------------------------------------

    def _reader_main(self) -> None:
        """Background thread: continuously read stdout lines and dispatch."""
        try:
            while not self._shutdown.is_set():
                if self._proc is None or self._proc.stdout is None:
                    break
                line = self._proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # If this is a response with an id, signal the waiting caller
                msg_id = msg.get("id")
                if msg_id is not None:
                    with self._pending_lock:
                        entry = self._pending.get(msg_id)
                        if entry is not None:
                            event, _ = entry
                            self._pending[msg_id] = (event, msg)
                            event.set()
        except Exception:
            pass

    def _stderr_main(self) -> None:
        """Background thread: drain stderr pipe to prevent blocking."""
        try:
            while not self._shutdown.is_set():
                if self._proc is None or self._proc.stderr is None:
                    break
                chunk = self._proc.stderr.read(4096)
                if not chunk:
                    break
                # Silently drain
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the Puppeteer MCP server subprocess and initialize."""
        with self._lock:
            if self._proc is not None:
                return

            env = {**os.environ}

            chrome_path = self._resolve_chrome_path()
            if chrome_path:
                env["PUPPETEER_EXECUTABLE_PATH"] = chrome_path
                chrome_dir = os.path.dirname(chrome_path)
                if chrome_dir not in env.get("PATH", ""):
                    env["PATH"] = chrome_dir + os.pathsep + env["PATH"]

            node_path = self._resolve_node_path()
            if node_path != "node":
                node_dir = os.path.dirname(node_path)
                env["PATH"] = node_dir + os.pathsep + env.get("PATH", "")
                npm_dir = os.path.join(node_dir, "node_modules", "npm", "bin")
                if os.path.isdir(npm_dir) and npm_dir not in env.get("PATH", ""):
                    env["PATH"] = npm_dir + os.pathsep + env["PATH"]

            env["PUPPETEER_LAUNCH_OPTIONS"] = '{"headless": true, "args": ["--no-sandbox", "--disable-setuid-sandbox"]}'
            env["ALLOW_DANGEROUS"] = "true"

            command = self._resolve_server_command()

            self._proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                bufsize=1,
            )

            # Start background reader / stderr drainer threads
            self._shutdown.clear()
            self._reader_thread = threading.Thread(target=self._reader_main, daemon=True)
            self._reader_thread.start()
            self._stderr_thread = threading.Thread(target=self._stderr_main, daemon=True)
            self._stderr_thread.start()

            # Perform MCP initialize handshake
            initialize_resp = self._call("initialize", {
                "protocolVersion": "0.1.0",
                "capabilities": {},
                "clientInfo": {"name": "puppeteer-mcp-client", "version": "1.0.0"},
            })
            if "error" in initialize_resp:
                raise RuntimeError(f"MCP initialize error: {initialize_resp['error']}")

            # Send initialized notification (fire-and-forget)
            self._send_notification("notifications/initialized", {})

    def stop(self) -> None:
        """Shut down the subprocess."""
        self._shutdown.set()
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.stdin.close()
                except Exception:
                    pass
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=10)
                except Exception:
                    self._proc.kill()
                self._proc = None

    # ------------------------------------------------------------------
    # JSON-RPC communication
    # ------------------------------------------------------------------

    def _call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the matching response."""
        req_id = str(uuid.uuid4())
        event = threading.Event()

        with self._pending_lock:
            self._pending[req_id] = (event, None)

        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })

        with self._lock:
            if self._proc is None or self._proc.stdin is None:
                with self._pending_lock:
                    del self._pending[req_id]
                raise RuntimeError("MCP server not started")
            self._proc.stdin.write(msg + "\n")
            self._proc.stdin.flush()

        # Wait for response (with timeout)
        if not event.wait(timeout=120):
            with self._pending_lock:
                del self._pending[req_id]
            raise TimeoutError(f"MCP request timed out: {method}")

        with self._pending_lock:
            _, resp = self._pending.pop(req_id, (None, {"error": {"message": "no response"}}))

        return resp

    def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no ID, no response expected)."""
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })
        with self._lock:
            if self._proc is None or self._proc.stdin is None:
                raise RuntimeError("MCP server not started")
            self._proc.stdin.write(msg + "\n")
            self._proc.stdin.flush()

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the Puppeteer MCP server and return result text."""
        resp = self._call("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        if "error" in resp:
            err = resp["error"]
            raise RuntimeError(f"MCP tool call error: {err.get('message', str(err))}")

        result = resp.get("result", {})
        content = result.get("content", [])
        text_parts: list[str] = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                text_parts.append(f"[Base64 image: {len(item.get('data', ''))} bytes]")
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the list of tools from the server."""
        resp = self._call("tools/list", {})

        if "error" in resp:
            err = resp["error"]
            raise RuntimeError(f"MCP list tools error: {err.get('message', str(err))}")

        result = resp.get("result", {})
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("inputSchema", {}),
            }
            for t in result.get("tools", [])
        ]


# ---------------------------------------------------------------------------
# Global singleton instance
# ---------------------------------------------------------------------------

_CLIENT: Optional[PuppeteerMCPClient] = None
_CLIENT_LOCK = threading.Lock()


def _get_client() -> PuppeteerMCPClient:
    """Get or create the global Puppeteer MCP client (thread-safe)."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = PuppeteerMCPClient()
            import time
            start = time.time()
            _CLIENT.start()
            elapsed = time.time() - start
            if elapsed > 2.0:
                print(f"[PERF] Puppeteer MCP server startup: {elapsed:.2f}s")
        return _CLIENT


def prewarm_puppeteer_client() -> None:
    """Pre-initialize the Puppeteer MCP client to avoid first-request delay."""
    try:
        _get_client()
    except Exception as exc:
        print(f"[WARN] Puppeteer MCP pre-warm failed: {exc}")


def close_puppeteer_client() -> None:
    """Shut down the global Puppeteer MCP client."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            _CLIENT.stop()
            _CLIENT = None


# ---------------------------------------------------------------------------
# LangGraph-compatible @tool functions
# ---------------------------------------------------------------------------


def _call_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Synchronous helper to call a tool via the global client."""
    client = _get_client()
    return client.call_tool(tool_name, arguments)


@tool
def puppeteer_navigate(url: str) -> str:
    """
    Navigate to a URL in the headless browser.
    Use this to open web pages that require JavaScript rendering.
    """
    return _call_tool("puppeteer_navigate", {"url": url})


@tool
def puppeteer_click(selector: str) -> str:
    """
    Click on an element identified by CSS selector.
    Example selectors: '#my-button', '.search-btn', 'button[type="submit"]'
    """
    return _call_tool("puppeteer_click", {"selector": selector})


@tool
def puppeteer_type_text(selector: str, text: str) -> str:
    """
    Type text into an input field identified by CSS selector.
    """
    return _call_tool("puppeteer_fill", {"selector": selector, "value": text})


@tool
def puppeteer_press_key(key: str) -> str:
    """
    Press a keyboard key (e.g., 'Enter', 'Escape', 'Tab', 'ArrowDown').
    Executed via JavaScript evaluate on the page.
    """
    js = f"document.dispatchEvent(new KeyboardEvent('keydown', {{key: '{key}'}}))"
    return _call_tool("puppeteer_evaluate", {"script": js})


@tool
def puppeteer_get_text() -> str:
    """Extract all visible text from the current page."""
    js = "document.body.innerText"
    return _call_tool("puppeteer_evaluate", {"script": js})


@tool
def puppeteer_get_html() -> str:
    """Get the full HTML content of the current page."""
    js = "document.documentElement.outerHTML"
    return _call_tool("puppeteer_evaluate", {"script": js})


@tool
def puppeteer_screenshot() -> str:
    """
    Take a screenshot of the current page and return it as base64.
    """
    return _call_tool("puppeteer_screenshot", {
        "name": "screenshot",
        "encoded": True,
    })


@tool
def puppeteer_search_google(query: str) -> str:
    """
    Search Google and return organic search results with titles, URLs, and snippets.
    Navigates to google.com search and extracts page text.
    """
    import urllib.parse
    url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
    _call_tool("puppeteer_navigate", {"url": url})
    text = _call_tool("puppeteer_evaluate", {"script": "document.body.innerText"})
    return text


# ---------------------------------------------------------------------------
# Convenience: all puppeteer tools in a list for easy registration
# ---------------------------------------------------------------------------
PUPPETEER_TOOLS = [
    puppeteer_navigate,
    puppeteer_click,
    puppeteer_type_text,
    puppeteer_press_key,
    puppeteer_get_text,
    puppeteer_get_html,
    puppeteer_screenshot,
    puppeteer_search_google,
]