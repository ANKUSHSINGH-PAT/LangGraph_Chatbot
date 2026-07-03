from __future__ import annotations

import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict


from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from mssql_checkpointer import MSSQLSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
import requests
from browser_summary import summarize_url_or_query

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
llm = ChatOpenAI(
    model="anthropic/claude-3-haiku",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_KEY,
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})


# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def _ocr_pdf(pdf_path: str) -> list[Document]:
    """
    Fallback: render each PDF page to an image and run Tesseract OCR on it.
    Requires:  pip install pdf2image pytesseract
    Also requires Tesseract to be installed on the system:
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
      Linux:   sudo apt install tesseract-ocr
      macOS:   brew install tesseract
    """
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OCR dependencies are not installed. "
            "Run:  pip install pdf2image pytesseract\n"
            "Also install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc

    # Point pytesseract at the default Windows install path if not on PATH
    tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Resolve poppler's bin directory so pdf2image doesn't need it on PATH.
    # Check common Windows install locations; fall back to None (use PATH).
    import glob as _glob
    _winget_base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    # Discover any WinGet-installed poppler version automatically
    _winget_poppler_bins = _glob.glob(os.path.join(_winget_base, "*oppler*", "poppler-*", "Library", "bin"))
    _POPPLER_CANDIDATES = _winget_poppler_bins + [
        r"C:\Program Files\poppler\Library\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\poppler\Library\bin",
        r"C:\poppler\bin",
    ]
    poppler_path: Optional[str] = None
    for _candidate in _POPPLER_CANDIDATES:
        if os.path.isdir(_candidate):
            poppler_path = _candidate
            break

    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    docs: list[Document] = []
    for i, page_img in enumerate(pages):
        text = pytesseract.image_to_string(page_img)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"page": i, "source": pdf_path}))
    return docs


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    If the PDF has no selectable text (scanned), OCR is attempted automatically.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        # --- OCR fallback for scanned / image-only PDFs ---
        if not chunks:
            docs = _ocr_pdf(temp_path)
            chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(
                "No text could be extracted from the PDF even after OCR. "
                "The file may be blank or the images unreadable."
            )

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=AB8HU9VQW5RYBYVQ"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


@tool
def web_summary_tool(url_or_query: str) -> dict:
    """
    Summarize a webpage or topic from the web.
    Accepts either a URL or a plain search query.
    """
    try:
        return summarize_url_or_query(url_or_query)
    except Exception as exc:
        return {"error": str(exc), "query": url_or_query}


tools = [search_tool, get_stock_price, calculator, rag_tool, web_summary_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 4b. Custom ToolNode with config injection
# -------------------
class ConfigInjectingToolNode:
    """Wraps ToolNode to inject thread_id from config into rag_tool calls."""

    def __init__(self, tools):
        self.tool_node = ToolNode(tools)

    def __call__(self, state: ChatState, config: Optional[dict] = None):
        # Inject thread_id from config into any rag_tool calls
        if config and isinstance(config, dict):
            thread_id = config.get("configurable", {}).get("thread_id")
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "tool_calls"):
                for tc in last_msg.tool_calls:
                    if tc["name"] == "rag_tool" and thread_id:
                        tc["args"]["thread_id"] = thread_id
        return self.tool_node.invoke(state, config)


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=
            f"""
You are a STRICT tool-using assistant.

Rules:

- If user asks about latest news or current events → ALWAYS call DuckDuckGo search tool.
- If user asks about stock price → ALWAYS call get_stock_price.
- If user asks math → ALWAYS call calculator.
- If user asks about uploaded PDF → ALWAYS call rag_tool with thread_id {thread_id}.
- If user asks to summarize a URL, webpage, or web topic → ALWAYS call web_summary_tool.
- NEVER answer directly when a tool applies.
- Prefer tools over your own knowledge.

Thread id: {thread_id}
"""
        
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ConfigInjectingToolNode(tools)

# -------------------
# 6. Checkpointer  (SQL Server)
# -------------------
checkpointer = MSSQLSaver.from_conn_string()

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})