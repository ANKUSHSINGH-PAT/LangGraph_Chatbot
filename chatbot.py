import streamlit as st
import time
from langgraph_rag_backend import (
    get_chatbot,
    ingest_pdf,
    thread_document_metadata,
    delete_thread_checkpoints,
)
from thread_DB import init_db, save_thread, get_threads, delete_thread
from langchain_core.messages import HumanMessage, AIMessage
from browser_summary import summarize_url_or_query
from puppeteer_mcp_client import prewarm_puppeteer_client
import uuid

# Performance monitoring
_perf_timers = {}

def _perf_start(label: str):
    _perf_timers[label] = time.time()

def _perf_end(label: str):
    if label in _perf_timers:
        elapsed = time.time() - _perf_timers[label]
        if elapsed > 1.0:  # Log slow operations (>1s)
            print(f"[PERF] {label}: {elapsed:.2f}s")
        del _perf_timers[label]

# Simple initialization (non-blocking)
if 'app_initialized' not in st.session_state:
    with st.spinner("🚀 Loading AI models and initializing services..."):
        # Initialise the SQL Server threads table (no-op if it already exists)
        init_db()
    
    st.session_state['app_initialized'] = True

# **************************************** utility functions *************************

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    save_thread(thread_id, "New Chat")
    st.session_state['message_history'] = []


def load_conversation(thread_id):
    state = get_chatbot().get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()
    save_thread(st.session_state['thread_id'], "New Chat")

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# ============================ Sidebar ============================
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Web summary")
st.sidebar.caption("You can also ask the assistant in chat: "
                   "'summarize https://example.com' or 'summarize AI news'")
web_query = st.sidebar.text_input("Search URL or topic", placeholder="https://example.com or AI news")
if st.sidebar.button("Summarize page", use_container_width=True):
    if web_query:
        with st.sidebar.status("Gathering page content…", expanded=True) as status_box:
            try:
                result = summarize_url_or_query(web_query)
                st.session_state["web_summary_result"] = result
                status_box.update(label="✅ Summary ready", state="complete", expanded=False)
            except Exception as exc:
                st.session_state["web_summary_result"] = {"error": str(exc)}
                status_box.update(label="⚠️ Summary failed", state="error", expanded=False)
    else:
        st.sidebar.warning("Enter a URL or a search topic first.")

if "web_summary_result" in st.session_state:
    result = st.session_state["web_summary_result"]
    if result.get("error"):
        st.sidebar.error(result["error"])
    else:
        st.sidebar.caption(f"Source: {result.get('url')}")
        st.sidebar.write(result.get("summary", ""))

st.sidebar.subheader("Past conversations")
past_threads = get_threads()  # [(thread_id, title), ...]
if not past_threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid, title in past_threads:
        display = title if title and title != "New Chat" else tid[:16] + "…"
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        with col1:
            if st.button(display, key=f"side-thread-{tid}", use_container_width=True):
                st.session_state['thread_id'] = tid
                messages = load_conversation(tid)
                temp_messages = []
                for msg in messages:
                    role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                    temp_messages.append({'role': role, 'content': msg.content})
                st.session_state['message_history'] = temp_messages
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del-thread-{tid}", help=f"Delete {display}"):
                # Delete from threads table
                delete_thread(tid)
                # Delete from checkpoints/writes tables
                delete_thread_checkpoints(tid)
                # Clean up in-memory PDF data if present
                if tid in st.session_state["ingested_docs"]:
                    del st.session_state["ingested_docs"][tid]
                # If the deleted thread is the current one, reset to a new chat
                if st.session_state['thread_id'] == tid:
                    reset_chat()
                st.rerun()

# **************************************** Main UI ************************************

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    _perf_start("total_response")
    
    # Update the thread title on first real message
    if len(st.session_state['message_history']) == 0:
        _perf_start("save_thread")
        save_thread(thread_key, user_input[:60])
        _perf_end("save_thread")

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            _perf_start("llm_generation")
            def ai_only_stream():
                for message_chunk, metadata in get_chatbot().stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages"
                ):
                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())
            _perf_end("llm_generation")

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    _perf_end("total_response")

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )
