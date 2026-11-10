import os
import sqlite3
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

DB_PATH = "chatbot.db"

# ******************************************************
# Database Helpers & Metadata Management
# ******************************************************

def get_db_connection():
    """Create a thread-safe SQLite connection."""
    conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Ensure the thread_metadata table exists."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_metadata (
                thread_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT,
                updated_at TEXT,
                model TEXT
            )
            """
        )
        conn.commit()

init_db()


def extract_text(content: Any) -> str:
    """Extract clean string text from various LangChain message content formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text", ""))
        return "".join(parts)
    if hasattr(content, "text"):
        return getattr(content, "text")
    return str(content) if content is not None else ""


def generate_title_from_text(text: str, max_len: int = 36) -> str:
    """Generate a clean, readable title from text."""
    clean = " ".join(text.strip().split())
    if not clean:
        return "New Conversation"
    if len(clean) > max_len:
        return clean[:max_len].rstrip() + "..."
    return clean


def save_thread_metadata(thread_id: str, title: Optional[str] = None, model: Optional[str] = None):
    """Insert or update metadata for a thread."""
    now_iso = datetime.utcnow().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT title, created_at, model FROM thread_metadata WHERE thread_id = ?", (thread_id,))
        row = cursor.fetchone()

        if row:
            new_title = title if title is not None else row["title"]
            new_model = model if model is not None else row["model"]
            cursor.execute(
                """
                UPDATE thread_metadata
                SET title = ?, model = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (new_title, new_model, now_iso, thread_id)
            )
        else:
            final_title = title if title else "New Conversation"
            cursor.execute(
                """
                INSERT INTO thread_metadata (thread_id, title, created_at, updated_at, model)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, final_title, now_iso, now_iso, model or "gemini-3.8-flash")
            )
        conn.commit()


def delete_thread(thread_id: str):
    """Delete all records associated with a thread from SQLite."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM thread_metadata WHERE thread_id = ?", (thread_id,))
        conn.commit()


def clear_all_threads():
    """Wipe all conversations and metadata."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints")
        cursor.execute("DELETE FROM writes")
        cursor.execute("DELETE FROM thread_metadata")
        conn.commit()


# ******************************************************
# LangGraph Setup
# ******************************************************

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def get_llm_instance(model_name: str = "gemini-3.8-flash", temperature: float = 0.7, api_key: Optional[str] = None):
    """Factory to create a ChatGoogleGenerativeAI instance with proper parameters."""
    key = api_key or os.getenv("GOOGLE_API_KEY")
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "google_api_key": key,
    }
    # Models like 'flash-lite' use fixed sampling and warn if temperature is passed
    if "flash-lite" not in model_name:
        kwargs["temperature"] = max(0.0, min(1.0, float(temperature)))
    return ChatGoogleGenerativeAI(**kwargs)


def chat_node(state: ChatState, config: Optional[RunnableConfig] = None):
    """Chatbot node supporting dynamic model, temperature, and persona system instructions."""
    conf = config.get("configurable", {}) if config else {}
    model_name = conf.get("model_name", "gemini-3.8-flash")
    temperature = conf.get("temperature", 0.7)
    system_prompt = conf.get("system_prompt", "")
    api_key = conf.get("api_key", None)

    llm = get_llm_instance(model_name=model_name, temperature=temperature, api_key=api_key)

    raw_messages = list(state.get("messages", []))

    # Prepend SystemMessage if configured and not already present
    messages_to_send = []
    if system_prompt:
        messages_to_send.append(SystemMessage(content=system_prompt))
    messages_to_send.extend(raw_messages)

    response = llm.invoke(messages_to_send)
    return {"messages": [response]}


# SQLite Checkpointer
db_conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=db_conn)

# Build StateGraph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


# ******************************************************
# Thread Retrieval & History Helpers
# ******************************************************

def retrieve_all_threads_info() -> List[Dict[str, Any]]:
    """
    Retrieve all conversation threads ordered by updated_at descending.
    Automatically backfills metadata for existing checkpoint threads that lack entries.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Discover all threads present in checkpoints
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        checkpoint_threads = {row["thread_id"] for row in cursor.fetchall()}

        # 2. Check metadata table
        cursor.execute("SELECT thread_id, title, created_at, updated_at, model FROM thread_metadata")
        existing_meta = {row["thread_id"]: dict(row) for row in cursor.fetchall()}

        # 3. Backfill any checkpoint threads missing in metadata
        for tid in checkpoint_threads:
            if tid not in existing_meta:
                title = "New Conversation"
                try:
                    state = chatbot.get_state({"configurable": {"thread_id": tid}})
                    msgs = state.values.get("messages", [])
                    for m in msgs:
                        if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
                            msg_text = extract_text(m.content)
                            if msg_text.strip():
                                title = generate_title_from_text(msg_text)
                                break
                except Exception:
                    pass
                save_thread_metadata(tid, title=title)

        # 4. Return all threads ordered by updated_at DESC
        cursor.execute(
            """
            SELECT thread_id, title, created_at, updated_at, model
            FROM thread_metadata
            ORDER BY updated_at DESC
            """
        )
        return [dict(row) for row in cursor.fetchall()]


def retrieve_allthreads() -> List[str]:
    """Backwards-compatible thread ID retriever."""
    return [t["thread_id"] for t in retrieve_all_threads_info()]


def load_conversation_messages(thread_id: str) -> List[Dict[str, str]]:
    """
    Load formatted conversation history for a thread.
    Returns: [{'role': 'user'|'assistant', 'content': str}, ...]
    """
    try:
        state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
        raw_msgs = state.values.get("messages", [])
        formatted = []
        for msg in raw_msgs:
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                role = "user"
            elif isinstance(msg, SystemMessage) or getattr(msg, "type", "") == "system":
                continue  # Don't show system prompts as chat messages
            else:
                role = "assistant"
            content = extract_text(msg.content)
            if content:
                formatted.append({"role": role, "content": content})
        return formatted
    except Exception:
        return []
