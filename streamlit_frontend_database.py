import json
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_database_backend import (
    chatbot,
    retrieve_all_threads_info,
    load_conversation_messages,
    save_thread_metadata,
    delete_thread,
    clear_all_threads,
    extract_text,
    generate_title_from_text,
)

# ******************************************************
# Page Configuration & Modern Styling
# ******************************************************
st.set_page_config(
    page_title="Gemini AI Chatbot • LangGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished look
st.markdown(
    """
    <style>
    /* Main container tweaks */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    
    /* Header badges */
    .badge-container {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 500;
        border-radius: 9999px;
        background-color: rgba(66, 133, 244, 0.12);
        color: #4285F4;
        border: 1px solid rgba(66, 133, 244, 0.25);
    }
    .badge-pill-secondary {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        font-size: 0.8rem;
        font-weight: 500;
        border-radius: 9999px;
        background-color: rgba(128, 128, 128, 0.12);
        color: #888888;
        border: 1px solid rgba(128, 128, 128, 0.25);
    }

    /* Welcome hero banner */
    .hero-card {
        padding: 2rem;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(66, 133, 244, 0.08) 0%, rgba(219, 68, 85, 0.05) 100%);
        border: 1px solid rgba(66, 133, 244, 0.18);
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .hero-card h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .hero-card p {
        margin-top: 0.5rem;
        font-size: 0.95rem;
        color: #666;
    }

    /* Sidebar button styles */
    div[data-testid="stSidebar"] button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] button[kind="secondary"] {
        border-radius: 8px;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ******************************************************
# Personas Definition
# ******************************************************
PERSONAS = {
    "🤖 General Assistant": "You are a helpful, respectful, and honest AI assistant. Provide clear, accurate, and insightful answers.",
    "👨‍💻 Senior Software Engineer": "You are a senior full-stack software engineer and code reviewer. Write production-quality, secure, and clean code with clear explanations and best practices.",
    "🎓 Patient Teacher & Tutor": "You are an encouraging and patient educator. Explain complex topics using simple analogies, intuitive step-by-step reasoning, and friendly examples.",
    "📝 Concise & Fluff-Free": "You are an ultra-concise assistant. Provide direct, bullet-pointed, high-density answers without conversational filler.",
    "🧠 Creative Brainstormer": "You are an imaginative ideation partner. Generate bold, inventive, and out-of-the-box concepts, options, and creative angles.",
    "✏️ Custom Instructions": ""
}

# ******************************************************
# Session State Setup
# ******************************************************
def generate_thread_id():
    return str(uuid.uuid4())

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "pending_input" not in st.session_state:
    st.session_state["pending_input"] = None

# ******************************************************
# Helpers
# ******************************************************
def reset_to_new_chat():
    """Start a fresh chat thread without cluttering until first message."""
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    st.session_state["message_history"] = []
    st.session_state["pending_input"] = None

def switch_to_thread(thread_id: str):
    """Switch active thread and load history."""
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = load_conversation_messages(thread_id)
    st.session_state["pending_input"] = None

def get_current_thread_title(all_threads_info: list, thread_id: str) -> str:
    for t in all_threads_info:
        if t["thread_id"] == thread_id:
            return t["title"]
    return "New Conversation"

# Load threads info
all_threads = retrieve_all_threads_info()
current_title = get_current_thread_title(all_threads, st.session_state["thread_id"])

# ******************************************************
# Sidebar
# ******************************************************
with st.sidebar:
    st.title("🤖 AI Chatbot")
    st.caption("Powered by LangGraph & Google Gemini")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        reset_to_new_chat()
        st.rerun()

    st.markdown("---")

    # Settings Expander
    with st.expander("⚙️ Model & Persona Settings", expanded=False):
        # Model Selection
        model_options = [
            ("gemini-3.8-flash", "⚡ gemini-3.8-flash (Recommended)"),
            ("gemini-3.5-flash-lite", "🚀 gemini-3.5-flash-lite (Ultra-Fast)"),
            ("gemini-3.1-pro-preview", "🧠 gemini-3.1-pro-preview (Deep Reasoning)")
        ]
        selected_model = st.selectbox(
            "Model",
            options=[opt[0] for opt in model_options],
            format_func=lambda x: dict(model_options)[x],
            index=0
        )

        # Persona Selection
        persona_choice = st.selectbox(
            "Persona / Role",
            options=list(PERSONAS.keys()),
            index=0
        )

        if persona_choice == "✏️ Custom Instructions":
            system_prompt = st.text_area(
                "Custom System Instructions",
                placeholder="E.g., You are an expert financial analyst...",
                height=100
            )
        else:
            system_prompt = PERSONAS[persona_choice]

        # Temperature
        is_fixed_temp = "flash-lite" in selected_model
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            disabled=is_fixed_temp,
            help="Higher values make output more random/creative. Ignored by flash-lite."
        )

        # Optional API Key override
        custom_key = st.text_input(
            "API Key Override (Optional)",
            type="password",
            placeholder="Uses GOOGLE_API_KEY from .env by default",
            help="Leave blank to use the server environment key."
        )

    st.subheader("Conversations")

    # Search / Filter
    search_query = st.text_input("🔍 Search chats", placeholder="Filter by title...", label_visibility="collapsed")

    filtered_threads = all_threads
    if search_query.strip():
        filtered_threads = [t for t in all_threads if search_query.lower() in t["title"].lower()]

    if not filtered_threads:
        st.caption("No conversations found." if search_query else "No past conversations yet.")
    else:
        for t in filtered_threads:
            tid = t["thread_id"]
            is_active = (tid == st.session_state["thread_id"])
            icon = "👉 " if is_active else "💬 "
            btn_label = f"{icon}{t['title']}"

            if st.button(
                btn_label,
                key=f"thread_btn_{tid}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if not is_active:
                    switch_to_thread(tid)
                    st.rerun()

    st.markdown("---")
    # Danger zone: Clear all
    with st.popover("🗑️ Clear All Chats", use_container_width=True):
        st.write("⚠️ This will permanently delete **all** conversations from the database.")
        if st.button("Confirm Clear Everything", type="primary", use_container_width=True):
            clear_all_threads()
            reset_to_new_chat()
            st.rerun()

# ******************************************************
# Main Chat Area Header
# ******************************************************
col_title, col_actions = st.columns([3, 1.2])

with col_title:
    st.subheader(current_title)
    st.markdown(
        f"""
        <div class="badge-container">
            <span class="badge-pill">⚡ {selected_model}</span>
            <span class="badge-pill-secondary">💬 {len(st.session_state['message_history'])} messages</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_actions:
    # Toolbar for active thread: Rename, Export, Delete
    act_col1, act_col2, act_col3 = st.columns(3)

    # 1. Rename Popover
    with act_col1:
        with st.popover("✏️", help="Rename conversation"):
            new_title_input = st.text_input("New Title", value=current_title)
            if st.button("Save", key="save_rename_btn", use_container_width=True):
                if new_title_input.strip():
                    save_thread_metadata(st.session_state["thread_id"], title=new_title_input.strip())
                    st.rerun()

    # 2. Export Popover
    with act_col2:
        with st.popover("📥", help="Export conversation"):
            st.write("**Export Chat**")
            # Build Markdown
            md_content = f"# Conversation: {current_title}\nModel: {selected_model}\n\n"
            for m in st.session_state["message_history"]:
                role_label = "👤 User" if m["role"] == "user" else "🤖 Assistant"
                md_content += f"### {role_label}\n{m['content']}\n\n"

            st.download_button(
                "📄 Download Markdown",
                data=md_content,
                file_name=f"{current_title.replace(' ', '_')[:25]}.md",
                mime="text/markdown",
                use_container_width=True
            )

            # Build JSON
            json_content = json.dumps({
                "thread_id": st.session_state["thread_id"],
                "title": current_title,
                "model": selected_model,
                "messages": st.session_state["message_history"]
            }, indent=2)

            st.download_button(
                "📦 Download JSON",
                data=json_content,
                file_name=f"{current_title.replace(' ', '_')[:25]}.json",
                mime="application/json",
                use_container_width=True
            )

    # 3. Delete Popover
    with act_col3:
        with st.popover("🗑️", help="Delete conversation"):
            st.write("Delete this chat?")
            if st.button("Confirm", key="delete_curr_thread_btn", type="primary", use_container_width=True):
                delete_thread(st.session_state["thread_id"])
                reset_to_new_chat()
                st.rerun()

st.divider()

# ******************************************************
# Empty State: Starter Prompts
# ******************************************************
if not st.session_state["message_history"]:
    st.markdown(
        """
        <div class="hero-card">
            <h2>👋 Welcome to your AI Assistant</h2>
            <p>Select a starter question below or type your own question in the chat bar.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if st.button("💡 Explain quantum computing simply", use_container_width=True):
            st.session_state["pending_input"] = "Explain quantum computing in simple, easy-to-understand terms with an analogy."
            st.rerun()
        if st.button("🐍 Write a Python script to parse JSON", use_container_width=True):
            st.session_state["pending_input"] = "Write a clean Python script using the json library to parse, validate, and manipulate JSON data."
            st.rerun()
    with p_col2:
        if st.button("📝 Draft a polite follow-up email", use_container_width=True):
            st.session_state["pending_input"] = "Draft a polite and professional follow-up email after a software engineering interview."
            st.rerun()
        if st.button("🧠 Brainstorm 5 innovative AI startup ideas", use_container_width=True):
            st.session_state["pending_input"] = "Brainstorm 5 innovative and feasible AI startup ideas for 2026, including their value proposition."
            st.rerun()

# ******************************************************
# Message Display
# ******************************************************
for message in st.session_state["message_history"]:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ******************************************************
# Input Handling & Streaming
# ******************************************************
# Check for pending starter prompt or direct user input
chat_input = st.chat_input("Ask me anything...")
user_input = st.session_state.get("pending_input") or chat_input

if user_input:
    # Clear pending input
    st.session_state["pending_input"] = None

    # Auto-generate title and register in SQLite on first turn
    is_first_message = (len(st.session_state["message_history"]) == 0)
    if is_first_message:
        thread_title = generate_title_from_text(user_input)
        save_thread_metadata(st.session_state["thread_id"], title=thread_title, model=selected_model)

    # Save user message to session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Render user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Prepare streaming config
    run_config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"],
            "model_name": selected_model,
            "temperature": temperature,
            "system_prompt": system_prompt,
            "api_key": custom_key.strip() if custom_key.strip() else None
        }
    }

    # Generator for streaming chunks
    def stream_response(text: str):
        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=text)]},
            config=run_config,
            stream_mode="messages"
        ):
            chunk_text = extract_text(message_chunk.content)
            if chunk_text:
                yield chunk_text

    # Render assistant response with streaming
    with st.chat_message("assistant", avatar="🤖"):
        try:
            ai_response = st.write_stream(stream_response(user_input))

            # Save assistant response
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": ai_response
            })

            # Update updated_at timestamp in database
            save_thread_metadata(st.session_state["thread_id"], model=selected_model)

            # Rerun on first message so the sidebar immediately displays the new chat title
            if is_first_message:
                st.rerun()

        except Exception as e:
            st.error(f"⚠️ **Error generating response:** {e}")
            st.info("💡 **Tips to resolve:**\n- Verify that your `GOOGLE_API_KEY` is valid.\n- Check your internet connection.\n- Ensure you haven't exceeded Google Gemini API rate limits.")
