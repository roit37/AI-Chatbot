# 🤖 Gemini AI Chatbot with LangGraph & SQLite Persistence

A modern, full-featured AI Chatbot built with **LangGraph**, **Google Gemini**, and **Streamlit**. It features persistent multi-turn conversational memory powered by SQLite checkpointers, customizable AI personas, conversation search, rename, export, and deletion capabilities.

---

## ✨ Features

- **🧠 Persistent Multi-Turn Memory**: Uses LangGraph's `SqliteSaver` checkpointer to preserve every conversation and message across app reloads.
- **⚡ Google Gemini Integration**: Supports the latest Gemini models (`gemini-3.8-flash`, `gemini-3.5-flash-lite`, and `gemini-3.1-pro-preview`).
- **💬 Conversation Management**:
  - **Auto-generated conversation titles** derived from the initial user prompt.
  - **Inline Chat Renaming**: Edit titles directly from the chat header.
  - **Search & Filter**: Quickly find past conversations by keyword.
  - **Delete Conversations**: Safely delete single or all conversations.
  - **Exporting**: Download any conversation as Markdown (`.md`) or structured JSON (`.json`).
- **🎭 AI Personas & Roles**:
  - 🤖 *General Assistant*
  - 👨‍💻 *Senior Software Engineer & Code Reviewer*
  - 🎓 *Patient Teacher & Tutor*
  - 📝 *Concise & Fluff-Free*
  - 🧠 *Creative Brainstormer*
  - ✏️ *Custom System Instructions*
- **🚀 Real-Time Token Streaming**: Fast token-by-token streaming using Streamlit's `st.write_stream`.
- **💡 Starter Prompt Cards**: Interactive cards for new conversations to spark ideas and jumpstart discussions.
- **🎨 Polished Modern UI**: Sleek badges, clean chat layout, responsive sidebar, and custom CSS styling.

---

## 📁 Project Structure

```
AIChatbot/
├── app.py                         # Application entrypoint
├── streamlit_frontend_database.py # Primary Streamlit UI with SQLite persistence
├── langgraph_database_backend.py  # LangGraph state graph, checkpointer, and DB logic
├── langgraph_backend.py           # In-memory LangGraph backend prototype
├── streamlit_frontend.py          # In-memory Streamlit frontend prototype
├── chatbot.db                     # SQLite database (checkpoints, writes, metadata)
├── requirements.txt               # Python package dependencies
├── .env                           # Environment variables (API keys)
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Activate your existing virtual environment:

```bash
# On macOS / Linux
source myenv/bin/activate
```

Or create a new one:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key

Ensure your `.env` file contains your Google Gemini API key:

```env
GOOGLE_API_KEY="your-gemini-api-key-here"
```

*(You can also supply or override your API key at runtime in the sidebar settings.)*

### 3. Run the Chatbot

Launch the application with Streamlit:

```bash
streamlit run app.py
```
*(Or `streamlit run streamlit_frontend_database.py`)*

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🛠️ Architecture Overview

1. **State Graph**: Defined with `langgraph.graph.StateGraph` using a state dictionary containing `messages: Annotated[list[BaseMessage], add_messages]`.
2. **Checkpointer**: Backed by `langgraph.checkpoint.sqlite.SqliteSaver` in `chatbot.db`. Each user conversation has a unique `thread_id` UUID.
3. **Dynamic Invocation**: `chat_node` dynamically binds the selected model, temperature, and persona system instructions per request using `RunnableConfig`.
4. **Metadata Store**: A SQLite table `thread_metadata` tracks `title`, `created_at`, `updated_at`, and `model` for fast indexing, sorting, and user renaming.
