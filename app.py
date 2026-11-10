"""
AI Chatbot Application Entry Point.
Delegates to the database-backed Streamlit frontend.
"""
import runpy
from pathlib import Path

target_file = Path(__file__).parent / "streamlit_frontend_database.py"
runpy.run_path(str(target_file), run_name="__main__")
