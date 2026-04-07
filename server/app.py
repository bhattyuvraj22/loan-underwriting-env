"""
server/app.py — re-exports the FastAPI app from main.py.
The real application lives in main.py; this file exists for compatibility.
"""
from main import app  # noqa: F401 — re-export for any tooling that imports server.app

__all__ = ["app"]