"""
server/app.py — re-exports the FastAPI app from main.py and provides a main() entry point.
"""
from main import app  # noqa: F401
import uvicorn

__all__ = ["app"]

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()