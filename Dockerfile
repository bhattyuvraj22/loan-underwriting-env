FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (venv, .git, __pycache__ excluded via .dockerignore)
COPY . .

EXPOSE 7860

# Health check so orchestrators know when the app is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
