FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies (CPU-only torch)
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models at build time (avoids runtime HF Hub rate limits)
RUN python -c "\
from transformers import pipeline; \
pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None, device=-1); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Models downloaded successfully')"

# Copy application code
COPY . .

# Create data directory for cache/history
RUN mkdir -p data/history

# HuggingFace Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER 1000

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
