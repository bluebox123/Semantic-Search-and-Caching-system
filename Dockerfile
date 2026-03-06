# ============================================================
# Dockerfile — Semantic Search & Caching System
# ============================================================
# Uses a slim Python base image. Pre-downloads the HuggingFace
# model during build so the container doesn't re-download on
# every startup. This makes cold starts significantly faster.
# ============================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System deps for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformer model during build
# so the container doesn't download it on every start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
