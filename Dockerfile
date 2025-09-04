FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools for ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-clean.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements-clean.txt

# Copy application files specifically (avoid conflicting app directory)
COPY main.py .
COPY haystack_new.py .
COPY enhanced_ml_processor.py .
COPY improved_indexing_qa.py .
COPY simple_ner_processor.py .
COPY streamlit/ ./streamlit/
COPY rest_api_configs/ ./rest_api_configs/

# Create necessary directories
RUN mkdir -p logs data models cache /root/.cache/huggingface

# Set environment variables for containerized deployment  
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HOME=/root/.cache/huggingface

# Use Ollama running on host machine
ENV SKIP_OPENSEARCH=true
ENV SKIP_REDIS=true
ENV USE_OLLAMA=true
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV DOCKER_ENV=true

# Expose ports
EXPOSE 8001 8505

# Health check for API
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the API application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
