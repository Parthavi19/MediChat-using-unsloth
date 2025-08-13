# Use Python slim image instead of CUDA (Cloud Run doesn't have GPUs)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install only essential dependencies for Cloud Run
RUN pip install --no-cache-dir \
    fastapi>=0.104.1 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    pydantic>=2.4.0 \
    torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu \
    transformers>=4.35.0 \
    datasets>=2.14.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static logs

# Set permissions
RUN chmod +x *.py

# Expose port
EXPOSE 8080

# Health check with longer timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "app.py"]
