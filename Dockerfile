# Use Python slim image for Cloud Run deployment
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

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU first (separate step to avoid conflicts)
RUN pip install --no-cache-dir torch>=2.0.0,<2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies from PyPI
RUN pip install --no-cache-dir \
    fastapi>=0.104.1,<0.110.0 \
    uvicorn[standard]>=0.24.0,<0.30.0 \
    python-multipart>=0.0.6,<0.1.0 \
    pydantic>=2.4.0,<2.7.0 \
    transformers>=4.35.0,<4.40.0 \
    datasets>=2.14.0,<2.20.0 \
    accelerate>=0.24.0,<0.30.0 \
    pandas>=2.0.0,<2.3.0 \
    numpy>=1.24.0,<1.27.0

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p static logs data

# Set permissions
RUN chmod +x *.py

# Expose port
EXPOSE 8080

# Health check with longer timeout for model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "app.py"]
