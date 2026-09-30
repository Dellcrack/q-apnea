# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for C-extensions if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, models, results, and web assets
COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/
COPY data/ ./data/
COPY public/ ./public/
COPY server.py test_inference.py test_api_endpoints.py README.md ./

# Expose default port
EXPOSE 8000

# Environment variables for cloud deployment
ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Start the web server
CMD ["python", "server.py"]
