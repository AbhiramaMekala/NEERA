# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install system dependencies needed for CatBoost, LightGBM, and general building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# Note: This includes the serialized model files and the master database
COPY *.py /app/
COPY weather_providers /app/weather_providers
COPY data /app/data
COPY outputs /app/outputs

# Expose port
EXPOSE 8000

# Health check to ensure service is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Command to run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
