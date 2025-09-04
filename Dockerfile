FROM python:3.9-slim

# Install system dependencies for OpenCV and dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    python3-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory structure
RUN mkdir -p models/anti_spoofing

# Download models at build time
RUN python download_models.py

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
