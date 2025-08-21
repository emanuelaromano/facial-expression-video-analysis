FROM python:3.10-slim-bullseye

# Fix HTTPS source issue
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    tesseract-ocr \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Pre-copy only requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies first
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now copy app code
COPY . .

# Run app
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=3000 --server.address=0.0.0.0"]
