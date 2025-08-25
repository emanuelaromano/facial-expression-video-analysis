FROM python:3.11-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (single layer, HTTPS mirrors)
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential git \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
      ffmpeg libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
      libjpeg-dev libpng-dev libtiff-dev \
      tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Copy only requirements for a stable, cacheable deps layer
COPY requirements.txt /app/requirements.txt

# 2) Install Python deps
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# 3) (Important) Pre-download HF model *before* app code so code edits don't invalidate this cache
RUN python - <<'PY'
from transformers import AutoImageProcessor, AutoModelForImageClassification
name = "mo-thecreator/vit-Facial-Expression-Recognition"
AutoImageProcessor.from_pretrained(name)
AutoModelForImageClassification.from_pretrained(name)
PY

# 4) Now copy the app code (later layers only)
COPY . /app

# 5) Entrypoint
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --no-access-log --log-level info"]
