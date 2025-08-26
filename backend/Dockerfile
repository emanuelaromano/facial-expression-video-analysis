# syntax=docker/dockerfile:1.6

##########################################################
# Download models
##########################################################
FROM --platform=$BUILDPLATFORM python:3.11.9-slim-bullseye@sha256:fe8df05de0fe659668a0b6fa204c652092e3e96832a9c7d6f8cc1e0e03e663d8 AS downloader
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN python -m pip install --upgrade "pip==25.2" \
 && pip install --no-cache-dir huggingface_hub

RUN mkdir -p /opt/models/vit-fer
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="mo-thecreator/vit-Facial-Expression-Recognition",
    local_dir="/opt/models/vit-fer",
    local_dir_use_symlinks=False
)
PY

##########################################################
# Final image
##########################################################
FROM --platform=linux/amd64 python:3.11.9-slim-bullseye@sha256:fe8df05de0fe659668a0b6fa204c652092e3e96832a9c7d6f8cc1e0e03e663d8

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_OFFLINE=1 \
    TRANSFORMERS_NO_TF=1 \
    TRANSFORMERS_NO_FLAX=1 \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=

# System deps (trim -dev libs if not needed)
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential git \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
      ffmpeg \
      libjpeg-dev libpng-dev libtiff-dev \
      tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade "pip==25.2" \
 && pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 \
 && apt-get purge -y build-essential git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Bring in the predownloaded model
COPY --from=downloader /opt/models/vit-fer /opt/models/vit-fer

# App code
COPY . /app

ENV PORT=8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --no-access-log --log-level info"]
