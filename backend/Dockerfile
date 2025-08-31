# syntax=docker/dockerfile:1.6

##########################################################
# Download models
##########################################################
FROM --platform=linux/amd64 python:3.11.9-slim-bullseye AS downloader
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# HF client only
RUN python -m pip install --upgrade "pip==25.2" \
 && pip install --no-cache-dir "huggingface_hub>=0.23,<0.27"

# Destination for models
RUN mkdir -p /opt/models/vit-fer

# Robust snapshot with retries (no hf_transfer; no apt)
RUN --mount=type=cache,target=/root/.cache/huggingface python - <<'PY'
import os, sys, time, glob, traceback
from huggingface_hub import snapshot_download

# avoid hf_transfer
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

DST = "/opt/models/vit-fer"
REPO = "mo-thecreator/vit-Facial-Expression-Recognition"
MAX_ATTEMPTS = 5
delay = 5

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        print(f"[HF] Download attempt {attempt} -> {REPO}")
        snapshot_download(
            repo_id=REPO,
            local_dir=DST,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        break
    except Exception as e:
        print(f"[HF] attempt {attempt} failed: {e}", file=sys.stderr)
        traceback.print_exc()
        if attempt == MAX_ATTEMPTS:
            sys.exit(1)
        time.sleep(delay)
        delay = min(delay * 2, 60)

files = [p for p in glob.glob(DST + "/**/*", recursive=True) if os.path.isfile(p)]
print("Downloaded files:", len(files))
if len(files) < 3:
    print("ERROR: No model files downloaded to", DST, file=sys.stderr)
    sys.exit(1)
PY


##########################################################
# Final runtime image
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

# System dependencies
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list \
 && apt-get update -o Acquire::Retries=5 \
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

# Bring in the predownloaded models
COPY --from=downloader /opt/models/vit-fer /opt/models/vit-fer

# Validate presence in final image (fail if empty)
RUN set -e; \
    echo "Validating model presence in final image..."; \
    find /opt/models/vit-fer -type f -size +0c | tee /tmp/final_model_files.txt; \
    [ "$(wc -l </tmp/final_model_files.txt)" -gt 3 ] || { echo "Model files missing in final image."; exit 1; }

# App code
COPY . /app

ENV PORT=8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info --use-colors"]