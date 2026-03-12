FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_SIZE=5
ENV ACTIVE_MODELS=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy only requirements first to cache dependencies
COPY requirements.txt /app/

# 2. Install large dependencies (only reruns if requirements.txt changes)
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchaudio \
    && python3 -m pip install "faster-qwen3-tts[demo]" \
    && python3 -m pip install -r requirements.txt

# 3. Copy the rest of the app (changes to app.py won't break the pip cache)
COPY . /app

EXPOSE 7860
CMD ["python3", "app.py"]
