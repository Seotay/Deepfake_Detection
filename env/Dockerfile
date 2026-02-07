FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# OpenCV runtime deps (libGL.so.1 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY env/requirements.txt /workspace/env/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /workspace/env/requirements.txt

COPY . /workspace
