FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libx11-6 \
        libxcb1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip uninstall -y torch torchvision \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.2.1+cu121" \
        "torchvision==0.17.1+cu121" \
    && python -m pip install --no-cache-dir \
        "pyiqa==0.1.14.1" \
        --no-deps \
    && python -m pip install --no-cache-dir \
        "numpy<2" \
        "scipy<1.15" \
        "pandas" \
        "timm" \
        "pillow" \
        "opencv-python-headless" \
        "scikit-image" \
        "joblib" \
        "tqdm" \
        "yacs" \
        "natsort" \
        "transformers==4.37.2" \
        "accelerate<=1.1.0" \
        "einops" \
        "sentencepiece" \
        "safetensors" \
        "huggingface-hub<1.0" \
        "datasets" \
        "openai-clip" \
        "facexlib" \
        "lmdb" \
        "addict" \
        "icecream" \
        "protobuf" \
        "bitsandbytes" \
        "future" \
        "pre-commit" \
        "pytest" \
        "ruff" \
        "tensorboard" \
        "yapf"
