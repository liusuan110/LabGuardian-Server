# ── GPU 版 (CUDA 12.8 runtime + PyTorch cu124 wheels) ─────────────────────
# PyTorch 官方 pip 仓库最高支持 cu124；cu124 wheel 完全兼容 CUDA 12.8 驱动。
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python 3.11 + OpenCV 运行时系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev curl \
        libgl1-mesa-glx libglib2.0-0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 依赖层 ── 先装 PyTorch GPU，再装项目其余依赖 ─────────────────────────
# 顺序不可颠倒：ultralytics 安装时若 torch 未存在会拉 CPU 版覆盖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# stub app/__init__.py 让 hatchling 能构建 wheel，从而把 pyproject.toml
# 里声明的所有依赖装进镜像；真实代码在后续 COPY 层覆盖
COPY pyproject.toml README.md ./
RUN mkdir -p app && touch app/__init__.py && \
    pip install --no-cache-dir . && \
    rm -f app/__init__.py

# ── 应用层 ── 代码变动只重建此层 ─────────────────────────────────────────
COPY app/ app/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
