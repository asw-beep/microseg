# Microscopy nuclei segmentation & quantification pipeline.
#
# CPU-only image by design: the pipeline runs at interactive speed on CPU, and a
# CUDA base image would add ~2.5 GB for a demo most people will run on a laptop.
# To build a GPU image, swap the torch install line for the cu121 index.
#
#   docker build -t microseg .
#   docker run -p 7860:7860 microseg                       # Gradio app
#   docker run -v "$PWD/data:/app/data" microseg pytest    # tests
#
FROM python:3.11-slim

# OpenCV needs libGL and libglib even in headless use; git is needed only if a
# dependency resolves to a VCS URL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MPLBACKEND=Agg

# Install torch from the CPU index first, in its own layer: it is the largest
# and slowest-changing dependency, so it stays cached across source edits.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source last, so editing code does not invalidate the dependency layers.
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

# Mount points for data and results.
RUN mkdir -p data/raw data/processed outputs weights

EXPOSE 7860

# Defaults to the app; override with any command, e.g.
#   docker run microseg python -m microseg.train --config configs/smoke.yaml
CMD ["python", "app/gradio_app.py", "--host", "0.0.0.0", "--port", "7860"]
