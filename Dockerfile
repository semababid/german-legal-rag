# The API and the embedding model. Qdrant runs as its own service and the
# language model stays on the host, so this image holds Python, the retrieval
# stack, and nothing else.
FROM python:3.13-slim

# curl, for the healthcheck. Nothing else: every wheel this needs has an
# aarch64 build, so there is no compiler in the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch and sentence-transformers are most of the build and almost never change,
# so requirements are copied and installed before the source. Editing a Python
# file then rebuilds in seconds rather than re-resolving the whole stack.
COPY requirements.txt .

# torch comes from PyTorch's CPU index, before anything else can pull the
# default wheel. The default one carries CUDA: 2.9 GB of nvidia libraries and
# 650 MB of triton, none of which a container without a GPU can use. Installing
# it that way made the image 9.09 GB. Nothing here is faster for it, since the
# language model runs on the host and the embedder runs on CPU either way.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# bge-m3 is about 2.3 GB. Baking it in would make the image enormous and pin it
# to one embedding model, so it downloads on first start into a mounted cache
# and survives rebuilds.
ENV HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Loading the model takes about ten seconds, so the start period has to cover it
# or the container is declared unhealthy while it is doing exactly what it should.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health | grep -q '"ready":true' || exit 1

CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
