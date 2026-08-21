# Image for Hugging Face Spaces. Differs from the local one in three ways that
# the platform forces: it listens on 7860, it runs as an unprivileged user, and
# it carries the prebuilt Qdrant index instead of building one at startup.
#
# That last point is the interesting one. A Space gets two cores and no GPU.
# Embedding 4,921 records took about thirty five minutes in a ten-core container
# with no Metal, so doing it during startup would leave the Space unreachable
# for hours after every restart. The index is 49 MB and is copied in instead.

FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as uid 1000. Anything the process writes to has to
# belong to that user, including the model cache, or startup fails on a
# permission error that reads like a download problem.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .

# CPU-only torch. The default wheel carries 3.5 GB of CUDA libraries that a
# Space without a GPU cannot use, and pulling them makes every cold start slower
# for nothing.
RUN pip install --no-cache-dir --user torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user src/ ./src/
COPY --chown=user web/ ./web/
COPY --chown=user data/processed/ ./data/processed/
COPY --chown=user qdrant_storage/ ./qdrant_storage/

# Retrieval only unless a generator is configured. See the note in api.py: a
# model small enough to run here would answer worse than the retrieval it sits
# on, so the deployment shows the provisions and says no answer was written.
# Adding ANTHROPIC_API_KEY and BACKEND=anthropic as Space secrets turns
# generation on with no other change.
ENV BACKEND=none \
    COLLECTION=absatz_ctx \
    RETRIEVER=domain \
    TOP_K=6 \
    LOG_QUESTIONS=false

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:7860/health | grep -q '"ready":true' || exit 1

CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "7860"]
