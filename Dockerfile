FROM python:3.14-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
COPY static ./static
COPY scripts ./scripts

ENV HOST=0.0.0.0 PORT=8000 DATA_DIR=/data
VOLUME /data
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
