FROM alpine:3.23 AS python-builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PYTHON_PREFERENCE=only-managed \
    VIRTUAL_ENV=/opt/venv

WORKDIR /app

RUN apk add --no-cache git

COPY ./src ./LICENSE ./README.md /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,ro \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,ro \
    uv sync --frozen --no-dev

FROM alpine:3.23

RUN apk add --no-cache ca-certificates tzdata

LABEL maintainer="Elias Benbourenane <eliasbenbourenane@gmail.com>" \
    org.opencontainers.image.title="llm-shim" \
    org.opencontainers.image.description="A minimal OpenAI-compatible API shim for LLMs." \
    org.opencontainers.image.authors="Elias Benbourenane <eliasbenbourenane@gmail.com>" \
    org.opencontainers.image.url="https://github.com/eliasbenb/llm-shim" \
    org.opencontainers.image.documentation="https://github.com/eliasbenb/llm-shim" \
    org.opencontainers.image.source="https://github.com/eliasbenb/llm-shim" \
    org.opencontainers.image.licenses="MIT"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHON_JIT=1 \
    LLM_SHIM_DATA_DIR=/config

WORKDIR /app

COPY --from=python-builder /python /python
COPY --from=python-builder /opt/venv /opt/venv

COPY . /app

RUN mkdir -p /config

VOLUME ["/config"]

EXPOSE 8000

CMD ["python", "/app/main.py"]
