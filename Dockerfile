FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY scripts ./scripts
COPY config ./config
COPY examples ./examples

RUN useradd --create-home --uid 10001 alphaops \
    && mkdir -p /app/data /app/logs \
    && chown -R alphaops:alphaops /app/data /app/logs \
    && chmod +x scripts/run_daily.sh

USER alphaops
ENV ALPHAOPS_DB=/app/data/alphaops.db \
    ALPHAOPS_METRICS_DIR=/app/data/metrics

HEALTHCHECK --interval=1h --timeout=30s CMD alphaops health >/dev/null || exit 1
ENTRYPOINT ["alphaops"]
CMD ["--help"]
