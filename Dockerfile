FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir . && \
    python -m spacy download en_core_web_sm

# Application code
COPY polymarket_geo/ polymarket_geo/

# Default: start API server + scheduler
CMD ["python", "-m", "polymarket_geo", "serve"]
