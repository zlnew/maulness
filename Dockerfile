FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specifications
COPY pyproject.toml README.md ./

# Install dependencies into system environment
RUN uv pip install --system --no-cache -e .

# Copy source code
COPY src/ ./src/

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "maulness.daemon"]
