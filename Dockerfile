FROM python:3.13-slim

WORKDIR /app

# Install uv for fast deterministic dependency installation
RUN pip install --no-cache-dir uv

# Install dependencies into /app/.venv first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source files (.venv and .env excluded via .dockerignore)
COPY . .

ENV PORT=8080
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
