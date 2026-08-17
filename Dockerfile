# Application image for the PEA extraction/aggregation pipeline + DB migrator.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# System libs for psycopg2 / pyarrow builds
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry>=2.0,<3.0"

# Resolve + install runtime deps only (re-locks so freshly-added deps are picked up)
COPY pyproject.toml poetry.lock* ./
RUN poetry lock && poetry install --no-root --only main --no-interaction

COPY . .

# Default: load existing flat files into the DB. Override in compose/CLI as needed.
CMD ["python", "-m", "scripts.migrate_parquet_to_db"]
