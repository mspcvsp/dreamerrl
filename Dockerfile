FROM ghcr.io/astral-sh/uv:0.13.9 AS uv
FROM python:3.10.19-slim-bookworm
COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Submodule contents must be initialized in the build context. Generated runs,
# virtual environments, cached wheels and the historical worktree are excluded.
COPY . .
RUN uv python install 3.10.19 \
    && uv run --no-project --managed-python --python 3.10.19 python environments/cage2/prepare_native_wheel.py \
    && uv sync --locked
CMD ["uv", "run", "--locked", "python", "scripts/verify_uv.py"]
