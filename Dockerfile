# # app/Dockerfile
# FROM python:3.10-slim-trixie
# COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/


# WORKDIR /app
# COPY . /app
# RUN apt-get update && apt-get install -y build-essential git curl && rm -rf /var/lib/apt/lists/*
# RUN uv sync --locked


# An example using multi-stage image builds to create a final image without uv.

# First, build the application in the `/app` directory.


FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder
RUN apt-get update && apt-get install -y build-essential git curl && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy



# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev


# Then, use a final image without uv
FROM python:3.10-slim-bookworm
RUN apt-get update && apt-get install -y build-essential git curl && rm -rf /var/lib/apt/lists/*
# It is important to use the image that matches the builder, as the path to the
# Python executable must be the same, e.g., using `python:3.11-slim-bookworm`
# will fail.

# Setup a non-root user
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# Copy the application from the builder
COPY --from=builder --chown=nonroot:nonroot /app /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Use the non-root user to run our application
USER nonroot

# Use `/app` as the working directory
WORKDIR /app

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "JR_3_website.py", "--server.address=0.0.0.0", "--server.port=8501"]

