# app/Dockerfile
FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/


WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*
RUN uv sync --locked


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["uv", "run", "streamlit", "run", "JR_3_website.py", "--server.address=0.0.0.0", "--server.port=8501"]