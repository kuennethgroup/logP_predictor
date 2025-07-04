# app/Dockerfile

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./ ./

RUN git clone https://github.com/autogluon/autogluon && \
    cd autogluon && \
    git checkout v1.1.1 && \
    pip install -e core/[all] -e features/ -e tabular/[all] -e autogluon/
 
RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "JR_3_website.py", "--server.port=8501", "--server.address=0.0.0.0"]
