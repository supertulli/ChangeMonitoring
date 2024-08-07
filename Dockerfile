FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    # git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip3 install -r requirements.txt

COPY src /app/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD curl --fail http://localhost:8501/_stcore/HEALTHCHECK

ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0" ]