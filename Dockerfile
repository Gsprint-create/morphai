FROM python:3.10-slim

WORKDIR /app

# system deps often needed for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
