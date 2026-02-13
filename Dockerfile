FROM python:3.10-slim

WORKDIR /app

# Install compiler + libs needed for insightface build and opencv runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
  && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV PORT=8000
CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
