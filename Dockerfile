FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/.models
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ONNX models into the image so cold starts don't download them.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY forge ./forge
COPY data/ontology ./data/ontology
COPY data/processed ./data/processed
COPY web ./web

RUN useradd -m app && chown -R app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn forge.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
