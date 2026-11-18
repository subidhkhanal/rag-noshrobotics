# Full profile: BM25 + bge-small + cross-encoder rerank. For the serverless fast profile, deploy to Vercel instead.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/.models FORGE_PROFILE=full
WORKDIR /app

COPY requirements.txt requirements-full.txt ./
RUN pip install --no-cache-dir -r requirements-full.txt

# Bake the ONNX models into the image so cold starts don't download them.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY forge ./forge
COPY data/ontology ./data/ontology
COPY data/processed ./data/processed
COPY data/models ./data/models
COPY public ./public

RUN useradd -m app && chown -R app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn forge.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
