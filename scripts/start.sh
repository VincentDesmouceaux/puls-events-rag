#!/bin/sh

set -e

cd /app

if [ ! -f "data/faiss_index/index.faiss" ] || \
   [ ! -f "data/faiss_index/index.pkl" ]; then
    echo "Index FAISS absent : reconstruction..."
    .venv/bin/python -m scripts.build_faiss_index
else
    echo "Index FAISS existant : reconstruction non nécessaire."
fi

exec .venv/bin/uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000