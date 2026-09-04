#!/bin/sh

set -e

if [ ! -f "data/faiss_index/index.faiss" ] || \
   [ ! -f "data/faiss_index/index.pkl" ]; then
    echo "Index FAISS absent : reconstruction..."
    uv run python scripts/build_faiss_index.py
else
    echo "Index FAISS existant : reconstruction non nécessaire."
fi

exec uv run uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000