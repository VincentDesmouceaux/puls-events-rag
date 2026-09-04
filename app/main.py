from fastapi import FastAPI


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API REST exposant le système RAG "
        "de recommandation d'événements."
    ),
    version="0.1.0",
)


@app.get(
    "/health",
    tags=["Health"],
    summary="Vérifie l'état de l'API",
)
def health() -> dict:
    """Retourne l'état de fonctionnement de l'API."""
    return {
        "status": "ok",
        "service": "puls-events-rag-api",
        "version": "0.1.0",
    }