from fastapi import FastAPI, HTTPException

from app.schemas import AskRequest, AskResponse
from scripts.rag_chain import answer_question


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


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["RAG"],
    summary="Pose une question au système RAG",
)
def ask(request: AskRequest) -> AskResponse:
    """Retourne une réponse augmentée à partir des événements FAISS."""
    try:
        answer = answer_question(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Impossible de générer une réponse RAG.",
        ) from exc

    return AskResponse(
        question=request.question,
        answer=answer,
    )