import os
import secrets

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.schemas import AskRequest, AskResponse
from scripts.build_faiss_index import main as rebuild_faiss_index
from scripts.rag_chain import (
    answer_question,
    clear_retriever_cache,
)


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API REST exposant le système RAG "
        "de recommandation d'événements."
    ),
    version="0.2.1",
)


rebuild_api_key_header = APIKeyHeader(
    name="X-Rebuild-Key",
    auto_error=False,
)


def verify_rebuild_api_key(
    api_key: str | None = Security(rebuild_api_key_header),
) -> None:
    """Vérifie la clé autorisant la reconstruction de l'index."""
    expected_api_key = os.getenv("REBUILD_API_KEY")

    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail="La protection de l'endpoint rebuild n'est pas configurée.",
        )

    if not api_key or not secrets.compare_digest(
        api_key,
        expected_api_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Clé d'accès invalide.",
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
        "version": "0.2.1",
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["RAG"],
    summary="Pose une question au système RAG",
    responses={
        500: {
            "description": "Erreur interne du système RAG.",
        },
        503: {
            "description": "Service de génération temporairement indisponible.",
        },
    },
)
def ask(request: AskRequest) -> AskResponse:
    """Retourne une réponse augmentée à partir des événements FAISS."""
    try:
        answer = answer_question(request.question)

    except Exception as exc:
        error_message = str(exc)

        print(
            f"Erreur RAG : "
            f"{type(exc).__name__}: "
            f"{error_message}"
        )

        if (
            "429" in error_message
            or "Rate limit exceeded" in error_message
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Le service de génération est temporairement "
                    "indisponible. Réessayez dans quelques instants."
                ),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="Impossible de générer une réponse RAG.",
        ) from exc

    return AskResponse(
        question=request.question,
        answer=answer,
    )


@app.post(
    "/rebuild",
    tags=["Index"],
    summary="Reconstruit la base vectorielle FAISS",
    dependencies=[
        Security(verify_rebuild_api_key),
    ],
    responses={
        401: {
            "description": "Clé d'accès absente ou invalide.",
        },
        500: {
            "description": "Erreur lors de la reconstruction de FAISS.",
        },
        503: {
            "description": "Configuration de sécurité indisponible.",
        },
    },
)
def rebuild() -> dict:
    """Recharge les événements et reconstruit l'index FAISS."""
    try:
        rebuild_faiss_index()
        clear_retriever_cache()

    except Exception as exc:
        print(
            f"Erreur rebuild : "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Impossible de reconstruire l'index FAISS.",
        ) from exc

    return {
        "status": "ok",
        "message": "Index FAISS reconstruit avec succès.",
    }