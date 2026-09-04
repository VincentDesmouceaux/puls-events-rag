import os
import secrets
from datetime import datetime, timezone
from threading import Lock

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Security,
    status,
)
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
    version="0.2.2",
)


rebuild_api_key_header = APIKeyHeader(
    name="X-Rebuild-Key",
    auto_error=False,
)


rebuild_lock = Lock()

rebuild_state = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "error": None,
}


def utc_now() -> str:
    """Retourne la date UTC actuelle au format ISO."""
    return datetime.now(timezone.utc).isoformat()


def verify_rebuild_api_key(
    api_key: str | None = Security(rebuild_api_key_header),
) -> None:
    """Vérifie la clé autorisant la reconstruction de l'index."""
    expected_api_key = os.getenv("REBUILD_API_KEY")

    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "La protection de l'endpoint rebuild "
                "n'est pas configurée."
            ),
        )

    if not api_key or not secrets.compare_digest(
        api_key,
        expected_api_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Clé d'accès invalide.",
        )


def run_faiss_rebuild() -> None:
    """Reconstruit FAISS en arrière-plan et met à jour son état."""
    try:
        rebuild_faiss_index()
        clear_retriever_cache()

        with rebuild_lock:
            rebuild_state["status"] = "completed"
            rebuild_state["completed_at"] = utc_now()
            rebuild_state["error"] = None

        print("Rebuild FAISS terminé avec succès.")

    except Exception as exc:
        with rebuild_lock:
            rebuild_state["status"] = "failed"
            rebuild_state["completed_at"] = utc_now()
            rebuild_state["error"] = str(exc)

        print(
            f"Erreur rebuild : "
            f"{type(exc).__name__}: "
            f"{exc}"
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
        "version": "0.2.2",
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
            "description": (
                "Service de génération temporairement indisponible."
            ),
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
    summary="Lance la reconstruction de FAISS en arrière-plan",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[
        Security(verify_rebuild_api_key),
    ],
    responses={
        202: {
            "description": "Reconstruction lancée en arrière-plan.",
        },
        401: {
            "description": "Clé d'accès absente ou invalide.",
        },
        409: {
            "description": "Une reconstruction est déjà en cours.",
        },
        503: {
            "description": "Configuration de sécurité indisponible.",
        },
    },
)
def rebuild(
    background_tasks: BackgroundTasks,
) -> dict:
    """Programme une reconstruction FAISS sans bloquer la requête."""
    with rebuild_lock:
        if rebuild_state["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail="Une reconstruction FAISS est déjà en cours.",
            )

        rebuild_state["status"] = "running"
        rebuild_state["started_at"] = utc_now()
        rebuild_state["completed_at"] = None
        rebuild_state["error"] = None

    background_tasks.add_task(run_faiss_rebuild)

    return {
        "status": "accepted",
        "message": (
            "Reconstruction FAISS lancée en arrière-plan."
        ),
    }


@app.get(
    "/rebuild/status",
    tags=["Index"],
    summary="Consulte l'état de la reconstruction FAISS",
)
def rebuild_status() -> dict:
    """Retourne l'état courant de la reconstruction FAISS."""
    with rebuild_lock:
        return dict(rebuild_state)