import json
import os
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Security,
    status,
)
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader

from app.schemas import AskRequest, AskResponse
from scripts.build_faiss_index import (
    load_faiss_index,
    main as rebuild_faiss_index,
)
from scripts.rag_chain import (
    answer_question,
    clear_retriever_cache,
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

API_VERSION = "0.2.5"

FAISS_INDEX_DIR = Path("data/faiss_index")

RAGAS_RESULTS_PATH = Path(
    "data/evaluation/ragas_results.json"
)


# -------------------------------------------------------------------
# Application FastAPI
# -------------------------------------------------------------------

app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API REST exposant le système RAG "
        "de recommandation d'événements."
    ),
    version=API_VERSION,
)


# -------------------------------------------------------------------
# Sécurité de l'endpoint /rebuild
# -------------------------------------------------------------------

rebuild_api_key_header = APIKeyHeader(
    name="X-Rebuild-Key",
    auto_error=False,
)


# -------------------------------------------------------------------
# Locks
# -------------------------------------------------------------------

rebuild_lock = Lock()
metrics_lock = Lock()


# -------------------------------------------------------------------
# État de la reconstruction FAISS
# -------------------------------------------------------------------

rebuild_state = {
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "duration_seconds": None,
    "error": None,
    "progress": 0,
    "step": "En attente",
    "processed": None,
    "total": None,
}


# -------------------------------------------------------------------
# Métriques du système RAG
# -------------------------------------------------------------------

rag_metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "total_response_time_ms": 0.0,
    "last_response_time_ms": None,
    "last_request_at": None,
}


# -------------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------------

def utc_now() -> str:
    """Retourne la date UTC actuelle au format ISO."""

    return datetime.now(
        timezone.utc
    ).isoformat()


def verify_rebuild_api_key(
    api_key: str | None = Security(
        rebuild_api_key_header
    ),
) -> None:
    """Vérifie la clé autorisant la reconstruction FAISS."""

    expected_api_key = os.getenv(
        "REBUILD_API_KEY"
    )

    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "La protection de l'endpoint rebuild "
                "n'est pas configurée."
            ),
        )

    if (
        not api_key
        or not secrets.compare_digest(
            api_key,
            expected_api_key,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Clé d'accès invalide.",
        )


def record_rag_request(
    *,
    success: bool,
    duration_ms: float,
) -> None:
    """Enregistre les métriques d'une requête RAG."""

    with metrics_lock:
        rag_metrics["requests_total"] += 1

        if success:
            rag_metrics["requests_success"] += 1
        else:
            rag_metrics["requests_failed"] += 1

        rag_metrics[
            "total_response_time_ms"
        ] += duration_ms

        rag_metrics[
            "last_response_time_ms"
        ] = round(
            duration_ms,
            2,
        )

        rag_metrics[
            "last_request_at"
        ] = utc_now()


def update_rebuild_progress(
    progress: int,
    step: str,
    processed: int | None = None,
    total: int | None = None,
) -> None:
    """Met à jour la progression de la reconstruction FAISS."""

    bounded_progress = max(
        0,
        min(
            int(progress),
            100,
        ),
    )

    with rebuild_lock:
        rebuild_state["progress"] = bounded_progress
        rebuild_state["step"] = step
        rebuild_state["processed"] = processed
        rebuild_state["total"] = total


def run_faiss_rebuild() -> None:
    """Reconstruit FAISS en arrière-plan."""

    started = time.perf_counter()

    try:
        rebuild_faiss_index(
            progress_callback=update_rebuild_progress
        )

        clear_retriever_cache()

        duration = (
            time.perf_counter()
            - started
        )

        with rebuild_lock:
            rebuild_state["status"] = "completed"
            rebuild_state["completed_at"] = utc_now()

            rebuild_state[
                "duration_seconds"
            ] = round(
                duration,
                2,
            )

            rebuild_state["error"] = None
            rebuild_state["progress"] = 100
            rebuild_state[
                "step"
            ] = "Reconstruction FAISS terminée"

        print(
            "Rebuild FAISS terminé avec succès."
        )

    except Exception as exc:
        duration = (
            time.perf_counter()
            - started
        )

        with rebuild_lock:
            rebuild_state["status"] = "failed"
            rebuild_state["completed_at"] = utc_now()

            rebuild_state[
                "duration_seconds"
            ] = round(
                duration,
                2,
            )

            rebuild_state["error"] = str(exc)

            rebuild_state[
                "step"
            ] = "Échec de la reconstruction FAISS"

        print(
            f"Erreur rebuild : "
            f"{type(exc).__name__}: "
            f"{exc}"
        )


# ===================================================================
# ROOT
# ===================================================================

@app.get(
    "/",
    include_in_schema=False,
)
def root() -> RedirectResponse:
    """
    Redirige automatiquement la racine de l'API
    vers la documentation Swagger.
    """

    return RedirectResponse(
        url="/docs",
        status_code=307,
    )


# ===================================================================
# HEALTH
# ===================================================================

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
        "version": API_VERSION,
        "timestamp": utc_now(),
    }


# ===================================================================
# RAG
# ===================================================================

@app.post(
    "/ask",
    response_model=AskResponse,
    tags=["RAG"],
    summary="Pose une question au système RAG",
    responses={
        500: {
            "description": (
                "Erreur interne du système RAG."
            ),
        },
        503: {
            "description": (
                "Service de génération temporairement "
                "indisponible."
            ),
        },
    },
)
def ask(
    request: AskRequest,
) -> AskResponse:
    """Retourne une réponse augmentée depuis FAISS."""

    started = time.perf_counter()

    try:
        answer = answer_question(
            request.question
        )

    except Exception as exc:
        duration_ms = (
            time.perf_counter()
            - started
        ) * 1000

        record_rag_request(
            success=False,
            duration_ms=duration_ms,
        )

        error_message = str(exc)

        print(
            f"Erreur RAG : "
            f"{type(exc).__name__}: "
            f"{error_message}"
        )

        if (
            "429" in error_message
            or "Rate limit exceeded"
            in error_message
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Le service de génération est "
                    "temporairement indisponible. "
                    "Réessayez dans quelques instants."
                ),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible de générer "
                "une réponse RAG."
            ),
        ) from exc

    duration_ms = (
        time.perf_counter()
        - started
    ) * 1000

    record_rag_request(
        success=True,
        duration_ms=duration_ms,
    )

    return AskResponse(
        question=request.question,
        answer=answer,
    )


# ===================================================================
# MONITORING
# ===================================================================

@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Expose les métriques du système RAG",
)
def metrics() -> dict:
    """Retourne les statistiques des requêtes RAG."""

    with metrics_lock:
        current = dict(rag_metrics)

    total = current["requests_total"]
    success = current["requests_success"]

    if total:
        success_rate = round(
            (
                success
                / total
            )
            * 100,
            2,
        )

        average_response_time_ms = round(
            current[
                "total_response_time_ms"
            ]
            / total,
            2,
        )

    else:
        success_rate = 0.0
        average_response_time_ms = 0.0

    return {
        "requests_total": total,
        "requests_success": success,
        "requests_failed": (
            current["requests_failed"]
        ),
        "success_rate": success_rate,
        "average_response_time_ms": (
            average_response_time_ms
        ),
        "last_response_time_ms": (
            current["last_response_time_ms"]
        ),
        "last_request_at": (
            current["last_request_at"]
        ),
    }


# ===================================================================
# INFORMATIONS INDEX FAISS
# ===================================================================

@app.get(
    "/index/info",
    tags=["Index"],
    summary="Retourne les informations de l'index FAISS",
)
def index_info() -> dict:
    """Inspecte l'index FAISS actuellement disponible."""

    index_file = (
        FAISS_INDEX_DIR
        / "index.faiss"
    )

    metadata_file = (
        FAISS_INDEX_DIR
        / "index.pkl"
    )

    if not index_file.exists():
        raise HTTPException(
            status_code=503,
            detail="Index FAISS indisponible.",
        )

    try:
        index = load_faiss_index()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible de charger "
                "l'index FAISS."
            ),
        ) from exc

    faiss_index = index.index

    dimension = getattr(
        faiss_index,
        "d",
        None,
    )

    return {
        "status": "available",
        "index_type": type(
            faiss_index
        ).__name__,
        "vectors": int(
            faiss_index.ntotal
        ),
        "dimension": (
            int(dimension)
            if dimension is not None
            else None
        ),
        "index_path": str(
            FAISS_INDEX_DIR
        ),
        "index_file_exists": (
            index_file.exists()
        ),
        "metadata_file_exists": (
            metadata_file.exists()
        ),
    }


# ===================================================================
# ÉVALUATION RAGAS
# ===================================================================

@app.get(
    "/evaluation/status",
    tags=["Evaluation"],
    summary="Retourne les dernières métriques Ragas",
)
def evaluation_status() -> dict:
    """Lit le dernier artefact d'évaluation Ragas."""

    if not RAGAS_RESULTS_PATH.exists():
        return {
            "status": "not_evaluated",
            "results_available": False,
            "path": str(
                RAGAS_RESULTS_PATH
            ),
            "generated_at": None,
            "summary": {},
            "scenario_count": 0,
            "scenarios": [],
        }

    try:
        payload = json.loads(
            RAGAS_RESULTS_PATH.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible de lire "
                "les résultats Ragas."
            ),
        ) from exc

    return {
        "status": "evaluated",
        "results_available": True,
        "path": str(
            RAGAS_RESULTS_PATH
        ),
        "generated_at": payload.get(
            "generated_at"
        ),
        "summary": payload.get(
            "summary",
            {},
        ),
        "scenario_count": payload.get(
            "scenario_count",
            0,
        ),
        "scenarios": payload.get(
            "scenarios",
            [],
        ),
    }


# ===================================================================
# REBUILD FAISS
# ===================================================================

@app.post(
    "/rebuild",
    tags=["Index"],
    summary=(
        "Lance la reconstruction de FAISS "
        "en arrière-plan"
    ),
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[
        Security(
            verify_rebuild_api_key
        ),
    ],
    responses={
        202: {
            "description": (
                "Reconstruction lancée "
                "en arrière-plan."
            ),
        },
        401: {
            "description": (
                "Clé d'accès absente "
                "ou invalide."
            ),
        },
        409: {
            "description": (
                "Une reconstruction "
                "est déjà en cours."
            ),
        },
        503: {
            "description": (
                "Configuration de sécurité "
                "indisponible."
            ),
        },
    },
)
def rebuild(
    background_tasks: BackgroundTasks,
) -> dict:
    """Programme une reconstruction FAISS."""

    with rebuild_lock:
        if rebuild_state["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Une reconstruction FAISS "
                    "est déjà en cours."
                ),
            )

        rebuild_state["status"] = "running"
        rebuild_state["started_at"] = utc_now()
        rebuild_state["completed_at"] = None
        rebuild_state["duration_seconds"] = None
        rebuild_state["error"] = None
        rebuild_state["progress"] = 0

        rebuild_state[
            "step"
        ] = (
            "Initialisation "
            "de la reconstruction FAISS"
        )

        rebuild_state["processed"] = None
        rebuild_state["total"] = None

    background_tasks.add_task(
        run_faiss_rebuild
    )

    return {
        "status": "accepted",
        "message": (
            "Reconstruction FAISS lancée "
            "en arrière-plan."
        ),
    }


# ===================================================================
# STATUT DU REBUILD FAISS
# ===================================================================

@app.get(
    "/rebuild/status",
    tags=["Index"],
    summary="Consulte l'état de la reconstruction FAISS",
)
def rebuild_status() -> dict:
    """Retourne l'état courant du rebuild FAISS."""

    with rebuild_lock:
        return dict(rebuild_state)