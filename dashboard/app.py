import os

import httpx
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://p01--p9bot--wd4gpkqcrlm8.code.run",
)

REBUILD_API_KEY = os.getenv("REBUILD_API_KEY")


st.set_page_config(
    page_title="Puls-Events RAG",
    page_icon="🎵",
    layout="wide",
)

st.title("Puls-Events RAG")
st.caption(
    "Dashboard de démonstration et de monitoring "
    "du système de recommandation culturelle."
)


def get_api_health() -> dict:
    """Récupère l'état de santé de l'API FastAPI."""
    response = httpx.get(
        f"{API_BASE_URL}/health",
        timeout=5.0,
    )
    response.raise_for_status()

    return response.json()


def get_rebuild_status() -> dict:
    """Récupère l'état du rebuild FAISS."""
    response = httpx.get(
        f"{API_BASE_URL}/rebuild/status",
        timeout=5.0,
    )
    response.raise_for_status()

    return response.json()


def trigger_rebuild() -> dict:
    """Déclenche un rebuild FAISS protégé."""
    if not REBUILD_API_KEY:
        raise RuntimeError(
            "REBUILD_API_KEY is missing"
        )

    response = httpx.post(
        f"{API_BASE_URL}/rebuild",
        headers={
            "X-Rebuild-Key": REBUILD_API_KEY,
        },
        timeout=10.0,
    )
    response.raise_for_status()

    return response.json()


def ask_rag(question: str) -> dict:
    """Envoie une question au système RAG."""
    response = httpx.post(
        f"{API_BASE_URL}/ask",
        json={
            "question": question,
        },
        timeout=30.0,
    )
    response.raise_for_status()

    return response.json()


st.subheader("État de l'API")

try:
    health = get_api_health()

    st.success("API opérationnelle")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Status",
        health.get("status", "unknown"),
    )

    col2.metric(
        "Service",
        health.get("service", "unknown"),
    )

    col3.metric(
        "Version",
        health.get("version", "unknown"),
    )

except httpx.HTTPError as exc:
    st.error(
        "Impossible de contacter l'API FastAPI."
    )
    st.code(str(exc))

st.divider()

st.subheader("Index FAISS")

try:
    rebuild_status = get_rebuild_status()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rebuild",
        rebuild_status.get("status", "unknown"),
    )

    col2.metric(
        "Début",
        rebuild_status.get("started_at", "N/A"),
    )

    col3.metric(
        "Fin",
        rebuild_status.get("completed_at", "N/A"),
    )

    if rebuild_status.get("error"):
        st.error(
            rebuild_status["error"]
        )
    else:
        st.success(
            "Aucune erreur de reconstruction FAISS."
        )

except httpx.HTTPError as exc:
    st.error(
        "Impossible de récupérer l'état du rebuild FAISS."
    )
    st.code(str(exc))

if not REBUILD_API_KEY:
    st.warning(
        "REBUILD_API_KEY absente : "
        "le rebuild manuel est désactivé."
    )

if st.button(
    "Reconstruire l'index FAISS",
    disabled=not bool(REBUILD_API_KEY),
):
    try:
        result = trigger_rebuild()

        st.success(
            result.get(
                "message",
                "Rebuild FAISS lancé.",
            )
        )

    except (
        httpx.HTTPError,
        RuntimeError,
    ) as exc:
        st.error(
            "Impossible de lancer le rebuild FAISS."
        )
        st.code(str(exc))

st.divider()

st.subheader("Assistant culturel")

question = st.text_input(
    "Votre question",
    placeholder="Je cherche une soirée jazz à Paris.",
)

if st.button("Rechercher"):
    if not question.strip():
        st.warning(
            "Veuillez saisir une question."
        )
    else:
        with st.spinner(
            "Recherche dans les événements..."
        ):
            try:
                result = ask_rag(
                    question.strip()
                )

                st.markdown(
                    result.get(
                        "answer",
                        "Aucune réponse disponible.",
                    )
                )

            except httpx.HTTPError as exc:
                st.error(
                    "Erreur lors de l'appel au système RAG."
                )
                st.code(str(exc))

st.divider()

st.caption(
    f"API utilisée : {API_BASE_URL}"
)