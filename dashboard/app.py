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


def get_endpoint(path: str, timeout: float = 5.0) -> dict:
    """Appelle un endpoint GET de l'API FastAPI."""
    response = httpx.get(
        f"{API_BASE_URL}{path}",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_api_health() -> dict:
    """Récupère l'état de santé de l'API."""
    return get_endpoint("/health")


def get_metrics() -> dict:
    """Récupère les métriques runtime du système RAG."""
    return get_endpoint("/metrics")


def get_index_info() -> dict:
    """Récupère les informations de l'index FAISS."""
    return get_endpoint("/index/info", timeout=15.0)


def get_rebuild_status() -> dict:
    """Récupère l'état de la reconstruction FAISS."""
    return get_endpoint("/rebuild/status")


def get_evaluation_status() -> dict:
    """Récupère les résultats de l'évaluation Ragas."""
    return get_endpoint("/evaluation/status")


def trigger_rebuild() -> dict:
    """Déclenche une reconstruction FAISS protégée."""
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


def format_timestamp(value: str | None) -> str:
    """Rend un timestamp plus lisible dans le dashboard."""
    if not value:
        return "N/A"

    return value.replace("T", " ").replace("+00:00", " UTC")


overview_tab, assistant_tab, ragas_tab = st.tabs(
    [
        "Vue d'ensemble",
        "Assistant culturel",
        "Évaluation Ragas",
    ]
)


with overview_tab:
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

        st.caption(
            "Dernier contrôle : "
            f"{format_timestamp(health.get('timestamp'))}"
        )

    except httpx.HTTPError as exc:
        st.error("Impossible de contacter l'API FastAPI.")
        st.code(str(exc))

    st.divider()

    st.subheader("Monitoring RAG")

    try:
        metrics = get_metrics()

        total = metrics.get("requests_total", 0)
        success = metrics.get("requests_success", 0)
        failed = metrics.get("requests_failed", 0)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Requêtes",
            total,
        )
        col2.metric(
            "Succès",
            success,
        )
        col3.metric(
            "Échecs",
            failed,
        )
        col4.metric(
            "Taux de succès",
            f"{metrics.get('success_rate', 0.0):.1f} %",
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Temps moyen",
            f"{metrics.get('average_response_time_ms', 0.0):.0f} ms",
        )

        last_response_time = metrics.get(
            "last_response_time_ms"
        )

        col2.metric(
            "Dernière réponse",
            (
                f"{last_response_time:.0f} ms"
                if last_response_time is not None
                else "N/A"
            ),
        )

        st.caption(
            "Dernière requête : "
            f"{format_timestamp(metrics.get('last_request_at'))}"
        )

        if total > 0:
            st.bar_chart(
                {
                    "Requêtes": {
                        "Succès": success,
                        "Échecs": failed,
                    }
                }
            )
        else:
            st.info(
                "Aucune requête RAG enregistrée "
                "depuis le démarrage de l'API."
            )

        st.caption(
            "Ces métriques sont conservées en mémoire "
            "et sont réinitialisées au redémarrage de l'API."
        )

    except httpx.HTTPError as exc:
        st.error(
            "Impossible de récupérer les métriques RAG."
        )
        st.code(str(exc))

    st.divider()

    st.subheader("Index FAISS")

    try:
        index_info = get_index_info()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "État",
            index_info.get("status", "unknown"),
        )
        col2.metric(
            "Vecteurs",
            index_info.get("vectors", 0),
        )
        col3.metric(
            "Dimension",
            index_info.get("dimension", 0),
        )
        col4.metric(
            "Type",
            index_info.get("index_type", "unknown"),
        )

        st.caption(
            f"Index : {index_info.get('index_path', 'N/A')}"
        )

    except httpx.HTTPError as exc:
        st.error(
            "Impossible de récupérer les informations FAISS."
        )
        st.code(str(exc))

    st.divider()

    st.subheader("Reconstruction FAISS")

    try:
        rebuild_status = get_rebuild_status()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "État",
            rebuild_status.get("status", "unknown"),
        )
        col2.metric(
            "Début",
            format_timestamp(
                rebuild_status.get("started_at")
            ),
        )
        col3.metric(
            "Fin",
            format_timestamp(
                rebuild_status.get("completed_at")
            ),
        )

        duration = rebuild_status.get(
            "duration_seconds"
        )

        col4.metric(
            "Durée",
            (
                f"{duration:.2f} s"
                if duration is not None
                else "N/A"
            ),
        )

        if rebuild_status.get("error"):
            st.error(rebuild_status["error"])
        elif rebuild_status.get("status") == "completed":
            st.success(
                "Dernière reconstruction FAISS terminée."
            )
        else:
            st.info(
                "Aucune reconstruction FAISS en cours."
            )

    except httpx.HTTPError as exc:
        st.error(
            "Impossible de récupérer l'état "
            "de la reconstruction FAISS."
        )
        st.code(str(exc))

    if not REBUILD_API_KEY:
        st.warning(
            "REBUILD_API_KEY absente : "
            "la reconstruction manuelle est désactivée."
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
                    "Reconstruction FAISS lancée.",
                )
            )

            st.rerun()

        except (
            httpx.HTTPError,
            RuntimeError,
        ) as exc:
            st.error(
                "Impossible de lancer "
                "la reconstruction FAISS."
            )
            st.code(str(exc))


with assistant_tab:
    st.subheader("Assistant culturel")

    st.write(
        "Interrogez le système RAG sur les événements "
        "culturels présents dans l'index FAISS."
    )

    question = st.text_input(
        "Votre question",
        placeholder="Je cherche une soirée jazz à Paris.",
    )

    if st.button(
        "Rechercher",
        type="primary",
    ):
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

                    st.subheader("Réponse")
                    st.markdown(
                        result.get(
                            "answer",
                            "Aucune réponse disponible.",
                        )
                    )

                except httpx.HTTPError as exc:
                    st.error(
                        "Erreur lors de l'appel "
                        "au système RAG."
                    )
                    st.code(str(exc))


with ragas_tab:
    st.subheader("Évaluation Ragas")

    st.write(
        "Évaluation automatique de la qualité du système "
        "RAG sur un jeu de scénarios de référence."
    )

    try:
        evaluation = get_evaluation_status()

        if not evaluation.get("results_available"):
            st.warning(
                "Aucune évaluation Ragas disponible."
            )

        else:
            summary = evaluation.get(
                "summary",
                {},
            )

            scenario_count = evaluation.get(
                "scenario_count",
                0,
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Scénarios",
                scenario_count,
            )
            col2.metric(
                "Faithfulness",
                f"{summary.get('faithfulness', 0.0):.3f}",
            )
            col3.metric(
                "Answer relevancy",
                f"{summary.get('answer_relevancy', 0.0):.3f}",
            )
            col4.metric(
                "Context recall",
                f"{summary.get('context_recall', 0.0):.3f}",
            )

            st.caption(
                "Les scores Ragas sont compris entre 0 et 1. "
                "Une valeur élevée indique une meilleure qualité."
            )

            st.subheader("Scores globaux")

            global_scores = {
                "Faithfulness": summary.get(
                    "faithfulness",
                    0.0,
                ),
                "Answer relevancy": summary.get(
                    "answer_relevancy",
                    0.0,
                ),
                "Context recall": summary.get(
                    "context_recall",
                    0.0,
                ),
            }

            st.bar_chart(global_scores)

            scenarios = evaluation.get(
                "scenarios",
                [],
            )

            if scenarios:
                st.subheader(
                    "Résultats par scénario"
                )

                table_rows = []

                for scenario in scenarios:
                    table_rows.append(
                        {
                            "Question": scenario.get(
                                "user_input",
                                "",
                            ),
                            "Faithfulness": round(
                                scenario.get(
                                    "faithfulness",
                                    0.0,
                                ),
                                3,
                            ),
                            "Answer relevancy": round(
                                scenario.get(
                                    "answer_relevancy",
                                    0.0,
                                ),
                                3,
                            ),
                            "Context recall": round(
                                scenario.get(
                                    "context_recall",
                                    0.0,
                                ),
                                3,
                            ),
                        }
                    )

                st.dataframe(
                    table_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader(
                    "Comparaison des scénarios"
                )

                scenario_chart = {}

                for index, scenario in enumerate(
                    scenarios,
                    start=1,
                ):
                    scenario_chart[
                        f"Scénario {index}"
                    ] = {
                        "Faithfulness": scenario.get(
                            "faithfulness",
                            0.0,
                        ),
                        "Answer relevancy": scenario.get(
                            "answer_relevancy",
                            0.0,
                        ),
                        "Context recall": scenario.get(
                            "context_recall",
                            0.0,
                        ),
                    }

                st.bar_chart(scenario_chart)

            st.caption(
                "Résultats chargés depuis : "
                f"{evaluation.get('path', 'N/A')}"
            )

    except httpx.HTTPError as exc:
        st.error(
            "Impossible de récupérer "
            "les résultats Ragas."
        )
        st.code(str(exc))


st.divider()

st.caption(
    f"API utilisée : {API_BASE_URL}"
)