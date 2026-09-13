from fastapi.testclient import TestClient

import app.main as main_module


client = TestClient(main_module.app)


def reset_rebuild_state() -> None:
    """Réinitialise l'état du rebuild entre les tests."""

    with main_module.rebuild_lock:
        main_module.rebuild_state["status"] = "idle"
        main_module.rebuild_state["started_at"] = None
        main_module.rebuild_state["completed_at"] = None
        main_module.rebuild_state["duration_seconds"] = None
        main_module.rebuild_state["error"] = None
        main_module.rebuild_state["progress"] = 0
        main_module.rebuild_state["step"] = "En attente"
        main_module.rebuild_state["processed"] = 0
        main_module.rebuild_state["total"] = 0


def test_health_endpoint() -> None:
    """Vérifie que l'API répond correctement sur /health."""

    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["service"] == "puls-events-rag-api"
    assert payload["version"] == "0.2.6"
    assert payload["timestamp"]


def test_ask_success(monkeypatch) -> None:
    """Vérifie une réponse RAG simulée sans appeler Mistral."""

    def fake_answer_question(question: str) -> str:
        return f"Réponse simulée pour : {question}"

    monkeypatch.setattr(
        main_module,
        "answer_question",
        fake_answer_question,
    )

    payload = {
        "question": "Je cherche un événement jazz à Paris."
    }

    response = client.post(
        "/ask",
        json=payload,
    )

    assert response.status_code == 200

    assert response.json() == {
        "question": payload["question"],
        "answer": (
            "Réponse simulée pour : "
            "Je cherche un événement jazz à Paris."
        ),
    }


def test_ask_empty_question() -> None:
    """Vérifie qu'une question vide est refusée."""

    response = client.post(
        "/ask",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 422


def test_ask_missing_question() -> None:
    """Vérifie qu'une question absente est refusée."""

    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_ask_rate_limit_returns_503(
    monkeypatch,
) -> None:
    """Vérifie la gestion d'une limitation Mistral."""

    def fake_answer_question(
        question: str,
    ) -> str:
        raise RuntimeError(
            "429 Rate limit exceeded"
        )

    monkeypatch.setattr(
        main_module,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/ask",
        json={
            "question": "Un concert jazz ?",
        },
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Le service de génération est temporairement "
            "indisponible. Réessayez dans quelques instants."
        )
    }


def test_ask_internal_error_returns_500(
    monkeypatch,
) -> None:
    """Vérifie la gestion d'une erreur RAG interne."""

    def fake_answer_question(
        question: str,
    ) -> str:
        raise RuntimeError(
            "Erreur interne simulée"
        )

    monkeypatch.setattr(
        main_module,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/ask",
        json={
            "question": "Un événement à Paris ?",
        },
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "Impossible de générer une réponse RAG."
        )
    }


def test_metrics_initial_state() -> None:
    """Vérifie que /metrics expose les statistiques."""

    response = client.get("/metrics")

    assert response.status_code == 200

    payload = response.json()

    assert "requests_total" in payload
    assert "requests_success" in payload
    assert "requests_failed" in payload
    assert "success_rate" in payload
    assert "average_response_time_ms" in payload
    assert "last_response_time_ms" in payload
    assert "last_request_at" in payload


def test_rebuild_without_key_returns_401(
    monkeypatch,
) -> None:
    """Vérifie que /rebuild refuse une requête sans clé."""

    reset_rebuild_state()

    monkeypatch.setenv(
        "REBUILD_API_KEY",
        "test-secret-key",
    )

    response = client.post(
        "/rebuild"
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Clé d'accès invalide."
    }


def test_rebuild_with_invalid_key_returns_401(
    monkeypatch,
) -> None:
    """Vérifie que /rebuild refuse une mauvaise clé."""

    reset_rebuild_state()

    monkeypatch.setenv(
        "REBUILD_API_KEY",
        "test-secret-key",
    )

    response = client.post(
        "/rebuild",
        headers={
            "X-Rebuild-Key": "mauvaise-cle",
        },
    )

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Clé d'accès invalide."
    }


def test_rebuild_without_configuration_returns_503(
    monkeypatch,
) -> None:
    """Vérifie le refus si la clé serveur n'est pas configurée."""

    reset_rebuild_state()

    monkeypatch.delenv(
        "REBUILD_API_KEY",
        raising=False,
    )

    response = client.post(
        "/rebuild",
        headers={
            "X-Rebuild-Key": "une-cle",
        },
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "La protection de l'endpoint rebuild "
            "n'est pas configurée."
        )
    }


def test_rebuild_success(
    monkeypatch,
) -> None:
    """Vérifie le lancement et la réussite du rebuild background."""

    reset_rebuild_state()

    monkeypatch.setenv(
        "REBUILD_API_KEY",
        "test-secret-key",
    )

    calls = {
        "rebuild": False,
        "clear_cache": False,
    }

    def fake_rebuild(
        progress_callback=None,
    ) -> None:
        calls["rebuild"] = True

        if progress_callback is not None:
            progress_callback(
                65,
                (
                    "Génération des embeddings "
                    "et construction FAISS"
                ),
                0,
                310,
            )

            progress_callback(
                100,
                "Reconstruction FAISS terminée",
                310,
                310,
            )

    def fake_clear_cache() -> None:
        calls["clear_cache"] = True

    monkeypatch.setattr(
        main_module,
        "rebuild_faiss_index",
        fake_rebuild,
    )

    monkeypatch.setattr(
        main_module,
        "clear_retriever_cache",
        fake_clear_cache,
    )

    response = client.post(
        "/rebuild",
        headers={
            "X-Rebuild-Key": "test-secret-key",
        },
    )

    assert response.status_code == 202

    assert response.json() == {
        "status": "accepted",
        "message": (
            "Reconstruction FAISS lancée "
            "en arrière-plan."
        ),
    }

    assert calls["rebuild"] is True
    assert calls["clear_cache"] is True

    status_response = client.get(
        "/rebuild/status"
    )

    assert status_response.status_code == 200

    data = status_response.json()

    assert data["status"] == "completed"
    assert data["started_at"] is not None
    assert data["completed_at"] is not None
    assert data["duration_seconds"] is not None
    assert data["error"] is None

    assert data["progress"] == 100
    assert data["step"] == (
        "Reconstruction FAISS terminée"
    )
    assert data["processed"] == 310
    assert data["total"] == 310


def test_rebuild_background_error_is_reported(
    monkeypatch,
) -> None:
    """Vérifie qu'une erreur background apparaît dans le statut."""

    reset_rebuild_state()

    monkeypatch.setenv(
        "REBUILD_API_KEY",
        "test-secret-key",
    )

    def fake_rebuild(
        progress_callback=None,
    ) -> None:
        if progress_callback is not None:
            progress_callback(
                65,
                (
                    "Génération des embeddings "
                    "et construction FAISS"
                ),
                0,
                310,
            )

        raise RuntimeError(
            "Erreur rebuild simulée"
        )

    monkeypatch.setattr(
        main_module,
        "rebuild_faiss_index",
        fake_rebuild,
    )

    response = client.post(
        "/rebuild",
        headers={
            "X-Rebuild-Key": "test-secret-key",
        },
    )

    assert response.status_code == 202

    status_response = client.get(
        "/rebuild/status"
    )

    assert status_response.status_code == 200

    data = status_response.json()

    assert data["status"] == "failed"
    assert data["started_at"] is not None
    assert data["completed_at"] is not None
    assert data["duration_seconds"] is not None
    assert data["error"] == (
        "Erreur rebuild simulée"
    )

    assert data["progress"] == 65
    assert data["step"] == (
        "Échec de la reconstruction FAISS"
    )
    assert data["processed"] == 0
    assert data["total"] == 310


def test_rebuild_status_idle() -> None:
    """Vérifie le statut initial du système de reconstruction."""

    reset_rebuild_state()

    response = client.get(
        "/rebuild/status"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "idle"
    assert payload["started_at"] is None
    assert payload["completed_at"] is None
    assert payload["duration_seconds"] is None
    assert payload["error"] is None

    assert payload["progress"] == 0
    assert payload["step"] == "En attente"
    assert payload["processed"] == 0
    assert payload["total"] == 0


def test_update_rebuild_progress() -> None:
    """Vérifie la mise à jour manuelle de la progression."""

    reset_rebuild_state()

    main_module.update_rebuild_progress(
        55,
        "Découpage des événements en chunks",
        155,
        310,
    )

    response = client.get(
        "/rebuild/status"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["progress"] == 55
    assert payload["step"] == (
        "Découpage des événements en chunks"
    )
    assert payload["processed"] == 155
    assert payload["total"] == 310


def test_rebuild_already_running_returns_409(
    monkeypatch,
) -> None:
    """Vérifie qu'un second rebuild simultané est refusé."""

    reset_rebuild_state()

    monkeypatch.setenv(
        "REBUILD_API_KEY",
        "test-secret-key",
    )

    with main_module.rebuild_lock:
        main_module.rebuild_state["status"] = (
            "running"
        )

        main_module.rebuild_state["started_at"] = (
            "2026-09-04T12:00:00+00:00"
        )

        main_module.rebuild_state["progress"] = 65

        main_module.rebuild_state["step"] = (
            "Génération des embeddings "
            "et construction FAISS"
        )

    response = client.post(
        "/rebuild",
        headers={
            "X-Rebuild-Key": "test-secret-key",
        },
    )

    assert response.status_code == 409

    assert response.json() == {
        "detail": (
            "Une reconstruction FAISS "
            "est déjà en cours."
        )
    }

    reset_rebuild_state()