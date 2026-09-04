from fastapi.testclient import TestClient

import app.main as main_module


client = TestClient(main_module.app)


def test_health_endpoint() -> None:
    """Vérifie que l'API répond correctement sur /health."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "puls-events-rag-api",
        "version": "0.1.0",
    }


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
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ask_missing_question() -> None:
    """Vérifie qu'une question absente est refusée."""
    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_ask_rate_limit_returns_503(monkeypatch) -> None:
    """Vérifie la gestion d'une limitation Mistral."""

    def fake_answer_question(question: str) -> str:
        raise RuntimeError("429 Rate limit exceeded")

    monkeypatch.setattr(
        main_module,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/ask",
        json={"question": "Un concert jazz ?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "Le service de génération est temporairement "
            "indisponible. Réessayez dans quelques instants."
        )
    }


def test_ask_internal_error_returns_500(monkeypatch) -> None:
    """Vérifie la gestion d'une erreur RAG interne."""

    def fake_answer_question(question: str) -> str:
        raise RuntimeError("Erreur interne simulée")

    monkeypatch.setattr(
        main_module,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/ask",
        json={"question": "Un événement à Paris ?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Impossible de générer une réponse RAG."
    }


def test_rebuild_without_key_returns_401(monkeypatch) -> None:
    """Vérifie que /rebuild refuse une requête sans clé."""
    monkeypatch.setenv(
        "REBUILD_API_KEY",
        "test-secret-key",
    )

    response = client.post("/rebuild")

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Clé d'accès invalide."
    }


def test_rebuild_with_invalid_key_returns_401(monkeypatch) -> None:
    """Vérifie que /rebuild refuse une mauvaise clé."""
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


def test_rebuild_success(monkeypatch) -> None:
    """Vérifie une reconstruction simulée avec une clé valide."""
    monkeypatch.setenv(
        "REBUILD_API_KEY",
        "test-secret-key",
    )

    calls = {
        "rebuild": False,
        "clear_cache": False,
    }

    def fake_rebuild() -> None:
        calls["rebuild"] = True

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

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "message": "Index FAISS reconstruit avec succès.",
    }

    assert calls["rebuild"] is True
    assert calls["clear_cache"] is True


def test_rebuild_internal_error_returns_500(monkeypatch) -> None:
    """Vérifie la gestion d'une erreur pendant le rebuild."""
    monkeypatch.setenv(
        "REBUILD_API_KEY",
        "test-secret-key",
    )

    def fake_rebuild() -> None:
        raise RuntimeError("Erreur rebuild simulée")

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

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Impossible de reconstruire l'index FAISS."
    }