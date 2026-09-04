import requests


API_URL = "http://127.0.0.1:8000"


def check_health() -> None:
    """Vérifie l'endpoint de santé de l'API."""
    response = requests.get(
        f"{API_URL}/health",
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    assert data["status"] == "ok"

    print("GET /health : OK")


def check_ask() -> None:
    """Vérifie l'endpoint RAG avec une vraie question."""
    payload = {
        "question": "Je cherche un événement jazz à Paris."
    }

    response = requests.post(
        f"{API_URL}/ask",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    assert data["question"] == payload["question"]
    assert data["answer"]

    print("POST /ask : OK")
    print("Réponse RAG :")
    print(data["answer"])


def check_rebuild() -> None:
    """Vérifie la reconstruction de l'index FAISS."""
    response = requests.post(
        f"{API_URL}/rebuild",
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    assert data["status"] == "ok"

    print("POST /rebuild : OK")
    print(data["message"])


def main() -> None:
    """Exécute les tests fonctionnels de l'API."""
    print("Test de l'API Puls-Events RAG")
    print("-" * 50)

    check_health()
    check_ask()
    check_rebuild()

    print("-" * 50)
    print("Tous les tests fonctionnels sont passés.")


if __name__ == "__main__":
    main()