import os

import requests
from dotenv import load_dotenv


load_dotenv()

OPENAGENDA_API_URL = "https://api.openagenda.com/v2"


def fetch_agendas(size: int = 10) -> dict:
    """Récupère une liste d'agendas depuis l'API OpenAgenda."""

    api_key = os.getenv("OPENAGENDA_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAGENDA_API_KEY is missing")

    response = requests.get(
        f"{OPENAGENDA_API_URL}/agendas",
        headers={"key": api_key},
        params={"size": size},
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


if __name__ == "__main__":
    data = fetch_agendas(size=5)

    print(f"Agendas récupérés : {len(data.get('agendas', []))}")
    print(f"Total disponible : {data.get('total')}")