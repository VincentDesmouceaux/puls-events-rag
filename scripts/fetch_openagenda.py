import os
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone


load_dotenv()

OPENAGENDA_API_URL = "https://api.openagenda.com/v2"


def fetch_agendas(size: int = 10, search: str | None = None) -> dict:
    """Récupère une liste d'agendas depuis l'API OpenAgenda."""

    api_key = os.getenv("OPENAGENDA_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAGENDA_API_KEY is missing")

    params = {"size": size}

    if search:
        params["search"] = search

    response = requests.get(
        f"{OPENAGENDA_API_URL}/agendas",
        headers={"key": api_key},
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()

def fetch_events(agenda_uid: int, size: int = 20) -> dict:
    """Récupère les événements d'un agenda OpenAgenda."""

    api_key = os.getenv("OPENAGENDA_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAGENDA_API_KEY is missing")

    response = requests.get(
        f"{OPENAGENDA_API_URL}/agendas/{agenda_uid}/events",
        headers={"key": api_key},
        params={"size": size},
        timeout=30,
    )

    response.raise_for_status()

    return response.json()
def normalize_keywords(keywords) -> list[str]:
    """Normalise les mots-clés multilingues OpenAgenda."""

    if isinstance(keywords, dict):
        return keywords.get("fr") or keywords.get("en") or []

    if isinstance(keywords, list):
        return keywords

    return []
def normalize_event(event: dict) -> dict:
    """Transforme un événement OpenAgenda en structure simple."""

    title = event.get("title") or {}
    description = event.get("description") or {}
    date_range = event.get("dateRange") or {}
    location = event.get("location") or {}
    first_timing = event.get("firstTiming") or {}
    last_timing = event.get("lastTiming") or {}

    return {
        "uid": event.get("uid"),
        "title": title.get("fr") or title.get("en") or "",
        "description": description.get("fr") or description.get("en") or "",
        "city": location.get("city"),
        "address": location.get("address"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "start_date": first_timing.get("begin"),
        "end_date": last_timing.get("end"),
        "date_range": date_range.get("fr") or date_range.get("en") or "",
        "keywords": normalize_keywords (event.get ("keywords")),
        "status": event.get("status"),
    }

def events_to_dataframe(events: list[dict]) -> pd.DataFrame:
    """Normalise une liste d'événements OpenAgenda dans un DataFrame."""

    normalized_events = [normalize_event(event) for event in events]

    return pd.DataFrame(normalized_events)

if __name__ == "__main__":
    data = fetch_agendas(size=10, search="Paris")

    for agenda in data.get("agendas", []):
        print(
            f"UID: {agenda.get('uid')} | "
            f"Titre: {agenda.get('title')}"
        )
def filter_events(
    df: pd.DataFrame,
    city: str = "Paris",
) -> pd.DataFrame:
    """Filtre les événements par ville et sur 1 an d'historique + futur."""

    cutoff = datetime.now(timezone.utc) - timedelta(days=365)

    filtered = df.copy()

    filtered["start_date"] = pd.to_datetime(
        filtered["start_date"],
        utc=True,
        errors="coerce",
    )

    filtered = filtered[
        filtered["city"].eq(city)
        & filtered["start_date"].notna()
        & (filtered["start_date"] >= cutoff)
    ]

    return filtered.reset_index(drop=True)
def fetch_all_events(
    agenda_uid: int,
    size: int = 100,
) -> list[dict]:
    """Récupère tous les événements d'un agenda OpenAgenda."""

    events = []
    after = None

    while True:
        api_key = os.getenv("OPENAGENDA_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAGENDA_API_KEY is missing")

        params = {"size": size}

        if after:
            params["after"] = after

        response = requests.get(
            f"{OPENAGENDA_API_URL}/agendas/{agenda_uid}/events",
            headers={"key": api_key},
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        events.extend(data.get("events", []))

        after = data.get("after")

        if not after:
            break

    return events