from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

import scripts.fetch_openagenda as openagenda


def test_normalize_event():
    event = {
        "uid": 123,
        "title": {"fr": "Concert Jazz"},
        "description": {"fr": "Un concert à Paris"},
        "location": {
            "city": "Paris",
            "address": "10 rue de Paris",
            "latitude": 48.85,
            "longitude": 2.35,
        },
        "firstTiming": {
            "begin": "2026-09-04T19:00:00+02:00",
        },
        "lastTiming": {
            "end": "2026-09-04T22:00:00+02:00",
        },
        "dateRange": {"fr": "Vendredi 4 septembre"},
        "keywords": {"fr": ["Jazz", "Concert"]},
        "status": 1,
    }

    result = openagenda.normalize_event(event)

    assert result["uid"] == 123
    assert result["title"] == "Concert Jazz"
    assert result["description"] == "Un concert à Paris"
    assert result["city"] == "Paris"
    assert result["keywords"] == ["Jazz", "Concert"]


def test_normalize_event_handles_missing_fields():
    result = openagenda.normalize_event({})

    assert result["uid"] is None
    assert result["title"] == ""
    assert result["description"] == ""
    assert result["keywords"] == []


def test_filter_events_keeps_paris_recent_and_future_events():
    now = datetime.now(timezone.utc)

    df = pd.DataFrame(
        [
            {
                "title": "Paris récent",
                "city": "Paris",
                "start_date": (now - timedelta(days=30)).isoformat(),
            },
            {
                "title": "Paris futur",
                "city": "Paris",
                "start_date": (now + timedelta(days=30)).isoformat(),
            },
            {
                "title": "Paris trop ancien",
                "city": "Paris",
                "start_date": (now - timedelta(days=500)).isoformat(),
            },
            {
                "title": "Lyon",
                "city": "Lyon",
                "start_date": (now + timedelta(days=30)).isoformat(),
            },
        ]
    )

    result = openagenda.filter_events(df)

    assert len(result) == 2
    assert result["title"].tolist() == [
        "Paris récent",
        "Paris futur",
    ]


def test_add_embedding_text():
    df = pd.DataFrame(
        [
            {
                "title": "Concert Jazz",
                "description": "Concert live",
                "address": "10 rue de Paris",
                "city": "Paris",
                "date_range": "Vendredi soir",
                "keywords": ["Jazz", "Live"],
            }
        ]
    )

    result = openagenda.add_embedding_text(df)

    text = result.iloc[0]["embedding_text"]

    assert "Titre: Concert Jazz" in text
    assert "Description: Concert live" in text
    assert "Paris" in text
    assert "Jazz, Live" in text


def test_save_processed_events(tmp_path):
    output_file = tmp_path / "events.jsonl"

    df = pd.DataFrame(
        [
            {
                "uid": 1,
                "title": "Test event",
                "city": "Paris",
            }
        ]
    )

    openagenda.save_processed_events(
        df,
        str(output_file),
    )

    assert output_file.exists()

    loaded = pd.read_json(
        output_file,
        lines=True,
    )

    assert len(loaded) == 1
    assert loaded.iloc[0]["title"] == "Test event"


def test_add_mistral_embeddings(monkeypatch):
    class FakeEmbeddings:
        def create(self, model, inputs):
            assert model == "mistral-embed"

            data = [
                SimpleNamespace(
                    embedding=[0.1, 0.2, 0.3]
                )
                for _ in inputs
            ]

            return SimpleNamespace(data=data)

    class FakeMistral:
        def __init__(self, api_key):
            assert api_key == "fake-test-key"
            self.embeddings = FakeEmbeddings()

    monkeypatch.setenv(
        "MISTRAL_API_KEY",
        "fake-test-key",
    )

    monkeypatch.setattr(
        openagenda,
        "Mistral",
        FakeMistral,
    )

    df = pd.DataFrame(
        {
            "embedding_text": [
                "Premier événement",
                "Deuxième événement",
            ]
        }
    )

    result = openagenda.add_mistral_embeddings(
        df,
        batch_size=1,
    )

    assert len(result) == 2
    assert result.iloc[0]["embedding"] == [
        0.1,
        0.2,
        0.3,
    ]
    assert result.iloc[1]["embedding"] == [
        0.1,
        0.2,
        0.3,
    ]


def test_fetch_all_events_pagination(monkeypatch):
    responses = [
        {
            "events": [{"uid": 1}],
            "after": "next-page",
        },
        {
            "events": [{"uid": 2}],
            "after": None,
        },
    ]

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    call_count = 0

    def fake_get(*args, **kwargs):
        nonlocal call_count

        response = FakeResponse(
            responses[call_count]
        )

        call_count += 1
        return response

    monkeypatch.setenv(
        "OPENAGENDA_API_KEY",
        "fake-openagenda-key",
    )

    monkeypatch.setattr(
        openagenda.requests,
        "get",
        fake_get,
    )

    events = openagenda.fetch_all_events(
        20272888,
        size=100,
    )

    assert events == [
        {"uid": 1},
        {"uid": 2},
    ]

    assert call_count == 2
