from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import scripts.rag_chain as rag_chain


class FakeRetriever:
    def invoke(self, question):
        return [
            Document(
                page_content="Concert de jazz à Paris",
                metadata={
                    "uid": 1,
                    "title": "Jazz Night",
                    "description": "Une soirée jazz.",
                    "city": "Paris",
                    "address": "141 Rue de Tolbiac",
                    "start_date": "2026-10-03T22:30:00+00:00",
                    "end_date": "2026-10-04T00:30:00+00:00",
                    "date_range": "Samedi 3 octobre à 22h30",
                    "keywords": ["Jazz", "Jam session"],
                    "chunk_index": 0,
                },
            )
        ]


def fake_llm_response(messages):
    return AIMessage(
        content=(
            "Je vous recommande Jazz Night, "
            "le 3 octobre 2026 à Paris."
        )
    )


def test_format_documents_contains_event_metadata():
    documents = [
        Document(
            page_content="Concert de jazz à Paris",
            metadata={
                "uid": 1,
                "title": "Jazz Night",
                "description": "Une soirée jazz.",
                "city": "Paris",
                "address": "141 Rue de Tolbiac",
                "start_date": "2026-10-03T22:30:00+00:00",
                "end_date": "2026-10-04T00:30:00+00:00",
                "date_range": "Samedi 3 octobre à 22h30",
                "keywords": ["Jazz", "Jam session"],
                "chunk_index": 0,
            },
        )
    ]

    context = rag_chain.format_documents(documents)

    assert "Jazz Night" in context
    assert "Une soirée jazz." in context
    assert "141 Rue de Tolbiac" in context
    assert "Paris" in context
    assert "Samedi 3 octobre à 22h30" in context
    assert "2026-10-03T22:30:00+00:00" in context
    assert "2026-10-04T00:30:00+00:00" in context
    assert "Jazz, Jam session" in context


def test_format_documents_handles_missing_metadata():
    documents = [
        Document(
            page_content="Événement incomplet",
            metadata={
                "title": "Événement test",
            },
        )
    ]

    context = rag_chain.format_documents(documents)

    assert "Événement test" in context
    assert "Description :" in context
    assert "Lieu :" in context
    assert "Date affichée :" in context
    assert "Début ISO :" in context
    assert "Fin ISO :" in context
    assert "Mots-clés :" in context


def test_format_documents_handles_multiple_events():
    documents = [
        Document(
            page_content="Premier événement",
            metadata={
                "title": "Jazz Night",
                "description": "Concert jazz",
                "city": "Paris",
                "address": "141 Rue de Tolbiac",
                "start_date": "2026-10-03T22:30:00+00:00",
                "end_date": "2026-10-04T00:30:00+00:00",
                "date_range": "Samedi 3 octobre",
                "keywords": ["Jazz"],
            },
        ),
        Document(
            page_content="Deuxième événement",
            metadata={
                "title": "Swing Session",
                "description": "Concert swing",
                "city": "Paris",
                "address": "141 Rue de Tolbiac",
                "start_date": "2026-11-07T20:00:00+00:00",
                "end_date": "2026-11-07T22:00:00+00:00",
                "date_range": "Samedi 7 novembre",
                "keywords": ["Swing"],
            },
        ),
    ]

    context = rag_chain.format_documents(documents)

    assert "Jazz Night" in context
    assert "Swing Session" in context
    assert "---" in context


def test_answer_question_uses_retriever_and_llm(monkeypatch):
    monkeypatch.setattr(
        rag_chain,
        "get_retriever",
        lambda k=5: FakeRetriever(),
    )

    fake_llm = RunnableLambda(fake_llm_response)

    monkeypatch.setattr(
        rag_chain,
        "get_llm",
        lambda: fake_llm,
    )

    response = rag_chain.answer_question(
        "Je cherche un concert de jazz à Paris."
    )

    assert "Jazz Night" in response
    assert "3 octobre 2026" in response
    assert "Paris" in response