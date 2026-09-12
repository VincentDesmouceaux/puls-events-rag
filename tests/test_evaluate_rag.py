import scripts.evaluate_rag as evaluate_rag


def test_evaluate_answer_correct():
    result = evaluate_rag.evaluate_answer(
        "Jazz Night a lieu à Paris au 141 Rue de Tolbiac en 2026.",
        ["Paris", "jazz"],
        ["Jazz Night", "141 Rue de Tolbiac", "2026"],
    )

    assert result["score"] == 1.0
    assert result["label"] == "correcte"
    assert result["found_terms"] == ["Paris", "jazz"]
    assert result["fact_score"] == 1.0
    assert result["found_facts"] == [
        "Jazz Night",
        "141 Rue de Tolbiac",
        "2026",
    ]


def test_evaluate_answer_partially_correct():
    result = evaluate_rag.evaluate_answer(
        "Voici un concert de jazz à Paris.",
        ["Paris", "jazz"],
        ["Jazz Night", "141 Rue de Tolbiac", "2026"],
    )

    assert result["score"] == 0.5
    assert result["label"] == "partiellement correcte"
    assert result["found_terms"] == ["Paris", "jazz"]
    assert result["fact_score"] == 0.0
    assert result["found_facts"] == []


def test_evaluate_answer_incorrect():
    result = evaluate_rag.evaluate_answer(
        "Aucun résultat correspondant.",
        ["Paris", "jazz"],
        ["Jazz Night", "141 Rue de Tolbiac", "2026"],
    )

    assert result["score"] == 0.0
    assert result["label"] == "incorrecte"
    assert result["found_terms"] == []
    assert result["fact_score"] == 0.0
    assert result["found_facts"] == []


def test_evaluate_answer_without_expected_facts():
    result = evaluate_rag.evaluate_answer(
        "Aucun événement pertinent n'a été trouvé.",
        ["aucun événement pertinent"],
        [],
    )

    assert result["score"] == 1.0
    assert result["label"] == "correcte"
    assert result["found_terms"] == [
        "aucun événement pertinent"
    ]
    assert result["fact_score"] == 1.0
    assert result["found_facts"] == []


def test_build_ragas_dataset_without_network(monkeypatch):
    questions = [
        {
            "id": 1,
            "question": "Je cherche un événement jazz à Paris.",
            "expected_answer_contains": [
                "Paris",
                "jazz",
            ],
            "expected_facts": [
                "TANA JAZZ NIGHT",
                "141 Rue de Tolbiac",
                "2026",
            ],
            "reference_answer": (
                "Je peux recommander TANA JAZZ NIGHT, "
                "une jam session jazz organisée au "
                "141 Rue de Tolbiac à Paris en 2026."
            ),
        }
    ]

    fake_rag_result = {
        "question": questions[0]["question"],
        "answer": (
            "TANA JAZZ NIGHT est un événement jazz "
            "au 141 Rue de Tolbiac à Paris en 2026."
        ),
        "retrieved_contexts": [
            (
                "Titre : TANA JAZZ NIGHT\n"
                "Description : Jam session jazz\n"
                "Lieu : 141 Rue de Tolbiac, Paris\n"
                "Début ISO : 2026-10-03T22:30:00+00:00"
            )
        ],
    }

    monkeypatch.setattr(
        evaluate_rag,
        "answer_question_with_context",
        lambda question: fake_rag_result,
    )

    dataset, generated_results = (
        evaluate_rag.build_ragas_dataset(questions)
    )

    assert len(dataset) == 1
    assert len(generated_results) == 1

    generated = generated_results[0]

    assert generated["id"] == 1
    assert generated["question"] == questions[0]["question"]
    assert generated["answer"] == fake_rag_result["answer"]
    assert (
        generated["reference_answer"]
        == questions[0]["reference_answer"]
    )
    assert (
        generated["retrieved_contexts"]
        == fake_rag_result["retrieved_contexts"]
    )

    assert generated["business_evaluation"]["score"] == 1.0
    assert (
        generated["business_evaluation"]["label"]
        == "correcte"
    )

    sample = dataset[0]

    assert sample.user_input == questions[0]["question"]
    assert sample.response == fake_rag_result["answer"]
    assert (
        sample.retrieved_contexts
        == fake_rag_result["retrieved_contexts"]
    )
    assert sample.reference == questions[0]["reference_answer"]

def test_save_ragas_results_without_network(tmp_path):
    import json

    import pandas as pd

    class FakeRagasResult:
        def to_pandas(self):
            return pd.DataFrame(
                [
                    {
                        "user_input": "Question 1",
                        "faithfulness": 1.0,
                        "answer_relevancy": 0.8,
                        "context_recall": 1.0,
                    },
                    {
                        "user_input": "Question 2",
                        "faithfulness": 0.8,
                        "answer_relevancy": 0.6,
                        "context_recall": 0.9,
                    },
                ]
            )

    output_path = tmp_path / "ragas_results.json"

    evaluate_rag.save_ragas_results(
        FakeRagasResult(),
        output_path=str(output_path),
    )

    assert output_path.exists()

    payload = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert payload["scenario_count"] == 2

    assert payload["summary"]["faithfulness"] == 0.9
    assert payload["summary"]["answer_relevancy"] == 0.7
    assert payload["summary"]["context_recall"] == 0.95

    assert len(payload["scenarios"]) == 2
    assert payload["scenarios"][0]["user_input"] == "Question 1"