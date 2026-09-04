from scripts.evaluate_rag import evaluate_answer


def test_evaluate_answer_correct():
    result = evaluate_answer(
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
    result = evaluate_answer(
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
    result = evaluate_answer(
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
    result = evaluate_answer(
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