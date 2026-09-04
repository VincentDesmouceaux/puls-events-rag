from scripts.evaluate_rag import evaluate_answer


def test_evaluate_answer_correct():
    result = evaluate_answer(
        "Voici un concert de jazz à Paris.",
        ["Paris", "jazz"],
    )

    assert result["score"] == 1.0
    assert result["label"] == "correcte"
    assert result["found_terms"] == ["Paris", "jazz"]


def test_evaluate_answer_partially_correct():
    result = evaluate_answer(
        "Voici un concert de jazz.",
        ["Paris", "jazz"],
    )

    assert result["score"] == 0.5
    assert result["label"] == "partiellement correcte"
    assert result["found_terms"] == ["jazz"]


def test_evaluate_answer_incorrect():
    result = evaluate_answer(
        "Aucun résultat correspondant.",
        ["Paris", "jazz"],
    )

    assert result["score"] == 0.0
    assert result["label"] == "incorrecte"
    assert result["found_terms"] == []
