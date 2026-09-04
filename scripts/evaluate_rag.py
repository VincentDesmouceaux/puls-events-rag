import json
from pathlib import Path

from scripts.rag_chain import answer_question


EVALUATION_FILE = Path("data/evaluation/rag_questions.json")


def load_questions() -> list[dict]:
    """Charge les scénarios d'évaluation RAG."""
    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate_terms(
    answer: str,
    expected_terms: list[str],
) -> dict:
    """Évalue la présence des termes attendus."""
    if not expected_terms:
        return {
            "score": 1.0,
            "found_terms": [],
        }

    normalized_answer = answer.lower()

    found_terms = [
        term
        for term in expected_terms
        if term.lower() in normalized_answer
    ]

    score = len(found_terms) / len(expected_terms)

    return {
        "score": score,
        "found_terms": found_terms,
    }


def evaluate_answer(
    answer: str,
    expected_terms: list[str],
    expected_facts: list[str] | None = None,
) -> dict:
    """Évalue une réponse avec les termes et faits attendus."""
    expected_facts = expected_facts or []

    terms_evaluation = evaluate_terms(
        answer,
        expected_terms,
    )

    facts_evaluation = evaluate_terms(
        answer,
        expected_facts,
    )

    if expected_facts:
        score = (
            terms_evaluation["score"]
            + facts_evaluation["score"]
        ) / 2
    else:
        score = terms_evaluation["score"]

    if score == 1:
        label = "correcte"
    elif score > 0:
        label = "partiellement correcte"
    else:
        label = "incorrecte"

    return {
        "score": score,
        "label": label,
        "found_terms": terms_evaluation["found_terms"],
        "fact_score": facts_evaluation["score"],
        "found_facts": facts_evaluation["found_terms"],
    }


def main() -> None:
    """Exécute les scénarios d'évaluation du chatbot RAG."""
    questions = load_questions()

    for item in questions:
        print("=" * 80)
        print(f"Scénario {item['id']}")
        print(f"Question : {item['question']}")

        answer = answer_question(
            item["question"]
        )

        evaluation = evaluate_answer(
            answer,
            item["expected_answer_contains"],
            item.get("expected_facts", []),
        )

        print("\nRéponse IA :")
        print(answer)

        print("\nRéponse de référence :")
        print(item["reference_answer"])

        print("\nÉvaluation :")
        print(
            f"Score global : "
            f"{evaluation['score']:.2f}"
        )
        print(
            f"Classement : "
            f"{evaluation['label']}"
        )
        print(
            f"Termes trouvés : "
            f"{evaluation['found_terms']}"
        )
        print(
            f"Score des faits : "
            f"{evaluation['fact_score']:.2f}"
        )
        print(
            f"Faits trouvés : "
            f"{evaluation['found_facts']}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()